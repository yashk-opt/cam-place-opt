import numpy as np
from numba import jit
import open3d as o3d
import pickle
import gurobipy as gp
from gurobipy import GRB
from view_calculations import calculate_camera_view
from camera import Camera
from tqdm import tqdm
import ast
from scene_visualizer import create_voxels_subset, visualize_list
from custom_objects import create_coordinate_axes_mesh, get_arrow
import logging
import warnings
import sys


class MILPModel:
    def __init__(self, scene, voxel_directions, num_cameras=None):

        self.mipgap = None
        self.model = None
        self.lp_model = None

        self.lp_value = None
        self.ip_value = None
        self.runtime = None
        self.node_count = None
        self.var_num = None
        self.constr_num = None
        self.best_dual_bound = None

        self.scene_data = scene
        self.voxel_directions_dict = voxel_directions

        self.V = None
        self.PAbar = None
        self.P = None
        self.A_p = None

        self.V_pa = None
        self.PA_v = None
        self.C_pa = None
        self.K = num_cameras
        if num_cameras is not None:
            self.count_camera = True
        else:
            self.count_camera = False

        self.x_pa = None
        self.y_v = None

        self.xpa_indexset = None
        self.yv_indexset = None

        self.viewable_points_pa = None
        self.bounding_frustum_pa_c = None
        self.x_pa_solve = None

        pass

    def pre_process(self, warm_start=False, **kwargs):
        """ Create mapping for parameters, sets and decision variables """

        self.P = set(self.voxel_directions_dict.keys())
        self.V = {tuple(row) for row in self.scene_data.free_space_points}

        self.PAbar = {(key[0], key[1], key[2], *arr)
                      for key, arr in self.voxel_directions_dict.items()
                      for arr in arr.tolist()}

        if self.count_camera is True:
            self.C_pa = {key: 1 for key in self.PAbar}

        self.V_pa = dict()
        for s in tqdm(self.PAbar):
            cam = Camera()
            cam_center = tuple(np.array(s[:3]) + np.array(s[-3:]))
            cam.set_params(fov_deg=90, center=cam_center, eye=s[:3], width_px=640, height_px=480, up=(0, 1, 0))
            free_space_covered = calculate_camera_view(self.scene_data, cam)
            self.V_pa[s] = {tuple(row) for row in free_space_covered}

        self.PA_v = dict()
        for v in tqdm(self.V):
            self.PA_v[v] = set()
            for s in self.PAbar:
                if v in self.V_pa[s]:
                    self.PA_v[v].add(s)

        self.A_p = self.voxel_directions_dict

        self.xpa_indexset = list(self.PAbar)
        self.yv_indexset = self.V

    def create_model(self, model_name):
        """ Create Mixed Integer Model"""
        model = gp.Model(model_name)

        x_pa = model.addVars(self.xpa_indexset, vtype=GRB.BINARY, name="x")
        y_v = model.addVars(self.yv_indexset, vtype=GRB.BINARY, name="u", lb=0)

        # Objective 1
        obj = 0
        for v in self.V:
            obj += y_v[v]

        model.setObjective(obj, GRB.MAXIMIZE)

        # Constraint 2
        for v in self.V:
            lhs = y_v[v]
            rhs = 0
            for s in self.PA_v[v]:
                rhs += x_pa[s]

            model.addConstr(lhs <= rhs, name=f"2_{v}")

        # Constraint 3
        lhs = 0
        rhs = self.K
        for s in self.PAbar:
            lhs += self.C_pa[s] * x_pa[s]
        model.addConstr(lhs <= rhs, name=f"3")

        # Constraint 3
        for p in self.P:
            rhs = 1
            lhs = 0
            for a in self.A_p[p]:
                lhs += x_pa[(*p, *a)]
            model.addConstr(lhs <= rhs, name=f"4_{p}")

        self.model = model
        self.x_pa = x_pa
        self.y_v = y_v

    def optimize(self, max_run_time, log_path=None, verbose=False, **kwargs):

        set_mipgap = kwargs.get("set_mipgap", 0.01)
        self.model.setParam('MIPGap', set_mipgap)

        self.model.setParam('TimeLimit', max_run_time)

        if verbose is False:
            self.model.Params.LogToConsole = 0
        else:
            self.model.Params.LogToConsole = 1

        if log_path is not None:
            print(log_path)
            # self.model.write(str(log_path))
            # Set the log file parameter
            self.model.setParam(gp.GRB.Param.LogFile, str(log_path))

            # Set the LogToConsole parameter to 0 (false)
            self.model.setParam(gp.GRB.Param.LogToConsole, 1)

        self.model.optimize()

        self.mipgap = self.model.MIPGap
        self.lp_model = self.model.relax()
        self.lp_model.Params.LogToConsole = 0
        self.lp_model.optimize()

        self.lp_value = getattr(self.lp_model, "ObjVal", np.nan)
        self.ip_value = getattr(self.model, "ObjVal", np.nan)
        self.runtime = self.model.runtime
        self.node_count = self.model.NodeCount
        self.var_num = self.model.NumVars
        self.constr_num = self.model.NumConstrs
        self.best_dual_bound = getattr(self.model, "ObjBoundC", np.nan)

    def post_process(self):
        """ Organize solutions obtained from the MILP model """
        self.x_pa_solve = {tuple(ast.literal_eval(var.varName[1:])): var.X
                           for var in self.model.getVars()
                           if (var.varName.startswith("x") and var.X == 1)}

        viewable_points_pa = dict()
        bounding_frustum_pa_c = dict()
        for s in self.x_pa_solve.keys():
            cam = Camera()
            cam_center = tuple(np.array(s[:3]) + np.array(s[-3:]))
            cam.set_params(fov_deg=90, center=cam_center, eye=s[:3], width_px=640, height_px=480, up=(0, 1, 0))
            cam.calculate_bounding_frustum()
            viewable_points_pa[s] = calculate_camera_view(self.scene_data, cam)
            bounding_frustum_pa_c[s] = {
                "c11_unit_dir": cam.corner_11_dir,
                "c12_unit_dir": cam.corner_12_dir,
                "c21_unit_dir": cam.corner_21_dir,
                "c22_unit_dir": cam.corner_22_dir,
                "h1_unit_dir": cam.hor_1_dir,
                "h2_unit_dir": cam.hor_2_dir,
                "v1_unit_dir": cam.ver_1_dir,
                "v2_unit_dir": cam.ver_2_dir,
            }

        self.bounding_frustum_pa_c = bounding_frustum_pa_c
        self.viewable_points_pa = viewable_points_pa

    def visualize(self, view_solution=True, view_axes=True, view_invisible_points=True, cameras_all=True,
                  camera_num=[1]):
        """ Visualize the solution obtained """

        if cameras_all is True:
            camera_num = list(range(1, len(self.x_pa_solve) + 1))

        view_list = [self.scene_data.mesh]

        if view_invisible_points is True:
            unviewable_points = create_voxels_subset(self.scene_data.voxel_grid,
                                                     self.scene_data.free_space_points,
                                                     voxel_size=1 * self.scene_data.voxel_size / 10,
                                                     object_type="point", color="gold")
            view_list.append(unviewable_points)

        if view_solution is True:
            for i, s in enumerate(self.x_pa_solve.keys()):
                if i + 1 in camera_num:
                    camera_position = create_voxels_subset(self.scene_data.voxel_grid,
                                                           np.array(s[:3]).reshape(1, -1),
                                                           voxel_size=self.scene_data.voxel_size,
                                                           object_type="point", color="black")

                    view_list.append(camera_position)

                    viewable_points = create_voxels_subset(self.scene_data.voxel_grid,
                                                           self.viewable_points_pa[s],
                                                           voxel_size= self.scene_data.voxel_size / 4,
                                                           object_type="point", color="deep_pink")
                    view_list.append(viewable_points)

                    for key in self.bounding_frustum_pa_c[s].keys():
                        # Creating arrows which are twice the size of a voxel
                        arrow = get_arrow(np.array(s[:3]) + 2 * self.bounding_frustum_pa_c[s][key], np.array(s[:3]))
                        if key.startswith("c"):
                            arrow.paint_uniform_color(np.array([0, 0, 0]))
                        elif key.startswith("h"):
                            arrow.paint_uniform_color(np.array([1, 0, 0]))
                        elif key.startswith("v"):
                            arrow.paint_uniform_color(np.array([0, 1, 0]))
                        else:
                            pass
                        view_list.append(arrow)

        if view_axes is True:
            axes = create_coordinate_axes_mesh(np.array([0, 0, 0]))
            view_list += list(axes)

        visualize_list(view_list)
