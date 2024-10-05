import open3d as o3d
import numpy as np
import pandas as pd

from utils.custom_object_functions import (create_hollow_room, get_arrow, create_coordinate_axes_mesh, develop_mesh_line_set,
                                           filter_lines_parallel_to_axes, extract_voxel_linesets)
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions
from utils.scene import Scene
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view, calculate_camera_view_v2
from utils.utils import get_project_data, get_project_root
from utils.array_operations import arrays_within_tolerance_complete, remove_similar_rows
from data_processing.csv_reader import extract_info
from models.sample_configurations import sample_voxel_directions, sample_explore_exploit
from data_processing.voxel_calculations import voxel_block_count
from models.mip_solver import MILPModel

import pickle
import pathlib
import os
import time
import json
import copy

if __name__ == "__main__":

    root_path = get_project_root()
    data_path = get_project_data()
    scene_folder = data_path / "apartment_0_simple"
    scenes_list = os.listdir(scene_folder)
    seed_folder = root_path / "analytics" / "case_study_runs" / "algo_sols_su_7_True"

    data_columns = ['Folder Name', 'Scene Name', 'Sample Voxels Per Iteration',
                    'Angles per Voxel per Axis', 'Camera Seed', 'Camera Budget',
                    'Exploit Fraction', 'Voxel Perturbation Allowance',
                    'Angle Perturbation Allowance', 'Angle Selection', 'Number of Iterations']

    batch_run = pd.read_excel(root_path / "analytics" / "analysis_set_september_2024"
                              / "model-runs-apartment_0-su.xlsx")

    floodfill_data_folder = root_path / "analytics" / "algo-floodfill-visualization"

    scene_name = "apartment_0_simple"
    camera_budget = batch_run.iloc[1]["Camera Budget"]
    # camera_budget = 6
    block_size = batch_run.iloc[1]["Super Voxel Size"]
    num_iterations = batch_run.iloc[1]["Number of Iterations"]

    model = Scene(filepath="None", obj_type="None")
    model.init_object = "mesh"

    # model.pcd = o3d.io.read_point_cloud(str(scene_folder / scene_name / f'{scene_name}_pcd.pcd'))
    model.mesh = o3d.io.read_triangle_mesh(str(scene_folder / f'{scene_name}_mesh.ply'))
    model.voxel_grid = o3d.io.read_voxel_grid(str(scene_folder / f'{scene_name}_voxel_grid.ply'))
    model.free_space_points = np.load(scene_folder / f'{scene_name}_free_space.npy', allow_pickle=True)
    model.voxel_tensor = np.load(scene_folder / f'{scene_name}_voxel_tensor.npy', allow_pickle=True)

    model.voxel_grid_dim = model.calc_voxel_grid_dim()
    model.voxel_size = model.voxel_grid.voxel_size

    # check_list = [model.mesh]
    json_path = str(pathlib.Path.cwd() / "DepthCamera_2024-09-25-10-34-59.json")

    x_pa_solve = {}
    x_pa = {}
    camera_voxel = {}
    viewable_points = {}
    viewable_points_vis = {}
    unviewable_points_vis = {}

    rng = np.random.default_rng(1)

    temp_voxel_dict = sample_voxel_directions(free_space_points=model.free_space_points,
                                              num_voxels=1,
                                              num_points_axis=2,
                                              directions_per_voxel="same",
                                              direction_selection_type="uniform",
                                              rng=rng)

    direction_set = list(temp_voxel_dict.values())[0]

    iteration = 10
    camera_voxel[iteration] = {}
    with open(seed_folder / f'algo_sols_solve_{iteration}.pkl', 'rb') as file:
        x_pa_solve[iteration] = list(pickle.load(file))

    for count in range(camera_budget):

        camera_orient = x_pa_solve[iteration][count]

        # load camera voxels
        camera_voxel[iteration][count + 1] = create_voxels_subset(model.voxel_grid,
                                                                  np.array(x_pa_solve[iteration][count][:3]).reshape(1, -1),
                                                                  voxel_size=model.voxel_size / 2,
                                                                  object_type="point", color="black")

        break

    floodfill_iter = 10
    view_list = [model.mesh]

    stack_voxels = np.load(floodfill_data_folder / f'stack_{floodfill_iter}.npy', allow_pickle=True)
    view_set_voxels = np.load(floodfill_data_folder / f'view_set_{floodfill_iter}.npy', allow_pickle=True)

    stack_voxels_vis = create_voxels_subset(model.voxel_grid,
                                            stack_voxels,
                                            voxel_size=1 * model.voxel_size / 10,
                                            object_type="point", color="indian_red")

    view_set_vis = create_voxels_subset(model.voxel_grid,
                                        view_set_voxels,
                                        voxel_size=1 * model.voxel_size / 5,
                                        object_type="point", color="royal_blue")

    pa = x_pa_solve[iteration][count]
    arrow = get_arrow(end=0.8 * np.array(pa[3:]) + np.array(pa[:3]), origin=np.array(pa[:3]))
    arrow.paint_uniform_color(np.array([0, 0, 0]) / 255)

    view_list.append(camera_voxel[iteration][count + 1])
    view_list.append(arrow)
    view_list.append(stack_voxels_vis)
    view_list.append(view_set_vis)
    view_list.append(arrow)

    visualize_list(view_list, json_path=json_path)
    print(1)

    for count in range(camera_budget):

        camera_orient = x_pa_solve[iteration][count]

        # load camera voxels
        camera_voxel[iteration][count + 1] = create_voxels_subset(model.voxel_grid,
                                                                  np.array(x_pa_solve[iteration][count][:3]).reshape(1, -1),
                                                                  voxel_size=model.voxel_size / 2,
                                                                  object_type="point", color="black")

        camera = Camera()
        cam_center = tuple(np.array(camera_orient[:3]) + np.array(camera_orient[-3:]))
        camera.set_params(fov_deg=90, center=cam_center, eye=camera_orient[:3], width_px=640, height_px=480,
                          up=(0, 1, 0))

        calculate_camera_view_v2(model, camera)

        # calculate camera network viewable points
        if iteration in viewable_points.keys():
            viewable_points[iteration] = np.vstack((viewable_points[iteration], calculate_camera_view(model, camera)))
