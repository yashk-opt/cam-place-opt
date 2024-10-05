import open3d as o3d
import numpy as np
import pandas as pd

from utils.custom_object_functions import (create_hollow_room, get_arrow, create_coordinate_axes_mesh, develop_mesh_line_set,
                                           filter_lines_parallel_to_axes)
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions
from utils.scene import Scene
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view
from utils.utils import get_project_data, get_project_root
from utils.array_operations import arrays_within_tolerance_complete, remove_similar_rows
from data_processing.csv_reader import extract_info
from models.sample_configurations import sample_voxel_directions, sample_explore_exploit

from models.mip_solver import MILPModel
import time

import pickle
import pathlib
import os


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

    scene_name = "apartment_0_simple"
    camera_budget = batch_run.iloc[1]["Camera Budget"]
    # camera_budget = 6
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

    unviewable_points = create_voxels_subset(model.voxel_grid,
                                             model.free_space_points,
                                             voxel_size=1 * model.voxel_size / 20,
                                             object_type="point", color="gold")

    x_pa_solve = {}
    x_pa = {}
    camera_voxel = {}
    viewable_points = {}
    viewable_points_vis = {}

    rng = np.random.default_rng(1)

    temp_voxel_dict = sample_voxel_directions(free_space_points=model.free_space_points,
                                              num_voxels=1,
                                              num_points_axis=2,
                                              directions_per_voxel="same",
                                              direction_selection_type="uniform",
                                              rng=rng)

    direction_set = list(temp_voxel_dict.values())[0]

    explore_voxels = {}
    exploit_voxels = {}

    voxel_direction_dict = {}

    for iteration in range(1, num_iterations + 1):

        # load data
        with open(seed_folder / f'algo_sols_solve_{iteration}.pkl', 'rb') as file:
            x_pa_solve[iteration] = list(pickle.load(file))
        with open(seed_folder / f'algo_sols_all_{iteration}.pkl', 'rb') as file:
            x_pa[iteration] = list(pickle.load(file))

        # load camera data
        camera_voxel[iteration] = {}
        for count in range(camera_budget):

            camera_orient = x_pa_solve[iteration][count]

            # load camera voxels
            camera_voxel[iteration][count + 1] = create_voxels_subset(model.voxel_grid,
                                                                      np.array(x_pa_solve[iteration][count][:3]).reshape(1, -1),
                                                                      voxel_size=model.voxel_size / 2,
                                                                      object_type="point", color="black")

            camera = Camera()
            cam_center = tuple(np.array(camera_orient[:3]) + np.array(camera_orient[-3:]))
            camera.set_params(fov_deg=90, center=cam_center, eye=camera_orient[:3], width_px=640, height_px=480, up=(0, 1, 0))

            # calculate camera network viewable points
            if iteration in viewable_points.keys():
                viewable_points[iteration] = np.vstack((viewable_points[iteration], calculate_camera_view(model, camera)))
            else:
                viewable_points[iteration] = calculate_camera_view(model, camera)

        # voxelize camera network viewable points
        viewable_points_vis[iteration] = create_voxels_subset(model.voxel_grid, viewable_points[iteration],
                                                              voxel_size=1 * model.voxel_size / 10,
                                                              object_type="point")

        # convert config data into voxel direction dict data
        voxel_direction_dict[iteration] = {}
        for config in x_pa[iteration]:
            if tuple(config[:3]) not in voxel_direction_dict[iteration].keys():
                voxel_direction_dict[iteration][tuple(config[:3])] = np.array(config[3:]).reshape(1, -1)

            else:
                voxel_direction_dict[iteration][tuple(config[:3])] = (
                    np.vstack((voxel_direction_dict[iteration][tuple(config[:3])], np.array(config[3:]).reshape(1, -1)))
                )

        explore_voxels[iteration] = []
        for voxel in voxel_direction_dict[iteration].keys():
            if len(arrays_within_tolerance_complete(voxel_direction_dict[iteration][voxel],
                                                 direction_set, tolerance=0.01)) == 8:
                explore_voxels[iteration].append(voxel)

            if iteration != 1:
                exploit_voxels[iteration] = {}
                for voxel in voxel_direction_dict[iteration].keys():

                    common_direction_set = arrays_within_tolerance_complete(voxel_direction_dict[iteration][voxel],
                                                                            direction_set, tolerance=0.01)
                    excluded_direction_set = remove_similar_rows(voxel_direction_dict[iteration][voxel], direction_set)

                    if (
                            (len(common_direction_set) != 8)
                            or
                            (len(voxel_direction_dict[iteration][voxel]) > 8)
                    ):

                        if voxel in exploit_voxels[iteration].keys():
                            exploit_voxels[iteration][voxel] = np.vstack(
                                (exploit_voxels[iteration][voxel], excluded_direction_set)
                            )

                        else:
                            exploit_voxels[iteration][voxel] = excluded_direction_set

    # Visualization of elements
    iteration = 10
    view_list = [model.mesh]

    for voxel in explore_voxels[iteration]:

        view_list.append(create_voxels_subset(model.voxel_grid,
                                              np.array(voxel).reshape(1, -1),
                                              voxel_size=1 * model.voxel_size / 3,
                                              color="lime_green", object_type="point")
                         )

        for direction in direction_set:
            arrow = get_arrow(end=0.5 * direction + voxel, origin=voxel)
            arrow.paint_uniform_color(np.array([50, 205, 50])/255)
            view_list.append(arrow)

    if iteration != 1:
        for voxel in exploit_voxels[iteration].keys():
            view_list.append(create_voxels_subset(model.voxel_grid,
                                                  np.array(voxel).reshape(1, -1),
                                                  voxel_size=1 * model.voxel_size / 4,
                                                  color="orange_red", object_type="point")
                             )
            for direction in exploit_voxels[iteration][voxel]:
                arrow = get_arrow(end=0.5 * direction + voxel, origin=voxel)
                arrow.paint_uniform_color(np.array([255, 69, 0])/255)
                view_list.append(arrow)

    # line_set = filter_lines_parallel_to_axes(develop_mesh_line_set(model.mesh))
    # view_list.append(line_set)

    for count in range(1, camera_budget + 1):
        view_list.append(camera_voxel[iteration][count])

        pa = x_pa_solve[iteration][count - 1]
        arrow = get_arrow(end=0.8 * np.array(pa[3:]) + np.array(pa[:3]), origin=np.array(pa[:3]))
        arrow.paint_uniform_color(np.array([0, 0, 0]) / 255)
        view_list.append(arrow)

        if iteration != 1:

            pa_1 = x_pa_solve[iteration - 1][count - 1]

            cam_vox = create_voxels_subset(model.voxel_grid, np.array(pa_1[:3]).reshape(1, -1),
                                           voxel_size=0.45 * model.voxel_size,
                                           object_type="point",
                                           color=np.array([15, 69, 57]) / 255)
            view_list.append(cam_vox)

            arrow = get_arrow(end=0.7 * np.array(pa_1[3:]) + np.array(pa_1[:3]), origin=np.array(pa_1[:3]))
            arrow.paint_uniform_color(np.array([15, 69, 57]) / 255)
            view_list.append(arrow)

    view_list.append(viewable_points_vis[iteration])
    view_list.append(unviewable_points)
    visualize_list(view_list)
    print(1)

    # a dictionary for each camera's position, angle and set of voxels

    camera_voxel_dict = {}
    for camera_orient in x_pa_solve[iteration]:

        camera = Camera()
        cam_center = tuple(np.array(camera_orient[:3]) + np.array(camera_orient[-3:]))
        camera.set_params(fov_deg=90, center=cam_center, eye=camera_orient[:3], width_px=640, height_px=480,
                          up=(0, 1, 0))

        new_check = set(tuple(sublist) for sublist in np.round(calculate_camera_view(model, camera)).astype(int))

        # camera_voxel_dict[camera_orient] = np.array(list(new_check))
        camera_voxel_dict[camera_orient] = new_check

    unique_cam_dict = {}
    cam_list = list(camera_voxel_dict.keys())
    for i, cam in enumerate(cam_list):
        cam_set = camera_voxel_dict[cam]
        cam_set2 = set()
        for j, cam2 in enumerate(cam_list):
            # cam_set2 = np.empty((0, 3))
            if i != j:
                print(i, j)
                # cam_set2 = np.vstack([cam_set2, camera_voxel_dict[cam2]])
                cam_set2 = cam_set2 | camera_voxel_dict[cam2]
            # cam_set2 = np.round(cam_set2)

        unique_cam_dict[cam] = cam_set - cam_set2
        # remove_similar_rows(cam_set, cam_set2, tolerance=10**-2)

    new_view_list = [model.mesh]
    new_view_list.append(create_voxels_subset(model.voxel_grid,
                         np.array(x_pa_solve[iteration][0][:3]).reshape(1, -1),
                         voxel_size=model.voxel_size / 2,
                         object_type="point", color="black"))
    visualize_list(new_view_list)


    print("Completed Solution: 1")
