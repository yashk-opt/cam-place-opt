import open3d as o3d
import numpy as np
import pandas as pd

from utils.custom_object_functions import create_hollow_room, get_arrow, create_coordinate_axes_mesh
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions, sample_voxel_directions
from utils.scene import Scene
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view
from data_processing.csv_reader import extract_info

from models.mip_solver import MILPModel
import time

import pickle
import pathlib
import os


if __name__ == "__main__":

    data_path = pathlib.Path.cwd().parent / "data"
    scene_folder = data_path / "room_walls"
    scenes_list = os.listdir(scene_folder)

    batch_run_path = scene_folder
    batch_run = pd.read_excel(scene_folder / "batch_run.xlsx").iloc[1:]

    # solution_name = "W20-0H10-0D10-0VS1-0NW2S3R1-0NC4CS618"
    solution_name = "W20-0H10-0D10-0VS1-0NW2S3R1-0NC3CS792"

    scene_name = solution_name.split("NC")[0]
    width, height, depth, voxel_size, num_walls, room_seed, wall_edge_ratio = extract_info(scene_name)

    camera_seed = int(solution_name.split("CS")[1])

    sub1 = "NC"
    sub2 = "CS"

    test_str = solution_name.replace(sub1, "*")
    test_str = test_str.replace(sub2, "*")
    camera_budget = int(test_str.split("*")[1])

    model_name = f"{scene_name}NC{camera_budget}CS{camera_seed}"

    model = Scene(filepath="None", obj_type="None")
    model.init_object = "mesh"

    model.pcd = o3d.io.read_point_cloud(str(scene_folder / scene_name / f'{scene_name}_pcd.pcd'))
    model.mesh = o3d.io.read_triangle_mesh(str(scene_folder / scene_name / f'{scene_name}_mesh.ply'))
    model.voxel_grid = o3d.io.read_voxel_grid(str(scene_folder / scene_name / f'{scene_name}_voxel_grid.ply'))
    model.free_space_points = np.load(scene_folder / scene_name / f'{scene_name}_free_space.npy', allow_pickle=True)
    model.voxel_tensor = np.load(scene_folder / scene_name / f'{scene_name}_voxel_tensor.npy', allow_pickle=True)

    model.voxel_grid_dim = model.calc_voxel_grid_dim()
    model.voxel_size = model.voxel_grid.voxel_size

    # voxel_direction_dict = sample_voxel_directions(free_space_points=model.free_space_points, num_voxels=num_voxels,
    #                                                num_points_axis=num_directions_axis, seed=camera_seed)

    mip = MILPModel(scene=model, voxel_directions=None, num_cameras=camera_budget)
    # start = time.time()
    # mip.pre_process()
    # processing_time = time.time() - start
    # mip.create_model(model_name=model_name)
    # mip.optimize(max_run_time=3600, verbose=True)
    mip.post_process(save_solution=False, load_solution_path=str(scene_folder / scene_name / f'{solution_name}.pickle'))
    mip.visualize(view_solution=True, view_axes=True, view_invisible_points=True, cameras_all=True)

    print(1)
