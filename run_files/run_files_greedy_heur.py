import open3d as o3d
import numpy as np
import pandas as pd

from utils.custom_object_functions import create_hollow_room, get_arrow, create_coordinate_axes_mesh
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions
from utils.scene import Scene
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view
from utils.utils import get_project_data, get_project_root
from data_processing.csv_reader import extract_info
from models.sample_configurations import sample_voxel_directions

from models.mip_solver import MILPModel
import time

import pickle
import pathlib
import os


if __name__ == "__main__":

    project_path = get_project_root()

    data_path = get_project_data()
    scene_folder = data_path / "room_walls"
    scenes_list = os.listdir(scene_folder)

    batch_run_path = scene_folder

    data_columns = ["Folder Name", "Scene Name", "Sample Configurations",
                    "Samples per config per axis", "Camera Seed", "Camera Budget", "Optimization Type"]

    data_list = [
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 1, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 2, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 3, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 4, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 5, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 1, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 2, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 3, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 4, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 5, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 1, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 2, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 3, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 4, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 5, 4, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 1, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 2, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 3, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 4, 4, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 5, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 1, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 2, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 3, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 4, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 5, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 1, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 2, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 3, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 4, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 5, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 1, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 2, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 3, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 4, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 5, 2, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 1, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 2, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 3, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 4, 2, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 5, 2, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 1, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 2, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 3, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 4, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 5, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 1, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 2, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 3, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 4, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOA', 100, 2, 5, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 1, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 2, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 3, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 4, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 5, 8, "greedy"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 1, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 2, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 3, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 4, 8, "mip"],
        ["room_walls", 'W80H10D10VS1-0NW7S1111ZR0-8WOS', 100, 2, 5, 8, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 1, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 2, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 3, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 4, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 5, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 1, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 2, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 3, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 4, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOA', 100, 2, 5, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 1, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 2, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 3, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 4, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 5, 4, "greedy"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 1, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 2, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 3, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 4, 4, "mip"],
        ["room_walls", 'W40H10D10VS1-0NW3S1111ZR0-8WOS', 100, 2, 5, 4, "mip"],
                 ]

    batch_run = pd.DataFrame(data_list, columns=data_columns)
    # batch_run = pd.read_excel(scene_folder / "batch_run_mip.xlsx")

    all_data_list = []
    for index, row in batch_run.iterrows():
        scene_name = row["Scene Name"]
        (width, height, depth, voxel_size,
         num_walls, room_seed, wall_edge_ratio, wall_orient) = extract_info(scene_name)

        num_voxels = row["Sample Configurations"]
        num_directions_axis = row["Samples per config per axis"]

        camera_seed = row["Camera Seed"]
        camera_budget = row["Camera Budget"]

        optimization_type = row["Optimization Type"]

        # num_voxel_fraction = row["Configuration Count Fraction"]

        model_name = f"{scene_name}NC{camera_budget}CS{camera_seed}-MIP"

        model = Scene(filepath="None", obj_type="None")
        model.init_object = "mesh"

        model.pcd = o3d.io.read_point_cloud(str(scene_folder / scene_name / f'{scene_name}_pcd.pcd'))
        model.mesh = o3d.io.read_triangle_mesh(str(scene_folder / scene_name / f'{scene_name}_mesh.ply'))
        model.voxel_grid = o3d.io.read_voxel_grid(str(scene_folder / scene_name / f'{scene_name}_voxel_grid.ply'))
        model.free_space_points = np.load(scene_folder / scene_name / f'{scene_name}_free_space.npy', allow_pickle=True)
        model.voxel_tensor = np.load(scene_folder / scene_name / f'{scene_name}_voxel_tensor.npy', allow_pickle=True)

        model.voxel_grid_dim = model.calc_voxel_grid_dim()
        model.voxel_size = model.voxel_grid.voxel_size

        mip = MILPModel(scene=model, num_cameras=camera_budget)
        rng = np.random.default_rng(camera_seed)
        voxel_direction_dict = sample_voxel_directions(free_space_points=model.free_space_points, num_voxels=num_voxels,
                                                       num_points_axis=num_directions_axis, rng=rng,
                                                       # num_voxel_fraction=num_voxel_fraction
                                                       )

        start = time.time()
        mip.pre_process(voxel_directions_dict=voxel_direction_dict)
        processing_time = time.time() - start
        mip.create_model(model_name=model_name)
        if optimization_type.lower() == "mip":
            mip.optimize(max_run_time=3600, verbose=True)
        elif optimization_type.lower() == "greedy":
            mip.greedy_optimize()
        mip.post_process(save_solution=False, folder_path=scene_folder / scene_name)

        coverage = mip.ip_value / len(model.free_space_points)

        if optimization_type.lower() == "mip":
            runtime = mip.runtime
        elif optimization_type.lower() == "greedy":
            runtime = mip.greedy_runtime

        data_list = [width, height, depth, voxel_size, num_walls, room_seed, wall_edge_ratio, num_voxels,
                     num_directions_axis, camera_seed, camera_budget, model_name,

                     processing_time, runtime, len(model.free_space_points), mip.lp_value, mip.ip_value,
                     mip.best_dual_bound, mip.constr_num, mip.var_num, mip.node_count,
                     coverage,

                     ]

        all_data_list.append(data_list)

        batch_runs_solns = pd.DataFrame(all_data_list)
        batch_runs_solns.columns = ([
            "Width", "Height", "Depth", "Voxel Size", "No. Walls", "Room Seed", "Wall Edge ratio",
            "Sample Configurations", "Samples per config per axis", "Camera Seed", "Camera Budget", "Model Name",

            "Pre-processing Time", "Runtime", "Total Free Space", "LP Value", "IP Value", "Best Dual Bound",
            "Constraint Count", "Variable Count", "Nodes Traversed", "Coverage",
        ])

        # batch_runs_solns.round(3).to_excel(batch_run_path / f"batch_run_sols_std_mip_24-05-29.xlsx")
        batch_runs_solns.round(3).to_excel(get_project_root() / "analytics" / "analysis_set_september_2024"
                                           / f"batch_run_sols_std_greedy.xlsx")

        print(f"Index {index} completed")

    print(1)
