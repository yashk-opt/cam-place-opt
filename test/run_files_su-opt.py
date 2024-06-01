import open3d as o3d
import numpy as np
import pandas as pd

from utils.custom_object_functions import create_hollow_room, get_arrow, create_coordinate_axes_mesh
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions
from utils.scene import Scene
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view
from data_processing.csv_reader import extract_info
from models.sample_configurations import sample_voxel_directions, sample_explore_exploit, sample_uncovered_voxels
from data_processing.voxel_calculations import voxel_block_count

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
    batch_run = pd.read_excel(scene_folder / "batch_run_su.xlsx")

    all_data_list = []
    for index, row in batch_run.iterrows():
        scene_name = row["Scene Name"]
        width, height, depth, voxel_size, num_walls, room_seed, wall_edge_ratio = extract_info(scene_name)

        num_voxels = row["Sample Configurations"]
        num_points_axis = row["Samples per config per axis"]

        camera_seed = row["Camera Seed"]
        camera_budget = row["Camera Budget"]



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

        num_iterations = row["Number of Iterations"]
        block_size = row["Super Voxel Size"]
        uncovered_search_fraction = row["Uncovered Search Fraction"]
        explore_fraction = 1 - uncovered_search_fraction

        model_name = f"{scene_name}NC{camera_budget}CS{camera_seed}NI{num_iterations}SV{block_size}US{uncovered_search_fraction}"

        rng = np.random.default_rng(camera_seed)
        num_voxels = round(num_voxels / num_iterations)

        preprocessing_time_list = []
        mip_runtime_list = []
        coverage_list = []
        lp_coverage_list = []

        for iteration in range(1, num_iterations + 1):

            if iteration == 1:
                voxel_direction_dict = sample_voxel_directions(free_space_points=model.free_space_points,
                                                               num_voxels=num_voxels,
                                                               num_points_axis=num_points_axis, rng=rng)
                start = time.time()
                mip.pre_process(voxel_directions_dict=voxel_direction_dict)

            else:
                voxel_info = mip.extract_voxel_info(type_info="uncovered")
                voxel_collection_count, large_grid_info = voxel_block_count(model, voxel_info, block_size)

                voxel_direction_dict = sample_uncovered_voxels(eff_num_points=num_voxels,
                                                               num_points_axis=num_points_axis,
                                                               free_space_points=model.free_space_points,
                                                               large_grid_data=voxel_collection_count,
                                                               large_grid_params=large_grid_info, rng=rng,
                                                               uncovered_search_fraction=uncovered_search_fraction,
                                                               random_search_fraction=explore_fraction,
                                                               solutions=mip.x_pa_solve)

                start = time.time()
                mip.pre_process(voxel_directions_dict=voxel_direction_dict, update_existing=True)

            processing_time = time.time() - start
            mip.create_model(model_name=model_name)
            mip.optimize(max_run_time=3600, verbose=True)
            if iteration % 100 == 0:
                mip.post_process(save_solution=True, folder_path=scene_folder / scene_name, name_suffix=f"-{iteration}")

            else:
                mip.post_process(save_solution=False, folder_path=scene_folder / scene_name,
                                 name_suffix=f"-{iteration}")

            preprocessing_time_list.append(round(processing_time, 3))
            mip_runtime_list.append(round(mip.runtime, 3))
            coverage_list.append(round(mip.ip_value / len(model.free_space_points), 2))
            lp_coverage_list.append(round(mip.lp_value / len(model.free_space_points), 2))

        total_preprocess_time = sum(preprocessing_time_list)
        total_mip_runtime = sum(mip_runtime_list)
        total_coverage = coverage_list[-1]

        mip.post_process(save_solution=True, folder_path=scene_folder / scene_name, name_suffix=f"-final")

        data_list = [width, height, depth, voxel_size, num_walls, room_seed, wall_edge_ratio, num_voxels,
                     num_points_axis, camera_seed, camera_budget, model_name,

                     processing_time, mip.runtime, len(model.free_space_points), mip.lp_value, mip.ip_value,
                     mip.best_dual_bound, mip.constr_num, mip.var_num, mip.node_count,

                     block_size, uncovered_search_fraction, num_iterations,

                     total_coverage,
                     f"{preprocessing_time_list}".replace('[', '').replace(']', ''),
                     f"{mip_runtime_list}".replace('[', '').replace(']', ''),
                     f"{coverage_list}".replace('[', '').replace(']', '')

                     ]

        all_data_list.append(data_list)

        batch_runs_solns = pd.DataFrame(all_data_list)
        batch_runs_solns.columns = ([
            "Width", "Height", "Depth", "Voxel Size", "No. Walls", "Room Seed", "Wall Edge ratio",
            "Sample Configurations", "Samples per config per axis", "Camera Seed", "Camera Budget", "Model Name",

            "Pre-processing Time", "Runtime", "Total Free Space", "LP Value", "IP Value", "Best Dual Bound",
            "Constraint Count", "Variable Count", "Nodes Traversed",

            "Super Voxel Size", "Uncovered Search Fraction", "Number of Iterations",

            "Total Coverage (%)", "Ind. Preprocessing Times", "Ind. MIP runtimes", "Ind. Coverage list"

        ])

        batch_runs_solns.round(3).to_excel(batch_run_path / f"batch_run_sols_su_24-05-30.xlsx")
        print(f"Index {index} completed")
