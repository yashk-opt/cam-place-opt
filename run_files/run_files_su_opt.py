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
from models.sample_configurations import sample_voxel_directions, sample_explore_exploit, sample_uncovered_voxels
from data_processing.voxel_calculations import voxel_block_count

from models.mip_solver import MILPModel
import time

import pickle
import pathlib
import os


if __name__ == "__main__":

    project_path = get_project_root()
    data_path = get_project_data()
    analytics_path = project_path / "analytics"

    scene_folder = data_path / "room_walls"
    scenes_list = os.listdir(scene_folder)

    batch_run_path = scene_folder

    data_columns = ["Folder Name", "Scene Name", "Sample Voxels Per Iteration", "Angles per Voxel per Axis",
                    "Initial Point Seed", "Camera Seed", "Camera Budget", "Super Voxel Size",
                    "Uncovered Search Fraction", "Angle Selection", "Number of Iterations"]

    sheet_names = ["su_fraction", "num_iter", "supervoxel_size"]

    for sheet_name in sheet_names:

        batch_run = pd.read_excel(analytics_path / "analysis_set_august_2024" / "su_analysis.xlsx",
                                  sheet_name=sheet_name)

        all_data_list = []
        for index, row in batch_run.iterrows():
            scene_name = row["Scene Name"]
            # scene_name = "W30H10D10VS1-0NW2S1111ZR0-8WOS"
            (width, height, depth, voxel_size,
             num_walls, room_seed, wall_edge_ratio, wall_orient) = extract_info(scene_name)

            num_voxels_per_iter = row["Sample Voxels Per Iteration"]
            num_points_axis = row["Angles per Voxel per Axis"]

            initial_point_seed = row["Initial Point Seed"]

            camera_seed = row["Camera Seed"] * 10
            camera_budget = row["Camera Budget"]

            model = Scene(filepath="None", obj_type="None")
            model.init_object = "mesh"

            model.pcd = o3d.io.read_point_cloud(str(data_path / "room_walls" / scene_name / f'{scene_name}_pcd.pcd'))
            model.mesh = o3d.io.read_triangle_mesh(str(data_path / "room_walls" / scene_name / f'{scene_name}_mesh.ply'))
            model.voxel_grid = o3d.io.read_voxel_grid(str(data_path / "room_walls" / scene_name / f'{scene_name}_voxel_grid.ply'))
            model.free_space_points = np.load(data_path / "room_walls" / scene_name / f'{scene_name}_free_space.npy', allow_pickle=True)
            model.voxel_tensor = np.load(data_path / "room_walls" / scene_name / f'{scene_name}_voxel_tensor.npy', allow_pickle=True)

            model.voxel_grid_dim = model.calc_voxel_grid_dim()
            model.voxel_size = model.voxel_grid.voxel_size

            mip = MILPModel(scene=model, num_cameras=camera_budget)

            num_iterations = row["Number of Iterations"]
            block_size = row["Super Voxel Size"]
            uncovered_search_fraction = row["Uncovered Search Fraction"]
            explore_fraction = 1 - uncovered_search_fraction

            model_name = f"{scene_name}NC{camera_budget}CS{camera_seed}NI{num_iterations}SV{block_size}US{uncovered_search_fraction}"

            if row["Angle Selection"] == "random":
                directions_per_voxel = "random"
                direction_selection_type = "random"

            elif row["Angle Selection"] == "uniform":
                directions_per_voxel = "same"
                direction_selection_type = "uniform"

            else:
                raise ValueError(f"Incorrect Angle Selection Provided: {row['Angle Selection']}")

            rng = np.random.default_rng(initial_point_seed)

            # Ensures if initial point and camera seed are the same, same pseudorandom generator keeps being used
            if initial_point_seed == camera_seed:
                rng2 = rng

            else:
                rng2 = np.random.default_rng(camera_seed)


            preprocessing_time_list = []
            mip_runtime_list = []
            coverage_list = []
            lp_coverage_list = []
            voxel_covered_list = []

            start_overall = time.time()
            for iteration in range(1, num_iterations + 1):

                if time.time() - start_overall < 3600:

                    if iteration == 1:
                        voxel_direction_dict = sample_voxel_directions(free_space_points=model.free_space_points,
                                                                       num_voxels=num_voxels_per_iter,
                                                                       num_points_axis=num_points_axis,
                                                                       directions_per_voxel=directions_per_voxel,
                                                                       direction_selection_type=direction_selection_type,
                                                                       rng=rng)

                    else:
                        voxel_info = mip.extract_voxel_info(type_info="uncovered")
                        voxel_collection_count, large_grid_info = voxel_block_count(model, voxel_info, block_size)

                        voxel_direction_dict = sample_uncovered_voxels(eff_num_points=num_voxels_per_iter,
                                                                       num_points_axis=num_points_axis,
                                                                       free_space_points=model.free_space_points,
                                                                       large_grid_data=voxel_collection_count,
                                                                       large_grid_params=large_grid_info, rng=rng2,
                                                                       uncovered_search_fraction=uncovered_search_fraction,
                                                                       random_search_fraction=explore_fraction,
                                                                       solutions=mip.x_pa_solve,
                                                                       mesh=model.mesh, voxel_grid_size=model.voxel_size)

                    start = time.time()
                    if iteration == 1:
                        mip.pre_process(voxel_directions_dict=voxel_direction_dict)
                    else:
                        mip.pre_process(voxel_directions_dict=voxel_direction_dict, update_existing=True)

                    processing_time = time.time() - start

                    if iteration == 1:
                        mip.create_model(model_name=model_name)

                    else:
                        mip.create_model(model_name=model_name, load_sol=cur_best_cam)

                    mip.optimize(max_run_time=3600, verbose=True)

                    if iteration % 100 == 0:
                        mip.post_process(save_solution=False,
                                         folder_path=scene_folder,
                                         name_suffix=f"-{iteration}")

                        # with open(scene_folder / f'algo_sols_solve_{iteration}_v2.pkl', 'wb') as file:
                        #     pickle.dump(set(mip.x_pa_solve.keys()), file)
                        # with open(scene_folder / f'algo_sols_all_{iteration}_v2.pkl', 'wb') as file:
                        #     pickle.dump(set(mip.x_pa.keys()), file)

                    else:
                        mip.post_process(save_solution=False)

                    preprocessing_time_list.append(round(processing_time, 3))
                    mip_runtime_list.append(round(mip.runtime, 3))
                    coverage_list.append(round(mip.ip_value / len(model.free_space_points), 2))
                    voxel_covered_list.append(mip.ip_value)
                    lp_coverage_list.append(round(mip.lp_value / len(model.free_space_points), 2))

                    cur_best_cam = mip.x_pa_solve
                    print(f"Index {index} Iteration {iteration} completed")

            total_preprocess_time = sum(preprocessing_time_list)
            total_mip_runtime = sum(mip_runtime_list)
            total_coverage = coverage_list[-1]

            # mip.post_process(save_solution=True, folder_path=scene_folder / scene_name, name_suffix=f"-final")

            data_list = [width, height, depth, voxel_size, num_walls, room_seed, wall_edge_ratio, wall_orient,
                         num_voxels_per_iter, num_points_axis, initial_point_seed, camera_seed, camera_budget, model_name,

                         total_preprocess_time, total_mip_runtime, len(model.free_space_points), mip.lp_value, mip.ip_value,
                         mip.best_dual_bound, mip.constr_num, mip.var_num, mip.node_count,

                         block_size, uncovered_search_fraction, num_iterations,

                         total_coverage,
                         f"{preprocessing_time_list}".replace('[', '').replace(']', ''),
                         f"{mip_runtime_list}".replace('[', '').replace(']', ''),
                         f"{coverage_list}".replace('[', '').replace(']', ''),
                         f"{voxel_covered_list}".replace('[', '').replace(']', '')

                         ]

            all_data_list.append(data_list)

            batch_runs_solns = pd.DataFrame(all_data_list)
            batch_runs_solns.columns = ([
                "Width", "Height", "Depth", "Voxel Size", "No. Walls", "Room Seed", "Z Wall Edge ratio", "Wall Orient",
                "Sample Voxels Per Iteration", "Angles per Voxel per Axis", "Initial Point Seed", "Camera Seed",
                "Camera Budget", "Model Name",

                "Pre-processing Time (Total)", "Runtime (Total)", "Total Free Space", "LP Value", "IP Value",
                "Best Dual Bound", "Constraint Count", "Variable Count", "Nodes Traversed",

                "Super Voxel Size", "Uncovered Search Fraction", "Number of Iterations",

                "Total Coverage (%)", "Ind. Preprocessing Times", "Ind. MIP runtimes", "Ind. Coverage list",
                "Num Voxels Covered List"

            ])

            batch_runs_solns.round(3).to_excel(analytics_path / "analysis_set_august_2024" / f"sols_su_{sheet_name}.xlsx")
            print(f"{sheet_name} Index {index} completed")
