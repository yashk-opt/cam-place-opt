import pandas as pd
import open3d as o3d
import numpy as np

from multiprocessing import Pool, Manager
import os
from utils.utils import get_project_data, get_project_root
from data_processing.csv_reader import extract_info
from utils.scene import Scene

from models.mip_solver import MILPModel
import time
from models.sample_configurations import sample_voxel_directions, sample_explore_exploit



def run_explore_exploit(index, row):

    scene_name = row["Scene Name"]
    (width, height, depth, voxel_size,
     num_walls, room_seed, wall_edge_ratio, wall_orient) = extract_info(scene_name)

    num_voxels_per_iter = row["Sample Voxels Per Iteration"]
    num_points_axis = row["Angles per Voxel per Axis"]

    camera_seed = row["Camera Seed"]
    initial_point_seed = row["Initial Point Seed"]
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
    # num_configurations = 240
    angle_jitter_deg = row["Angle Perturbation Allowance"]
    voxel_jitter_num = row["Voxel Perturbation Allowance"]
    exploit_fraction = row["Exploit Fraction"]

    rng = np.random.default_rng(initial_point_seed)

    # Ensures if initial point and camera seed are the same, same pseudorandom generator keeps being used
    if initial_point_seed == camera_seed:
        rng2 = rng

    else:
        rng2 = np.random.default_rng(camera_seed)

    model_name = (f"{scene_name}NC{camera_budget}CS{camera_seed}NI{num_iterations}"
                  f"AJ{angle_jitter_deg}VJ{voxel_jitter_num}EX{exploit_fraction}")

    if row["Angle Selection"] == "random":
        directions_per_voxel = "random"
        direction_selection_type = "random"

    elif row["Angle Selection"] == "uniform":
        directions_per_voxel = "same"
        direction_selection_type = "uniform"

    else:
        raise ValueError(f"Incorrect Angle Selection Provided: {row['Angle Selection']}")

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
                voxel_direction_dict = sample_explore_exploit(solutions=mip.x_pa_solve,
                                                              eff_num_points=num_voxels_per_iter,
                                                              free_space_points=model.free_space_points,
                                                              num_points_axis=num_points_axis,
                                                              angle_jitter_deg=angle_jitter_deg,
                                                              voxel_jitter_num=voxel_jitter_num,
                                                              exploit_fraction=exploit_fraction,
                                                              rng=rng2, remove_vert=True,
                                                              voxel_size=model.voxel_size,
                                                              directions_per_voxel=directions_per_voxel,
                                                              direction_selection_type=direction_selection_type)

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
                mip.post_process(save_solution=True, folder_path=scene_folder / scene_name, name_suffix=f"-{iteration}")

            else:
                mip.post_process(save_solution=False)

            preprocessing_time_list.append(round(processing_time, 3))
            mip_runtime_list.append(round(mip.runtime, 3))
            coverage_list.append(round(mip.ip_value / len(model.free_space_points), 3))
            voxel_covered_list.append(mip.ip_value)
            lp_coverage_list.append(round(mip.lp_value / len(model.free_space_points), 3))

            cur_best_cam = mip.x_pa_solve

            print(f"Index {index} Iteration {iteration} completed")

    total_preprocess_time = sum(preprocessing_time_list)
    total_mip_runtime = sum(mip_runtime_list)
    total_coverage = coverage_list[-1]

    data_list = [width, height, depth, voxel_size, num_walls, room_seed, wall_edge_ratio, wall_orient,
                 num_voxels_per_iter, num_points_axis, initial_point_seed, camera_seed, camera_budget, model_name,

                 total_preprocess_time, total_mip_runtime, len(model.free_space_points), mip.lp_value, mip.ip_value,
                 mip.best_dual_bound, mip.constr_num, mip.var_num, mip.node_count,

                 exploit_fraction, voxel_jitter_num, angle_jitter_deg, direction_selection_type, num_iterations,

                 total_coverage,
                 f"{preprocessing_time_list}".replace('[', '').replace(']', ''),
                 f"{mip_runtime_list}".replace('[', '').replace(']', ''),
                 f"{coverage_list}".replace('[', '').replace(']', ''),
                 f"{voxel_covered_list}".replace('[', '').replace(']', '')

                 ]

    data_list = [str(ele) for ele in data_list]

    data_columns = [
        "Width", "Height", "Depth", "Voxel Size", "No. Walls", "Room Seed", "Z Wall Edge ratio", "Wall Orient",
        "Sample Voxels Per Iteration", "Angles per Voxel per Axis", "Initial Point Seed", "Camera Seed",
        "Camera Budget", "Model Name",

        "Pre-processing Time (Total)", "Runtime (Total)", "Total Free Space", "LP Value", "IP Value",
        "Best Dual Bound", "Constraint Count", "Variable Count", "Nodes Traversed",

        "Exploit Fraction", "Voxel Perturbation Allowance", 'Angle Perturbation Allowance', 'Angle Selection',
        "Number of Iterations",

        "Total Coverage (%)", "Ind. Preprocessing Times", "Ind. MIP runtimes", "Ind. Coverage list",
        "Num Voxels Covered List"
    ]

    return pd.Series(dict(zip(data_columns, data_list)))


def process_row(index, row, shared_dict, base_file_path, error_base_path):

    try:
        print(f"Index {index} started")

        # Run the explore-exploit strategy
        result = run_explore_exploit(index, row)
        result['idx'] = index

        # Construct file paths with the index
        file_path = str(base_file_path).replace(".csv", f"_{index}.csv")
        error_path = str(error_base_path).replace(".txt", f"_{index}.txt")

        # Save the result
        if os.path.exists(file_path):
            print("File exists, appending...")
            df = pd.read_csv(file_path)
            df = pd.concat([df, result.to_frame().T], ignore_index=True)
            df.to_csv(file_path, index=False)
        else:
            print("File does not exist, creating...")
            df = result.to_frame().T
            df.to_csv(file_path, index=False)

        shared_dict[index] = "worked"

    except Exception as e:
        # Log the error and continue
        print(f"Error processing row {index}: {e}")
        with open(error_path, 'a') as file:
            file.write(f"{index}\n")

        shared_dict[index] = "didnt"


# Using Pool for parallel execution
if __name__ == "__main__":
    data_path = get_project_data()
    scene_folder = data_path / "room_walls"
    scenes_list = os.listdir(scene_folder)

    batch_run_path = scene_folder / "qqq"

    # Base file paths for results and errors
    base_file_path = scene_folder / "qqq" / "yo.csv"
    error_base_path = scene_folder / "qqq" / "yo.txt"

    data_columns = ['Folder Name', 'Scene Name', 'Sample Voxels Per Iteration',
                    'Angles per Voxel per Axis', 'Camera Seed', 'Camera Budget',
                    'Exploit Fraction', 'Voxel Perturbation Allowance',
                    'Angle Perturbation Allowance', 'Angle Selection', 'Number of Iterations']

    # Initialize a Manager to share the result dictionary
    manager = Manager()
    shared_dict = manager.dict()

    # Load the data
    batch_run = pd.read_excel(scene_folder / "qqq" / "large_medium_room_ee_complete_240808.xlsx")
    data_formatted_columns = [ele.lower().replace(" ", "_") for ele in list(batch_run.columns)]
    batch_run.columns = data_formatted_columns
    batch_run = batch_run[batch_run.index.isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])]

    # Set up the Pool for parallel processing
    with Pool(processes=8) as pool:  # Adjust the number of processes as needed
        pool.starmap(process_row, [(idx, row, shared_dict, base_file_path, error_base_path)
                                   for idx, row in batch_run.iterrows()])

    print(1)