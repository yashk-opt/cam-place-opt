import open3d as o3d
import numpy as np

from utils.custom_object_functions import create_hollow_room, get_arrow, create_coordinate_axes_mesh
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions
from utils.scene import Scene
from utils.utils import get_project_data, get_project_root
from utils.transformations import rotate_vectors
from data_processing.segmentation import segment_points
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view
import pickle
import pathlib
import pandas as pd
from tqdm import tqdm
import os
from copy import deepcopy

from utils.custom_object_functions import create_coordinate_axes_mesh


if __name__ == "__main__":

    scene_name = "apartment_0_simple"

    project_path = get_project_root()
    data_path = get_project_data()
    analytics_path = project_path / "analytics"

    scene_folder = data_path / scene_name
    scenes_list = os.listdir(scene_folder)
    algo_folder = scene_folder / "algo_sols_su"

    batch_run_path = scene_folder

    model = Scene(filepath="None", obj_type="None")
    model.init_object = "mesh"

    model.mesh = o3d.io.read_triangle_mesh(str(scene_folder / f'{scene_name}_mesh.ply'))
    model.voxel_grid = o3d.io.read_voxel_grid(str(scene_folder / f'{scene_name}_voxel_grid.ply'))
    check_free_space_points = np.load(scene_folder / f'{scene_name}_free_space.npy', allow_pickle=True)

    model.voxel_grid_dim = model.calc_voxel_grid_dim()
    model.free_space_points = model.find_voxel_centers(negative_grid=True)

    print(1)
