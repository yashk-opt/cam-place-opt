import open3d as o3d
import numpy as np

from utils.custom_object_functions import create_hollow_room, get_arrow, create_coordinate_axes_mesh
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions
from utils.scene import Scene
from utils.utils import get_project_data
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view
import pickle
import pathlib
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    # data_path = pathlib.Path.cwd().parent / "data"
    data_path = get_project_data()
    data_attribute_path = data_path / "room_walls"

    scene_file_path = data_attribute_path / "room_attributes_24-06-09.xlsx"

    data_df = pd.read_excel(scene_file_path)

    for index, row in data_df.iterrows():
        width = row["width"]
        height = row["height"]
        depth = row["depth"]
        voxel_size = row["voxel size"]
        num_walls = int(row["wall"])
        seed = int(row["seed"])
        wall_edge_ratio = row["z wall edge ratio"]
        wall_orientation = row["wall orient"]

        foldername = (f"W{width}H{height}D{depth}VS{voxel_size}NW{num_walls}"
                      f"S{seed}ZR{wall_edge_ratio}WO{wall_orientation.upper()[0]}").replace(".", "-")

        # foldername = row["foldername"]

        room, line_set = create_hollow_room(width=width, height=height, depth=depth, num_walls=num_walls, seed=seed,
                                            wall_edge_ratio=wall_edge_ratio)

        model = Scene(filepath="None", obj_type="None")
        model.init_object = "mesh"

        model.mesh = room
        model.initialize_objects(voxel_size=voxel_size)
        model.calculate_center()

        scene_name = f"{foldername}".replace(".", "-")

        scene_path = data_attribute_path / scene_name
        scene_path.mkdir(parents=True, exist_ok=True)

        o3d.io.write_point_cloud(str(scene_path / f'{scene_name}_pcd.pcd'), model.pcd)
        o3d.io.write_triangle_mesh(str(scene_path / f'{scene_name}_mesh.ply'), model.mesh)
        o3d.io.write_voxel_grid(str(scene_path / f'{scene_name}_voxel_grid.ply'), model.voxel_grid)
        np.save(scene_path / f'{scene_name}_free_space.npy', model.free_space_points)
        np.save(scene_path / f'{scene_name}_voxel_tensor.npy', model.voxel_tensor)

        # free_space_voxels = create_voxels_subset(model.voxel_grid, model.free_space_points,
        #                                          voxel_size=model.voxel_size / 20, object_type="point",
        #                                          color="slate_blue")

        print(f"{index} completed")