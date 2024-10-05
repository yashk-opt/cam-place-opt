from utils.scene import Scene
import open3d as o3d
from camera.camera_class import Camera
import pathlib
import time
from numba import jit

from visualization.visualization_utils import create_voxels_subset, visualize_list
from utils.custom_object_functions import create_coordinate_axes_mesh, create_hollow_room
from utils.view_calculations import calculate_camera_view, calculate_camera_view_v2
from utils.utils import get_project_data

import numpy as np


if __name__ == "__main__":

    start = time.time()
    # o3d.visualization.draw_geometries([triangle_mesh])
    data_path = get_project_data() / "data"

    room, line_set = create_hollow_room(width=30, height=18, depth=46, num_walls=0, seed=1,
                                        z_wall_edge_ratio=0.5)

    model = Scene(filepath="None", obj_type="None")
    model.init_object = "mesh"

    model.mesh = room
    model.initialize_objects(voxel_size=1)
    model.calculate_center()

    coordinate_frame = create_coordinate_axes_mesh(np.array([0, 0, 0]))
    free_space_voxels = create_voxels_subset(model.mesh, model.free_space_points, voxel_size=model.voxel_size / 10,
                                             object_type="point")

    # visualize_list([model.mesh] + list(coordinate_frame) + [free_space_voxels])
    # visualize_list([model.voxel_grid] + list(coordinate_frame) + [free_space_voxels])

    cam1 = Camera()

    cam1.set_params(fov_deg=90, center=(0, 0, 21), eye=tuple(map(float, (0, 0, 15))), width_px=640, height_px=480, up=(0, 1, 0))
    # cam1.set_params(fov_deg=90, center=(0, 1, 1), eye=(0, 1, -10), width_px=640, height_px=480, up=(0, 1, 0))

    cam1.calculate_bounding_frustum()

    # print(model.free_space_points.shape)
    selected_points_v1 = calculate_camera_view(model, cam1)
    selected_points_v2 = calculate_camera_view_v2(model, cam1)

    start = time.time()
    for _ in range(5):
        selected_points_v2 = calculate_camera_view_v2(model, cam1)
    print(time.time() - start)

    start = time.time()
    for _ in range(5):
        selected_points_v1 = calculate_camera_view(model, cam1)
    print(time.time() - start)

    viewable_points_v1 = create_voxels_subset(model.mesh, selected_points_v1, voxel_size=3*model.voxel_size / 4,
                                           object_type="point", color="deep_pink")
    viewable_points_v2 = create_voxels_subset(model.mesh, selected_points_v2, voxel_size=3 * model.voxel_size / 4,
                                              object_type="point", color="deep_pink")

    print(1)
    visualize_list(list(coordinate_frame) + [model.mesh, free_space_voxels, viewable_points_v1])
    pass