from utils.scene import Scene
import open3d as o3d
from camera.camera_class import Camera
import pathlib
import time
from numba import jit

from visualization.visualization_utils import create_voxels_subset, visualize_list
from utils.custom_object_functions import create_coordinate_axes_mesh
from utils.view_calculations import calculate_camera_view

import numpy as np


if __name__ == "__main__":
    start = time.time()

    triangle_mesh = o3d.geometry.TriangleMesh()
    np_vertices = np.array([[0, 2, -6],
                            [1, 0, -6],
                            [-1, 0, -6],
                            [0, 0, -5]
                            ])

    # np_triangles = np.array([[0, 2, 1],
    #                          [3, 0, 1],
    #                          [0, 3, 2],
    #                          [1, 2, 3]
    #                          ]).astype(np.int32)

    np_triangles = np.array([[0, 1, 2],
                             [1, 0, 3],
                             [0, 2, 3],
                             [1, 3, 2]
                             ]).astype(np.int32)

    triangle_mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
    triangle_mesh.triangles = o3d.utility.Vector3iVector(np_triangles)

    # o3d.visualization.draw_geometries([triangle_mesh])
    # data_path = pathlib.Path.cwd() / "data"

    model = Scene(filepath="None", obj_type="None")
    model.init_object = "mesh"
    # torus = o3d.geometry.TriangleMesh.create_torus().translate([0, 0, 2])
    #
    model.mesh = triangle_mesh
    model.initialize_objects(voxel_size=0.5)
    model.calculate_center()

    # visualize_selected_voxels(model, voxels=None, color="turquoise", window_title="Untitled")

    coordinate_frame = create_coordinate_axes_mesh(np.array([0, 0, 0]))
    free_space_voxels = create_voxels_subset(model.mesh, model.free_space_points, voxel_size=model.voxel_size/2,
                                             object_type="point")
    # visualize_list([model.mesh] + list(coordinate_frame) + [free_space_voxels])
    # visualize_list([model.voxel_grid] + list(coordinate_frame) + [free_space_voxels])

    cam1 = Camera()

    cam1.set_params(fov_deg=90, center=(0, 1, -1), eye=(0, 1, 0), width_px=640, height_px=480, up=(0, 1, 0))
    # cam1.set_params(fov_deg=90, center=(0, 1, 1), eye=(0, 1, -10), width_px=640, height_px=480, up=(0, 1, 0))

    cam1.calculate_bounding_frustum()

    # print(model.free_space_points.shape)
    selected_points = calculate_camera_view(model, cam1)
    viewable_points = create_voxels_subset(model.mesh, selected_points, voxel_size=3*model.voxel_size/4,
                                             object_type="point", color="deep_pink")

    visualize_list(list(coordinate_frame) + [model.mesh, free_space_voxels, viewable_points])
    pass