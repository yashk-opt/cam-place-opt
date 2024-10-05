import open3d as o3d
import numpy as np

from utils.custom_object_functions import create_hollow_room, get_arrow, create_coordinate_axes_mesh
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions
from utils.scene import Scene
from utils.utils import get_project_data
from utils.transformations import rotate_vectors
from data_processing.segmentation import segment_points
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view
import pickle
import pathlib
import pandas as pd
from tqdm import tqdm

from utils.custom_object_functions import create_coordinate_axes_mesh

from plyfile import PlyData




if __name__ == "__main__":

    simplify = True

    scene_name = "apartment_0"
    data_path = get_project_data()
    data_attribute_path = data_path / scene_name
    scene_file_path = data_attribute_path / "mesh.ply"

    foldername = "apartment_0"

    voxel_size = 1

    apartment_0_mesh = o3d.io.read_triangle_mesh(str(scene_file_path))

    axes = create_coordinate_axes_mesh([0, 0, 0])

    # Centering the vertices at the origin
    vertices = np.asarray(apartment_0_mesh.vertices)
    centroid = np.round(vertices.mean(axis=0), 2)
    updated_vertices = vertices - centroid

    # Scaling Vertices according to required cuboid volume
    min_values = updated_vertices.min(axis=0)
    max_values = updated_vertices.max(axis=0)
    ranges = max_values - min_values

    current_volume = np.prod(ranges)
    desired_volume = 20_000
    scaling_factor = np.round(np.cbrt(desired_volume / current_volume), 1)
    updated_vertices = updated_vertices * scaling_factor

    # Rotate Vertices to ensure
    # There is a function called get_minimal_oriented_bounding_box. Check it out if need be
    # figure out where y needs to align
    axis_index = np.argmin(ranges)

    min_axis = np.array([0, 0, 0])
    min_axis[axis_index] = 1

    if axis_index != 1:
        updated_vertices = rotate_vectors(updated_vertices, from_vector=min_axis, to_vector=np.array([0, 1, 0]))

    apartment_0_mesh.vertices = o3d.utility.Vector3dVector(updated_vertices)

    # Visualize the mesh
    print(1)
    # o3d.visualization.draw_geometries([apartment_0_mesh] + axes)

    if simplify is True:
        apartment_0_mesh = apartment_0_mesh.simplify_quadric_decimation(target_number_of_triangles=1_000_000)
        o3d.io.write_triangle_mesh(str(data_path / f"{scene_name}_simple" / f'{scene_name}_simple_mesh.ply'), apartment_0_mesh)

    # visualize_list([apartment_0_mesh] + axes)
    # print(1)

    model = Scene(filepath="None", obj_type="None")
    model.init_object = "mesh"
    model.mesh = apartment_0_mesh
    model.initialize_objects(voxel_size=voxel_size)

    # Find offsets and reset
    integer_part = np.floor(model.free_space_points[0])
    fractional_part = model.free_space_points[0] - integer_part

    updated_vertices = updated_vertices - fractional_part
    apartment_0_mesh.vertices = o3d.utility.Vector3dVector(updated_vertices)

    model = Scene(filepath="None", obj_type="None")
    model.init_object = "mesh"
    model.mesh = apartment_0_mesh
    model.initialize_objects(voxel_size=voxel_size)

    segments = segment_points(free_space_points=model.free_space_points)

    # voxel_set1 = create_voxels_subset(model.voxel_grid, segments[0], voxel_size=model.voxel_size / 2,
    #                                   object_type="point", color="deep_pink")

    voxel_set2 = create_voxels_subset(model.voxel_grid, segments[1], voxel_size=model.voxel_size / 2,
                                      object_type="point", color="royal_blue")

    model.free_space_points = segments[1]
    model.calculate_center()

    if simplify is True:
        scene_name = f"{scene_name}_simple"

    scene_path = data_path / scene_name

    # scene_path.mkdir(parents=True, exist_ok=True)

    # o3d.io.write_point_cloud(str(scene_path / f'{scene_name}_pcd.pcd'), model.pcd)
    o3d.io.write_triangle_mesh(str(scene_path / f'{scene_name}_mesh.ply'), model.mesh)
    o3d.io.write_voxel_grid(str(scene_path / f'{scene_name}_voxel_grid.ply'), model.voxel_grid)
    np.save(scene_path / f'{scene_name}_free_space.npy', model.free_space_points)
    np.save(scene_path / f'{scene_name}_voxel_tensor.npy', model.voxel_tensor)

    print(f"{scene_name} completed")