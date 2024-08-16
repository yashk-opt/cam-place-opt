import open3d as o3d
import numpy as np

from utils.custom_object_functions import create_hollow_room, get_arrow, create_coordinate_axes_mesh
from visualization.visualization_utils import visualize_list, create_voxels_subset
from models.sample_configurations import sample_directions
from utils.scene import Scene
from camera.camera_class import Camera
from utils.view_calculations import calculate_camera_view
import pickle
import pathlib
from models.sample_configurations import sample_voxel_directions
from models.mip_solver import MILPModel


if __name__ == "__main__":

    data_path = pathlib.Path.cwd().parent / "data"
    scene_name = f"empty_room_0-5"
    scene_path = data_path / scene_name
    seed_value = 42
    num_voxels = 100
    num_directions_axis = 3

    model = Scene(filepath="None", obj_type="None")
    model.init_object = "mesh"

    model.pcd = o3d.io.read_point_cloud(str(scene_path / f'{scene_name}_pcd.pcd'))
    model.mesh = o3d.io.read_triangle_mesh(str(scene_path / f'{scene_name}_mesh.ply'))
    model.voxel_grid = o3d.io.read_voxel_grid(str(scene_path / f'{scene_name}_voxel_grid.ply'))
    model.free_space_points = np.load(scene_path / f'{scene_name}_free_space.npy', allow_pickle=True)
    model.voxel_tensor = np.load(scene_path / f'{scene_name}_voxel_tensor.npy', allow_pickle=True)

    model.voxel_grid_dim = model.calc_voxel_grid_dim()
    model.voxel_size = model.voxel_grid.voxel_size

    free_space_voxels = create_voxels_subset(model.voxel_grid, model.free_space_points,
                                             voxel_size=model.voxel_size / 20, object_type="point", color="slate_blue")

    voxel_direction_dict = sample_voxel_directions(free_space_points=model.free_space_points, num_voxels=num_voxels,
                                                   num_points_axis=num_directions_axis, seed=seed_value)

    # with open(scene_path / f"V{num_voxels}D{num_directions_axis ** 3}S{seed_value}.pkl", 'wb') as file:
    #     pickle.dump(voxel_direction_dict, file)

    with open(scene_path / f"V{num_voxels}D{num_directions_axis ** 3}S{seed_value}.pkl", 'rb') as file:
        voxel_direction_dict = pickle.load(file)

    mip = MILPModel(scene=model, voxel_directions=voxel_direction_dict, num_cameras=3)
    # mip.pre_process()
    # mip.create_model(model_name="check")
    # mip.optimize(max_run_time=3600, verbose=True)
    # mip.post_process()
    # mip.visualize()



    mip.x_pa_solve = {(0,
      0,
      0,
      -0.5773502691896258,
      -0.5773502691896258,
      0.5773502691896258): 1.0,
     }

    viewable_points_pa = dict()
    bounding_frustum_pa_c = dict()
    for s in mip.x_pa_solve.keys():
        cam = Camera()
        cam_center = tuple(np.array(s[:3]) + np.array(s[-3:]))
        cam.set_params(fov_deg=90, center=cam_center, eye=s[:3], width_px=640, height_px=480, up=(0, 1, 0))
        cam.calculate_bounding_frustum()

        viewable_points_pa[s] = calculate_camera_view(mip.scene_data, cam)
        bounding_frustum_pa_c[s] = {
            "c11_unit_dir": cam.corner_11_dir,
            "c12_unit_dir": cam.corner_12_dir,
            "c21_unit_dir": cam.corner_21_dir,
            "c22_unit_dir": cam.corner_22_dir,
            "h1_unit_dir": cam.hor_1_dir,
            "h2_unit_dir": cam.hor_2_dir,
            "v1_unit_dir": cam.ver_1_dir,
            "v2_unit_dir": cam.ver_2_dir,
        }

    mip.bounding_frustum_pa_c = bounding_frustum_pa_c
    mip.viewable_points_pa = viewable_points_pa

    mip.visualize()
