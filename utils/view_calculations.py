import numpy as np
import open3d as o3d
from utils.array_operations import check_points_between_acute_angles


def calculate_camera_view(scene, camera):
    """
        Returns free space points which are covered by the viewing frustum of the camera and unobstructed by the scene

        Parameters:
        - scene (Scene.scene): This is a scene object. It should have the free space points calculated and must have a
        triangle mesh to work with
        - camera (Camera.camera): This is a camera object, it has a field of view and infinite depth.

        Returns:
        numpy.ndarray: The 2D array of free space points which are viewable by the camera, where each row is a 3D
        vector.
    """
    view = o3d.t.geometry.RaycastingScene()
    view.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene.mesh))

    # Extract direction vectors from eye to each free_space_points
    free_space_eye_dir = scene.free_space_points - camera.eye

    # Calculate normal vectors to the horizontal and vertical planes
    hor_normal_unit_dir = (np.cross(camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye)
                           / np.linalg.norm(np.cross(camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye)))
    ver_normal_unit_dir = (np.cross(camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye)
                           / np.linalg.norm(np.cross(camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye)))

    # free_space_eye_unit = free_space_eye_dir / np.linalg.norm(free_space_eye_dir, axis=1)[:, np.newaxis]

    # Calculate component of free space vector on the horizontal and vertical planes
    free_space_eye_hor = (free_space_eye_dir
                          - np.dot(free_space_eye_dir, hor_normal_unit_dir)[:, np.newaxis] * hor_normal_unit_dir)
    free_space_eye_ver = (free_space_eye_dir
                          - np.dot(free_space_eye_dir, ver_normal_unit_dir)[:, np.newaxis] * ver_normal_unit_dir)

    # Check if component of free space vector on the horizontal and vertical planes is in the viewing frustum
    check_ver = np.array(check_points_between_acute_angles(free_space_eye_ver,
                                                           camera.ver_1_pos - camera.eye,
                                                           camera.ver_2_pos - camera.eye))

    check_hor = np.array(check_points_between_acute_angles(free_space_eye_hor,
                                                           camera.hor_1_pos - camera.eye,
                                                           camera.hor_2_pos - camera.eye))

    free_space_check = check_ver * check_hor
    free_space_frustum_indices = np.where(free_space_check == 1)[0]

    # Subset the free space points which are covered by the viewing frustum
    frustum_scene_points_eye = free_space_eye_dir[free_space_frustum_indices]

    # Use raycasting on the subset of free space points to figure out how far these points are before hitting the scene
    camera_eyes = np.tile(camera.eye, (len(frustum_scene_points_eye), 1))
    frustum_space_rays = np.hstack((camera_eyes, frustum_scene_points_eye))
    frustum_space_collisions_distance_vector = (
        np.asarray(
            view.cast_rays(
                o3d.cpu.pybind.core.Tensor([frustum_space_rays.astype('float32', casting='same_kind')])
            )["t_hit"]).reshape(-1)
                                             )

    # If the length of the free space points from the eye is less than the length of the hit points from the eye,
    # than those free points are visible
    # return (frustum_scene_points_eye + camera.eye)[frustum_space_collisions_distance_vector > 1]
    return scene.free_space_points[free_space_frustum_indices[frustum_space_collisions_distance_vector > 1]]
