import numpy as np
import open3d as o3d
from utils.array_operations import (check_points_between_acute_angles, check_point_between_acute_angle,
                                    arrays_within_tolerance_indices_numba)
from collections import deque
from numba import jit


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
    if hasattr(o3d, 'cuda'):
        frustum_space_collisions_distance_vector = (
            np.asarray(
                view.cast_rays(
                    o3d.core.Tensor([frustum_space_rays.astype('float32', casting='same_kind')])
                )["t_hit"]).reshape(-1)
                                                 )

    else:
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


def return_view_frustum_points(scene, camera):

    hor_normal_unit_dir = (np.cross(camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye)
                           / np.linalg.norm(np.cross(camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye)))
    ver_normal_unit_dir = (np.cross(camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye)
                           / np.linalg.norm(np.cross(camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye)))

    delta = scene.voxel_size

    free_space_set = set(tuple(sublist) for sublist in scene.free_space_points)
    cam_eye = tuple(camera.eye)
    view_set = set()
    visited = {cam_eye}

    new_pos = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]) * delta
    stack = deque(new_pos + np.array(cam_eye))

    while stack:
        ele = stack.popleft()
        tup_ele = tuple(ele)
        if tup_ele not in visited:
            if tup_ele in free_space_set:
                visited.add(tup_ele)
                if check_viewing_frustum_point(camera.eye, camera.ver_1_pos, camera.ver_2_pos,
                                               camera.hor_1_pos, camera.hor_2_pos, np.array([ele]),
                                               hor_normal_unit_dir, ver_normal_unit_dir) is True:
                    view_set.add(tup_ele)
                    stack.extend(new_pos + ele)

    view_set = np.array(list(view_set))
    free_space_frustum_indices = arrays_within_tolerance_indices_numba(scene.free_space_points,
                                                                       view_set, tolerance=10**-4)
    return free_space_frustum_indices


@jit(nopython=True)
def check_viewing_frustum_point(eye, ver_1_pos, ver_2_pos, hor_1_pos, hor_2_pos, ele, hor_normal_unit_dir, ver_normal_unit_dir):

    free_space_eye_dir = ele - eye

    # Calculate component of free space vector on the horizontal and vertical planes
    free_space_eye_hor = (free_space_eye_dir
                          - np.dot(free_space_eye_dir, hor_normal_unit_dir)[:, np.newaxis] * hor_normal_unit_dir)
    free_space_eye_ver = (free_space_eye_dir
                          - np.dot(free_space_eye_dir, ver_normal_unit_dir)[:, np.newaxis] * ver_normal_unit_dir)

    # Check if component of free space vector on the horizontal and vertical planes is in the viewing frustum
    check_ver = check_point_between_acute_angle(ver_1_pos - eye,
                                                ver_2_pos - eye, free_space_eye_ver[0])

    check_hor = check_point_between_acute_angle(hor_1_pos - eye,
                                                hor_2_pos - eye, free_space_eye_hor[0])

    return bool(check_ver * check_hor)


def calculate_camera_view_v2(scene, camera):

    view = o3d.t.geometry.RaycastingScene()
    view.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene.mesh))
    free_space_frustum_indices = np.array(return_view_frustum_points(scene, camera))

    free_space_eye_dir = scene.free_space_points - camera.eye
    # Subset the free space points which are covered by the viewing frustum
    frustum_scene_points_eye = free_space_eye_dir[free_space_frustum_indices]

    # Use raycasting on the subset of free space points to figure out how far these points are before hitting the scene
    camera_eyes = np.tile(camera.eye, (len(frustum_scene_points_eye), 1))
    frustum_space_rays = np.hstack((camera_eyes, frustum_scene_points_eye))
    if hasattr(o3d, 'cuda'):
        frustum_space_collisions_distance_vector = (
            np.asarray(
                view.cast_rays(
                    o3d.core.Tensor([frustum_space_rays.astype('float32', casting='same_kind')])
                )["t_hit"]).reshape(-1)
        )

    else:
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