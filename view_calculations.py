import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from numba import jit


def calculate_camera_view(scene, camera):
    """returns set of free space points which are covered by viewing frustum"""

    view = o3d.t.geometry.RaycastingScene()
    view.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene.mesh))

    free_space_eye_dir = scene.free_space_points - camera.eye
    # free_space_eye_dir = np.array([camera.center - camera.eye, camera.center - camera.eye])

    hor_normal_unit_dir = (np.cross(camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye)
                           / np.linalg.norm(np.cross(camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye)))
    ver_normal_unit_dir = (np.cross(camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye)
                           / np.linalg.norm(np.cross(camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye)))

    # free_space_eye_unit = free_space_eye_dir / np.linalg.norm(free_space_eye_dir, axis=1)[:, np.newaxis]

    free_space_eye_hor = (free_space_eye_dir
                          - np.dot(free_space_eye_dir, hor_normal_unit_dir)[:, np.newaxis] * hor_normal_unit_dir)
    free_space_eye_ver = (free_space_eye_dir
                          - np.dot(free_space_eye_dir, ver_normal_unit_dir)[:, np.newaxis] * ver_normal_unit_dir)

    check_ver = np.array(check_points_between_acute_angles(free_space_eye_ver,
                                                           camera.ver_1_pos - camera.eye,
                                                           camera.ver_2_pos - camera.eye))

    check_hor = np.array(check_points_between_acute_angles(free_space_eye_hor,
                                                           camera.hor_1_pos - camera.eye,
                                                           camera.hor_2_pos - camera.eye))

    free_space_check = check_ver * check_hor
    free_space_frustum_indices = np.where(free_space_check == 1)[0]

    # Subset the free space points which are covered by the viewing frustum
    frustum_scene_points_eye = free_space_eye_dir[free_space_check == 1]

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
    return (frustum_scene_points_eye + camera.eye)[frustum_space_collisions_distance_vector > 1]


def find_3d_line_plane_intersection(line_point_0, line_point_1, plane_point, plane_normal, epsilon=1e-6):
    """
    Calculate the intersection point of a line and a plane in 3D space.

    Parameters:
    - line_point_0 (numpy.ndarray): A point on the line.
    - line_point_1 (numpy.ndarray): Another point on the line.
    - plane_point (numpy.ndarray): A point on the plane (plane coordinate).
    - plane_normal (numpy.ndarray): A normal vector defining the plane direction (does not need to be normalized).
    - epsilon (float, optional): Tolerance for considering parallel lines (default is 1e-6).

    Returns:
    numpy.ndarray or None: The intersection point if found; otherwise, None.
    """
    line_direction = line_point_1 - line_point_0
    dot = np.dot(plane_normal, line_direction)

    if abs(dot) > epsilon:
        w = line_point_0 - plane_point
        fac = -np.dot(plane_normal, w) / dot
        line_direction = fac * line_direction
        return line_point_0 + line_direction

    return None


@jit(nopython=True)
def check_points_between_acute_angles(array_points, vec_1_dir, vec_2_dir):
    point_between = []

    for point in array_points:
        if check_point_between_acute_angle(vec_1_dir, vec_2_dir, point):
            point_between.append(True)
        else:
            point_between.append(False)

    return point_between


@jit(nopython=True)
def check_point_between_acute_angle(vec_1_dir, vec_2_dir, between_check):
    """
    Check if a point is between the acute angle formed by the two vectors.

    Parameters:
    - vec_1_dir (numpy.ndarray): Direction of the first vector.
    - vec_2_dir (numpy.ndarray): Direction of the second vector.
    - between_check (numpy.ndarray): Direction of the point to be checked.

    Returns:
    bool: True if the point is between the vectors forming an acute angle, False otherwise.
    """
    if np.isclose(
            angle_degrees(vec_1_dir, vec_2_dir),
            angle_degrees(vec_1_dir, between_check) + angle_degrees(between_check, vec_2_dir)
                  ):
        return True

    else:
        return False


@jit(nopython=True)
def angle_degrees(vec_1_dir, vec_2_dir):
    """
    Calculate the angle in degrees between two vectors.

    Parameters:
    - vec_1_dir (numpy.ndarray): First vector.
    - vec_2_dir (numpy.ndarray): Second vector.

    Returns:
    float: The angle in degrees between the two vectors.
    """
    return np.arccos(vec_1_dir @ vec_2_dir / np.linalg.norm(vec_1_dir) / np.linalg.norm(vec_2_dir)) * 180 / np.pi


@jit(nopython=True)
def are_arrays_parallel(vec_1_dir, vec_2_dir):
    """
    Check if two arrays are parallel.

    Parameters:
    - vec_1_dir (numpy.ndarray): First array.
    - vec_2_dir (numpy.ndarray): Second array.

    Returns:
    bool: True if the arrays are parallel, False otherwise.
    """
    # Check if the arrays have the same shape
    if vec_1_dir.shape != vec_2_dir.shape:
        return False

    # Check if the cross product is zero
    cross_product = np.cross(vec_1_dir, vec_2_dir)
    return np.allclose(cross_product, np.zeros_like(cross_product))
