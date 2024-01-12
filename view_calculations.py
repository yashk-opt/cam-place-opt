import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from numba import jit


def calculate_camera_view(scene, camera):
    """returns set of free space centers which are covered by viewing frustum"""

    view = o3d.t.geometry.RaycastingScene()
    view.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(scene.mesh))

    free_space_eye_dir = scene.free_space_points - camera.eye
    # free_space_eye_dir = np.array([camera.center - camera.eye, camera.center - camera.eye])

    hor_normal_unit_dir = (np.cross(camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye)
                           / np.linalg.norm(np.cross(camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye)))
    ver_normal_unit_dir = (np.cross(camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye)
                           / np.linalg.norm(np.cross(camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye)))

    # free_space_eye_unit = free_space_eye_dir / np.linalg.norm(free_space_eye_dir, axis=1)[:, np.newaxis]

    free_space_eye_hor = free_space_eye_dir - np.dot(free_space_eye_dir, hor_normal_unit_dir)[:, np.newaxis] * hor_normal_unit_dir
    free_space_eye_ver = free_space_eye_dir - np.dot(free_space_eye_dir, ver_normal_unit_dir)[:, np.newaxis] * ver_normal_unit_dir

    check_ver = np.array(points_between_acute_angles(free_space_eye_ver, camera.ver_1_pos - camera.eye, camera.ver_2_pos - camera.eye))
    check_hor = np.array(points_between_acute_angles(free_space_eye_hor, camera.hor_1_pos - camera.eye, camera.hor_2_pos - camera.eye))
    free_space_check = check_ver * check_hor

    # Subset the free space points which are covered by the viewing frustum
    frustum_scene_points_eye = free_space_eye_dir[free_space_check == 1]

    # Use raycasting on the subset of free space points to figure out how far these points are before hitting the scene
    camera_eyes = np.tile(camera.eye, (len(frustum_scene_points_eye), 1))
    free_space_rays = np.hstack((camera_eyes, frustum_scene_points_eye))
    free_space_collisions_distance_vector = (
        np.asarray(
            view.cast_rays(
                o3d.cpu.pybind.core.Tensor([free_space_rays.astype('float32',casting='same_kind')])
            )["t_hit"]).reshape(-1)
                                             )

    # If the length of the free space points from the eye is less than the length of the hit points from the eye,
    # than those free points are visible
    return scene.free_space_points[free_space_collisions_distance_vector
                                    > np.linalg.norm(frustum_scene_points_eye, axis=1)]

def intersect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = p1 - p0
    dot = np.dot(p_no, u)

    if abs(dot) > epsilon:
        w = p0 - p_co
        fac = -np.dot(p_no, w) / dot
        u = fac * u
        return p0 + u

    return None



@jit(nopython=True)
def points_between_acute_angles(array_points, vec_1, vec_2):
    point_between = []

    for point in array_points:
        if check_between_acute_angle(vec_1, vec_2, point):
            point_between.append(True)
        else:
            point_between.append(False)

    return point_between


@jit(nopython=True)
def check_between_acute_angle(vec_1, vec_2, between_check):
    if np.isclose(angle(vec_1, vec_2), angle(vec_1, between_check) + angle(between_check, vec_2)):
        return True

    else:
        return False


@jit(nopython=True)
def angle(vec_1, vec_2):
    return np.arccos(vec_1 @ vec_2 / np.linalg.norm(vec_1) / np.linalg.norm(vec_2)) * 180 / np.pi

    # rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    #     fov_deg=camera.fov_horiz_deg,
    #     center=camera.center,
    #     eye=camera.eye,
    #     up=camera.up,
    #     width_px=camera.width_px,
    #     height_px=camera.height_px,
    # )
    #
    # ans = view.cast_rays(rays)
    # scene_view_triangle_ids = set(np.array(ans["primitive_ids"]).flatten())


    # frustum_corner_11 = hor_1_eye + vert_1_eye - camera.center
    # frustum_corner_12 = hor_1_eye + vert_2_eye - camera.center
    # frustum_corner_21 = hor_2_eye + vert_1_eye - camera.center
    # frustum_corner_22 = hor_2_eye + vert_2_eye - camera.center


def are_arrays_parallel(array1, array2):
    # Check if the arrays have the same shape
    if array1.shape != array2.shape:
        return False

    # Check if the cross product is zero
    cross_product = np.cross(array1, array2)
    return np.allclose(cross_product, np.zeros_like(cross_product))

    # # Find bounding frustum vectors of camera
    # hor_1_center_dir = np.cross(camera.center - camera.eye, camera.up)
    # hor_2_center_dir = -np.cross(camera.center - camera.eye, camera.up)
    #
    # center_eye_length = np.linalg.norm(camera.center - camera.eye)
    # hor_12_center_length = center_eye_length * np.tan(camera.fov_hor_deg / 2 * np.pi / 180)
    #
    # hor_12_center_dir_multiplier = hor_12_center_length / np.linalg.norm(hor_1_center_dir)
    #
    # camera.hor_1_pos = camera.center + (hor_12_center_dir_multiplier * hor_1_center_dir)
    # camera.hor_2_pos = camera.center + (hor_12_center_dir_multiplier * hor_2_center_dir)
    #
    # camera.ver_1_pos = (camera.center
    #               + (camera.height_px / camera.width_px * np.linalg.norm(camera.hor_1_pos - camera.center))
    #               * camera.up / np.linalg.norm(camera.up))
    #
    # camera.ver_2_pos = (camera.center
    #               - (camera.height_px / camera.width_px * np.linalg.norm(camera.hor_2_pos - camera.center))
    #               * camera.up / np.linalg.norm(camera.up))

    # Find free space points in camera frustum


