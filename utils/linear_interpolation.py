import numpy as np
from numba import jit
import open3d as o3d
from utils.array_operations import arrays_within_tolerance_complete


@jit(nopython=True)
def generate_line(point_start, point_end, voxel_size=1):
    """
        Generates a list of integer points representing a straight line
        between two 3D points using a digital differential analyzer (DDA) algorithm.

        Parameters:
        ----------
        point_start : np.ndarray
            A 1D numpy array of shape (3,) representing the starting point (x, y, z).
        point_end : np.ndarray
            A 1D numpy array of shape (3,) representing the ending point (x, y, z).
        voxel_size : int
            An integer value corresponding to the voxel size of the voxel grid

        Returns:
        -------
        np.ndarray
            A 2D numpy array of shape (num_steps + 1, 3), where each row represents
            a point (x, y, z) on the line from `point_start` to `point_end`.
    """
    dx = point_end[0] - point_start[0]
    dy = point_end[1] - point_start[1]
    dz = point_end[2] - point_start[2]

    remainder = np.mod(np.array([dx, dy, dz]), voxel_size)
    remainder_check = np.isclose(remainder, 0, atol=10**-4) | np.isclose(remainder, voxel_size, atol=10**-4)

    if np.sum(remainder_check) != 3:
        raise ValueError("The starting and ending point are not an exact multiple of `voxel_size` apart")

    point_start_scale = np.array([0, 0, 0])
    point_end_scale = (point_end - point_start) / voxel_size

    if np.sum(np.isclose(point_end_scale, np.round(point_end_scale), atol=10**-4)) != 3:
        raise ValueError("The starting and ending point are not an exact multiple of `voxel_size` apart")

    num_steps = round(max(abs(dx) / voxel_size, abs(dy) / voxel_size, abs(dz) / voxel_size))
    inv_num_steps = 0.0 if num_steps == 0 else 1.0 / num_steps

    x_step = dx * inv_num_steps
    y_step = dy * inv_num_steps
    z_step = dz * inv_num_steps

    x, y, z = point_start_scale[0], point_start_scale[1], point_start_scale[2]

    # Pre-allocate array to store points in 3D
    points = np.empty((num_steps + 1, 3), dtype=np.int64)

    for step in range(num_steps + 1):
        points[step, 0] = np.round(x, 0)
        points[step, 1] = np.round(y, 0)
        points[step, 2] = np.round(z, 0)
        x += x_step
        y += y_step
        z += z_step

    return points + point_start


def find_furthest_non_intersect(line_points, mesh, free_space_points):

    if len(line_points) == 1:
        return None

    view = o3d.t.geometry.RaycastingScene()
    view.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    start_point = line_points[0]
    grid_points = line_points[1:]
    grid_points = arrays_within_tolerance_complete(grid_points, free_space_points, 10**-4)

    start_points = np.tile(start_point, (len(grid_points), 1))

    start_point_dirs = start_points - grid_points

    distance_check_arr = np.hstack((grid_points, start_point_dirs))

    if hasattr(o3d, 'cuda'):
        frustum_space_collisions_distance_vector = (
            np.asarray(
                view.cast_rays(
                    o3d.core.Tensor([distance_check_arr.astype('float32', casting='same_kind')])
                )["t_hit"]).reshape(-1)
                                                 )

    else:
        frustum_space_collisions_distance_vector = (
            np.asarray(
                view.cast_rays(
                    o3d.cpu.pybind.core.Tensor([distance_check_arr.astype('float32', casting='same_kind')])
                )["t_hit"]).reshape(-1)
        )

    value = min(1, max(frustum_space_collisions_distance_vector))
    grid_points = grid_points[frustum_space_collisions_distance_vector >= value]

    if len(grid_points) == 0:
        return None
    
    return grid_points[-1]
