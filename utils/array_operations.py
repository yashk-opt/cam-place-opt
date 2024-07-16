import numpy as np
from numba import jit


def delete_row_if_exists(array_2d, check_row, tolerance=10**-4):
    return array_2d[~np.all(np.isclose(array_2d, check_row, atol=tolerance), axis=1)]


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
    try:
        return np.arccos(vec_1_dir @ vec_2_dir / np.linalg.norm(vec_1_dir) / np.linalg.norm(vec_2_dir)) * 180 / np.pi
    except:
        return 0


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
def check_points_between_acute_angles(array_points, vec_1_dir, vec_2_dir):
    """
        Check if a point is between the acute angle formed by the two vectors.

        Parameters:
        - array_points (numpy.ndarray): 2-D array where each row represents a vector to be checked
        - vec_1_dir (numpy.ndarray): Direction of the first vector.
        - vec_2_dir (numpy.ndarray): Direction of the second vector.
        - between_check (numpy.ndarray): Direction of the point to be checked.

        Returns:
        list: List of True or False based on whether the corresponding point is between the vectors
        forming an acute angle.
    """
    point_between = []

    for point in array_points:
        if check_point_between_acute_angle(vec_1_dir, vec_2_dir, point):
            point_between.append(True)
        else:
            point_between.append(False)

    return point_between


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


def generate_infinity_norm_arrays(num, voxel_size, arr=None):

    if arr is None:
        arr = [0, 0, 0]

    n = len(arr)
    ranges = [range(-num, num + 1)] * n

    if arr == [0, 0, 0]:
        all_combinations = (
                np.array(np.meshgrid(*ranges)).T.reshape(-1, n) * voxel_size
        )
    else:
        arr = np.array(arr)
        n = len(arr)
        all_combinations = (
                np.array(np.meshgrid(*ranges)).T.reshape(-1, n) * voxel_size + arr
        )

    return all_combinations


@jit(nopython=True)
def arrays_within_tolerance_numba(arr1, arr2, tolerance):

    result = []
    arr_size = arr1.shape[1]
    arr1_len = arr1.shape[0]
    arr2_len = arr2.shape[0]

    for i in range(arr1_len):

        for j in range(arr2_len):
            within_tolerance = True
            for k in range(arr_size):
                if np.abs(arr1[i, k] - arr2[j, k]) > tolerance:
                    within_tolerance = False
                    break

            if within_tolerance == True:
                result.append(list(arr1[i]))
                break

    result = np.array(result)
    return result
