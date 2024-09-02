# import open3d as o3d
import numpy as np
from numba import jit
import time
from utils.array_operations import arrays_within_tolerance_complete


def segment_tensor(voxel_tensor, value):
    indices_x, indices_y, indices_z = np.where(voxel_tensor[:, :, :] == value)
    indices_set = set(zip(indices_x, indices_y, indices_z))
    segment_list = []

    while len(indices_set) != 0:

        for index in indices_set:
            break
        segment_orthogonal = flood_fill_tensor(voxel_tensor, index[0], index[1], index[2])
        segment = set(zip(*segment_orthogonal))
        indices_set = indices_set - segment
        segment_list.append(segment)

    return segment_list


@jit(nopython=True)
def flood_fill_tensor(voxel_tensor, start_x, start_y, start_z):

    if not ((0 <= start_x < voxel_tensor.shape[0])
            and (0 <= start_y < voxel_tensor.shape[1])
            and (0 <= start_z < voxel_tensor.shape[2])):

        raise ValueError("index out of bounds of tensor")

    value = voxel_tensor[start_x, start_y, start_z]
    stack = [(start_x, start_y, start_z)]
    x_list = []
    y_list = []
    z_list = []

    visited = set()
    while len(stack) != 0:
        x, y, z = stack.pop()

        if ((0 <= x < voxel_tensor.shape[0]) and (0 <= y < voxel_tensor.shape[1]) and (0 <= z < voxel_tensor.shape[2])
                and (voxel_tensor[x, y, z] == value) and ((x, y, z) not in visited)):
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)

            stack.append((x + 1, y, z))  # Right
            stack.append((x - 1, y, z))  # Left
            stack.append((x, y + 1, z))  # Down
            stack.append((x, y - 1, z))  # Up
            stack.append((x, y, z + 1))  # Forward
            stack.append((x, y, z - 1))  # Backward

        visited.add((x, y, z))

    return np.array(x_list), np.array(y_list), np.array(z_list)


def is_within_tolerance(p1, p2, tolerance):
    """ Check if p1 is within tolerance of p2 """
    return np.all(np.abs(np.array(p1) - np.array(p2)) <= tolerance)


def flood_fill(point, free_space_points, visited, tolerance):
    # Define the stack for DFS
    stack = point.reshape(1, -1)
    component = np.empty((0, 3))

    while stack.size != 0:
        current = stack[-1]
        stack = np.delete(stack, -1, axis=0)

        if arrays_within_tolerance_complete(current.reshape(1, -1), visited, 10**-4).size != 0:
            continue

        visited = np.vstack([visited, current])
        component = np.vstack([component, current.reshape(1, -1)])

        # Explore neighbors (6-connected in 3D: x±1, y±1, z±1)
        x, y, z = current
        neighbors = [
            [x + 1, y, z], [x - 1, y, z],
            [x, y + 1, z], [x, y - 1, z],
            [x, y, z + 1], [x, y, z - 1]
        ]

        for neighbor in neighbors:

            # Check if any point in points_set is within tolerance of the neighbor
            check = np.array(neighbor).reshape(1, -1)
            if ((arrays_within_tolerance_complete(free_space_points, check, tolerance).size != 0) and
                    (arrays_within_tolerance_complete(visited, check, tolerance).size == 0)):

                select = arrays_within_tolerance_complete(free_space_points, check, tolerance)
                stack = np.vstack([stack, select])

    return component, visited


def segment_points(free_space_points, tolerance=10**-4):
    visited = np.empty((0, 3))
    segments = []

    total_num_points = len(free_space_points)

    for idx in range(total_num_points):
        point = free_space_points[idx]

        if arrays_within_tolerance_complete(visited, point.reshape(1, -1), tolerance).size == 0:
            component, visited_new = flood_fill(point, free_space_points, visited, tolerance)
            visited_stack = np.vstack([visited, visited_new])
            visited = arrays_within_tolerance_complete(free_space_points, visited_stack, tolerance)
            segments.append(component)

    return segments


if __name__ == "__main__":
    # Example usage:
    # matrix = np.array([
    #     [[1, 1, 1, 1, 1],
    #      [1, 1, 0, 0, 1],
    #      [1, 0, 1, 0, 1],
    #      [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1],
    #      [1, 1, 0, 0, 1],
    #      [1, 0, 1, 0, 1],
    #      [1, 1, 1, 1, 1]],
    #     [[1, 1, 1, 1, 1],
    #      [1, 1, 0, 0, 1],
    #      [1, 0, 1, 0, 1],
    #      [1, 1, 1, 1, 1]]
    # ])
    #
    # # flood_fill(matrix, start_x, start_y, start_z, target_color, replacement_color)
    # for _ in range(10):
    #     start = time.time()
    #     y = segment_tensor(matrix, value=0)
    #     print(time.time() - start)

    free_space = np.array([
        [0.0, 0.0, 0.0], [1.000001, 0.0, 0.0], [1.000001, 1.0, 0.0],
        [2.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.000001, 1.0, 1.0], [5, 0, 1], [10, 0, 1]
    ])

    segment_list = segment_points(free_space, tolerance=1e-4)
    for i, segment in enumerate(segment_list):
        print(f"Segment {i + 1}:")
        print(np.array(segment))

    pass
