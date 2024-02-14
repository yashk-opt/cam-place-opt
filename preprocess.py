# import open3d as o3d
import numpy as np
from numba import jit
import time


def segment_tensor(voxel_tensor, value):
    indices_x, indices_y, indices_z = np.where(voxel_tensor[:, :, :] == value)
    indices_set = set(zip(indices_x, indices_y, indices_z))
    segment_list = []

    while len(indices_set) != 0:

        for index in indices_set:
            break
        segment_orthogonal = flood_fill(voxel_tensor, index[0], index[1], index[2])
        segment = set(zip(*segment_orthogonal))
        indices_set = indices_set - segment
        segment_list.append(segment)

    return segment_list


@jit(nopython=True)
def flood_fill(voxel_tensor, start_x, start_y, start_z):

    if not ((0 <= start_x < matrix.shape[0]) and (0 <= start_y < matrix.shape[1]) and (0 <= start_z < matrix.shape[2])):
        raise ValueError("index out of bounds of tensor")

    value = voxel_tensor[start_x, start_y, start_z]
    stack = [(start_x, start_y, start_z)]
    x_list = []
    y_list = []
    z_list = []

    visited = set()
    while len(stack) != 0:
        x, y, z = stack.pop()

        if ((0 <= x < matrix.shape[0]) and (0 <= y < matrix.shape[1]) and (0 <= z < matrix.shape[2])
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


if __name__ == "__main__":
    # Example usage:
    matrix = np.array([
        [[1, 1, 1, 1, 1],
         [1, 1, 0, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1],
         [1, 1, 0, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1],
         [1, 1, 0, 0, 1],
         [1, 0, 1, 0, 1],
         [1, 1, 1, 1, 1]]
    ])

    # flood_fill(matrix, start_x, start_y, start_z, target_color, replacement_color)
    for _ in range(10):
        start = time.time()
        y = segment_tensor(matrix, value=0)
        print(time.time() - start)

    pass
