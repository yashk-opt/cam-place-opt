import numpy as np
import ast
from collections import defaultdict
import open3d as o3d


def voxel_block_count(scene, voxel_info, block_size, voxel_info_type="uncovered"):

    if block_size % 2 != 1:
        block_size += 1
        print(f"block_size to be odd. changed to {block_size}")

    if voxel_info_type == "uncovered":
        coverage_data = voxel_info["uncovered"]
        selected_voxels = {tuple(ast.literal_eval(num)) for num, value in coverage_data.items() if value == 0}

    large_grid_shape = np.array([int(np.ceil(dim / block_size)) for dim in scene.voxel_tensor.shape])
    small_grid_origin_voxel_pos = scene.voxel_grid.origin + scene.voxel_size / 2

    large_small_origin_index_diff = (large_grid_shape * block_size - np.array(scene.voxel_tensor.shape)) // 2
    large_grid_origin_voxel_pos = small_grid_origin_voxel_pos - large_small_origin_index_diff * scene.voxel_size

    large_grid_block_pos_dict = defaultdict(lambda: 0)
    for voxel in selected_voxels:
        voxel_small_grid_index = ((np.array(voxel) - small_grid_origin_voxel_pos) / scene.voxel_size).astype(int)
        voxel_large_grid_index = voxel_small_grid_index + large_small_origin_index_diff
        voxel_block_index = voxel_large_grid_index // block_size
        voxel_block_pos = (large_grid_origin_voxel_pos
                           + scene.voxel_size * (block_size // 2)
                           + voxel_block_index * block_size * scene.voxel_size)

        large_grid_block_pos_dict[tuple(voxel_block_pos)] += 1

    large_grid_info = {
        "block_size": block_size,
        "large_grid_origin_voxel_pos": large_grid_origin_voxel_pos,
        "large_grid_shape": large_grid_shape,
        "large_small_origin_index_diff": large_small_origin_index_diff,
        "voxel_size": scene.voxel_size
    }

    return dict(large_grid_block_pos_dict), large_grid_info


def bresenham_3d(x1, y1, z1, x2, y2, z2, voxel_size):

    line_points = [(x1, y1, z1)]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    if x2 > x1:
        xs = voxel_size
    else:
        xs = -voxel_size
    if y2 > y1:
        ys = voxel_size
    else:
        ys = -voxel_size
    if z2 > z1:
        zs = voxel_size
    else:
        zs = -voxel_size

    # Driving axis is X-axis
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            line_points.append((x1, y1, z1))

    # Driving axis is Y-axis
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            line_points.append((x1, y1, z1))

    # Driving axis is Z-axis
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            line_points.append((x1, y1, z1))
    return line_points
