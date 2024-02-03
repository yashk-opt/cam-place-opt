import numpy as np
from view_calculations import delete_row_if_exists


def sample_voxel_directions(free_space_points, num_voxels, num_points_axis, seed,
                            voxel_selection_type="random", directions_per_voxel="same",
                            direction_selection_type="uniform", remove_vertical=True):

    rng = np.random.default_rng(seed)
    if voxel_selection_type == "random":
        assert num_voxels <= len(free_space_points), "Sample required should be smaller than population"
        random_indices = rng.choice(free_space_points.shape[0], size=num_voxels, replace=False)
        sampled_rows = free_space_points[random_indices, :]

    else:
        raise ValueError(f"Wrong voxel selection type. {voxel_selection_type} not implemented")

    if directions_per_voxel == "same":
        if direction_selection_type == "uniform":
            directions = sample_directions(num_points_axis)

        elif direction_selection_type == "random":
            directions = sample_directions(num_points_axis, sample_type="random", rng=rng)

        else:
            raise ValueError(f"Wrong direction selection type. {direction_selection_type} not implemented")

        directions = directions[~np.any(np.isnan(directions), axis=1)]
        if remove_vertical is True:
            directions = delete_row_if_exists(directions, np.array([0, -1, 0]))
            directions = delete_row_if_exists(directions, np.array([0, 1, 0]))
        voxel_directions = {tuple(row): directions for row in sampled_rows}

    elif directions_per_voxel == "random":
        voxel_directions = dict()
        for row in sampled_rows:
            directions = sample_directions(num_points_axis, sample_type="random")
            directions = directions[~np.any(np.isnan(directions), axis=1)]
            if remove_vertical is True:
                directions = delete_row_if_exists(directions, np.array([0, -1, 0]))
                directions = delete_row_if_exists(directions, np.array([0, 1, 0]))
            voxel_directions[tuple(row)] = directions

    else:
        raise ValueError(f"Wrong direction per voxels type. {directions_per_voxel} not implemented")

    return voxel_directions


def sample_directions(num_points_axis, sample_type="uniform", angle_shift=0, angle_shift_type="constant", rng=None,
                      **kwargs):

    if sample_type == "uniform":
        if type(num_points_axis) is int:
            num_points_axis = [num_points_axis] * 3

        x_points = np.linspace(0, 1, num=num_points_axis[0]) * 2 - 1
        y_points = np.linspace(0, 1, num=num_points_axis[1]) * 2 - 1
        z_points = np.linspace(0, 1, num=num_points_axis[2]) * 2 - 1

        # Create 3D arrays from x, y, and z using meshgrid
        X, Y, Z = np.meshgrid(x_points, y_points, z_points)

        # Stack the 3D arrays into a single 2D array
        result = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        result /= np.linalg.norm(result, axis=1)[:, np.newaxis]

    elif sample_type == "random":
        if type(num_points_axis) is int:
            num_points = num_points_axis ** 3
        elif type(num_points_axis) is list:
            num_points = np.prod(num_points_axis)
        else:
            raise TypeError(f"num_points_axis cannot be {num_points_axis}")

        result = np.vstack([rng.random(num_points) for _ in range(3)]).T * 2 - 1
        result /= np.linalg.norm(result, axis=1)[:, np.newaxis]

    else:
        raise ValueError(f"Not implemented sample_type {sample_type}")

    return result
