import numpy as np
from utils.array_operations import delete_row_if_exists
from utils.transformations import rotate_vectors
from utils.utils import decimal_to_base_3d, update_voxel_directions_dict


def sample_explore_exploit(solutions, eff_num_points, free_space_points, num_points_axis, angle_jitter_deg,
                           voxel_jitter_num, explore_fraction, rng, remove_vert, voxel_size):

    solutions = list(solutions.keys())
    num_solutions = len(solutions)
    explore_points = round(eff_num_points * explore_fraction)
    explore_voxel_directions = sample_voxel_directions(free_space_points, explore_points, num_points_axis, rng,
                                                       remove_points=solutions)

    num_configs_per_point = round(eff_num_points * num_points_axis ** 3 * (1 - explore_fraction) / num_solutions)

    sampled_angles_z = sample_spherical_cap(rng, params={
        "N": num_configs_per_point * num_solutions,
        "deg": angle_jitter_deg
    })

    sampled_angles = np.vstack(tuple([rotate_vectors(sampled_angles_z[i:i + num_configs_per_point],
                                                     from_vector=np.array([0, 0, 1]),
                                                     to_vector=np.array(solutions[i // num_configs_per_point][3:]))
                                      for i in range(0, sampled_angles_z.shape[0], num_configs_per_point)]))

    sampled_place_num = rng.integers(low=0, high=(2 * voxel_jitter_num + 1) ** 3,
                                     size=num_configs_per_point * num_solutions)

    sampled_place_rel = voxel_size * (np.array([decimal_to_base_3d(num, 2 * voxel_jitter_num + 1)
                                               for num in sampled_place_num])
                                      - np.array([[voxel_jitter_num, voxel_jitter_num, voxel_jitter_num]]))

    solution_voxels = np.repeat(np.array([solution[:3] for solution in solutions]), num_configs_per_point, axis=0)
    solution_voxels_jittered = solution_voxels + sampled_place_rel

    exploit_voxel_directions = dict()
    for i in range(len(solution_voxels_jittered)):

        if tuple(solution_voxels_jittered[i]) in exploit_voxel_directions.keys():
            exploit_voxel_directions[tuple(solution_voxels_jittered[i])] = np.vstack(
                (exploit_voxel_directions[tuple(solution_voxels_jittered[i])], sampled_angles[i])
            )

        else:
            exploit_voxel_directions[tuple(solution_voxels_jittered[i])] = np.array([sampled_angles[i]])

    return update_voxel_directions_dict(explore_voxel_directions, exploit_voxel_directions)


def sample_spherical_cap(rng, params=None):
    params = params if params is not None else {'N': 1, 'z': 0}
    rad_per_deg = np.pi / 180

    min_z = (params['z'] if 'z' in params else
             (np.cos(params['deg'] * rad_per_deg) if 'deg' in params else
              (np.cos(params['rad']) if 'rad' in params else 0)))
    N = params['N'] if 'N' in params else 1

    def generate_sample():
        z = rng.uniform(0, 1 - min_z) + min_z
        r = np.sqrt(1 - z * z)
        theta = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return [x, y, z]

    samples = [generate_sample() for _ in range(N)]
    return np.array(samples)


def sample_voxel_directions(free_space_points, num_voxels, num_points_axis, rng,
                            voxel_selection_type="random", directions_per_voxel="same",
                            direction_selection_type="uniform", remove_vertical=True, remove_points=None):

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
