import numpy as np
from utils.array_operations import delete_row_if_exists, generate_infinity_norm_arrays, arrays_within_tolerance_numba
from utils.transformations import rotate_vectors
from utils.utils import decimal_to_base_3d, update_voxel_directions_dict


def sample_uncovered_voxels(eff_num_points, num_points_axis, free_space_points,
                            large_grid_params, large_grid_data, rng,
                            uncovered_search_fraction=1, random_search_fraction=0,
                            solutions=None, uncovered_search_min_dist_cutoff="auto"):

    solution_search_fraction = 1 - uncovered_search_fraction - random_search_fraction
    min_frac = min(solution_search_fraction, uncovered_search_fraction, random_search_fraction)
    max_frac = max(solution_search_fraction, uncovered_search_fraction, random_search_fraction)

    if min_frac < 0 and max_frac > 1:
        raise ValueError(f"search fractions not adding up to 1 or are negative.")

    if random_search_fraction > 0:
        solutions = list(solutions.keys())
        num_solutions = len(solutions)
        explore_points = round(eff_num_points * random_search_fraction)
        explore_voxel_directions = sample_voxel_directions(free_space_points, explore_points, num_points_axis, rng,
                                                           remove_points=solutions)

    if uncovered_search_fraction > 0:

        block_pos_list = []
        count_list = []

        for block_pos, uncovered_count in large_grid_data.items():
            block_pos_list.append(block_pos)
            count_list.append(uncovered_count)

        num_total_exploit_configs = int(eff_num_points * uncovered_search_fraction * num_points_axis ** 3)
        prob_block_selection = np.array(count_list) / sum(count_list)

        select_block_pos = rng.choice(block_pos_list, size=num_total_exploit_configs, p=prob_block_selection)
        select_free_space = rng.choice(list(free_space_points), size=num_total_exploit_configs)

        if uncovered_search_min_dist_cutoff == "auto":
            min_cutoff = int(np.ceil(large_grid_params["block_size"] // 2))
        elif type(uncovered_search_min_dist_cutoff) is int:
            min_cutoff = uncovered_search_min_dist_cutoff
        else:
            return ValueError(f"{uncovered_search_min_dist_cutoff} is not a correct option")

        replace_elements = {}
        directions = []
        for i, voxel in enumerate(select_free_space):
            distance = np.sum(np.abs(select_free_space[i] - select_block_pos[i]) / large_grid_params["voxel_size"])
            cutoff_flag = 0
            direction_unnormal = select_block_pos[i] - select_free_space[i]
            direction = direction_unnormal / np.linalg.norm(direction_unnormal)

            while ((distance < min_cutoff)
                   or np.all(np.isclose(direction, [0, 1, 0], atol=10 ** -4))
                   or np.all(np.isclose(direction, [0, -1, 0], atol=10 ** -4))):

                cutoff_flag = 1
                free_space_point = rng.choice(list(free_space_points))
                distance = np.sum(np.abs(free_space_point - select_block_pos[i]) / large_grid_params["voxel_size"])
                direction_unnormal = select_block_pos[i] - free_space_point
                direction = direction_unnormal / np.linalg.norm(direction_unnormal)

            if cutoff_flag == 1:
                replace_elements[i] = free_space_point

            directions.append(direction)

        for i, elements in replace_elements.items():
            select_free_space[i] = elements

        ## combine voxel and directions
        uncovered_voxel_directions_dict = {}
        for i in range(len(select_free_space)):
            if tuple(select_free_space[i]) in uncovered_voxel_directions_dict.keys():
                uncovered_voxel_directions_dict[tuple(select_free_space[i])] = np.vstack(
                    (uncovered_voxel_directions_dict[tuple(select_free_space[i])], directions[i]))
            else:
                uncovered_voxel_directions_dict[tuple(select_free_space[i])] = np.array([directions[i]])

    if solution_search_fraction > 0:
        solution_search_dict = {}
        raise ValueError("Solution search fraction > 0 feature not implemented yet.")

    voxel_directions_dict = {}
    if 'uncovered_voxel_directions_dict' in locals():
        voxel_directions_dict = update_voxel_directions_dict(voxel_directions_dict, uncovered_voxel_directions_dict)

    if 'explore_voxel_directions' in locals():
        voxel_directions_dict = update_voxel_directions_dict(voxel_directions_dict, explore_voxel_directions)

    if 'solution_search_dict' in locals():
        voxel_directions_dict = update_voxel_directions_dict(voxel_directions_dict, solution_search_dict)

    return voxel_directions_dict


def sample_explore_exploit(solutions, eff_num_points, free_space_points, num_points_axis, angle_jitter_deg,
                           voxel_jitter_num, exploit_fraction, rng, remove_vert, voxel_size, **kwargs):

    directions_per_voxel = kwargs.get("directions_per_voxel", "same")
    direction_selection_type = kwargs.get("direction_selection_type", "uniform")

    explore_fraction = 1 - exploit_fraction
    solutions = list(solutions.keys())
    num_solutions = len(solutions)
    explore_points = round(eff_num_points * explore_fraction)

    if explore_fraction > 0:
        explore_voxel_directions = sample_voxel_directions(free_space_points, explore_points, num_points_axis, rng,
                                                           remove_points=solutions,
                                                           directions_per_voxel=directions_per_voxel,
                                                           direction_selection_type=direction_selection_type)

    if 1 - explore_fraction > 0:

        num_configs_per_point = round(eff_num_points * num_points_axis ** 3 * (1 - explore_fraction) / num_solutions)
        voxel_rel_block = generate_infinity_norm_arrays(voxel_jitter_num, voxel_size)
        exploit_voxel_directions = dict()
        for solution in solutions:
            # solution_freespace_dict = {solution: arrays_within_tolerance_numba(voxel_rel_block + solution,
            #                                                                    free_space_points,
            #                                                                    tolerance=voxel_size/10)}

            selectable_free_space = arrays_within_tolerance_numba(voxel_rel_block + np.array(solution[:3]),
                                                                  free_space_points,
                                                                  tolerance=voxel_size/10)

            sampled_exploit_points = selectable_free_space[
                rng.integers(len(selectable_free_space), size=num_configs_per_point)
            ]

            sampled_angles_z = sample_spherical_cap(rng, params={
                "N": num_configs_per_point,
                "deg": angle_jitter_deg
            })

            sampled_angles = rotate_vectors(sampled_angles_z,
                                            from_vector=np.array([0, 0, 1]),
                                            to_vector=np.array(solution[3:]))

            for i in range(len(sampled_exploit_points)):

                if tuple(sampled_exploit_points[i]) in exploit_voxel_directions.keys():
                    exploit_voxel_directions[tuple(sampled_exploit_points[i])] = np.vstack(
                        (exploit_voxel_directions[tuple(sampled_exploit_points[i])], sampled_angles[i])
                    )

                else:
                    exploit_voxel_directions[tuple(sampled_exploit_points[i])] = np.array([sampled_angles[i]])

    if 0 < explore_fraction < 1:
        return update_voxel_directions_dict(explore_voxel_directions, exploit_voxel_directions)

    elif explore_fraction == 0:
        return exploit_voxel_directions

    else:
        return explore_voxel_directions


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
                            direction_selection_type="uniform", remove_vertical=True, num_voxel_fraction=1.0,
                            remove_points=None):

    if voxel_selection_type == "random":
        assert num_voxels <= len(free_space_points), "Sample required should be smaller than population"
        random_indices = rng.choice(free_space_points.shape[0], size=num_voxels, replace=False)
        sampled_rows = free_space_points[random_indices, :]

    else:
        raise ValueError(f"Wrong voxel selection type. {voxel_selection_type} not implemented")

    if 0 < num_voxel_fraction <= 1.0:
        sampled_rows = sampled_rows[:round(num_voxel_fraction * num_voxels)]

    else:
        raise ValueError(f"num_voxel_fraction {num_voxel_fraction} is not allowed")

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
            directions = sample_directions(num_points_axis, sample_type="random", rng=rng)
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

        # result = np.vstack([rng.random(num_points) for _ in range(3)]).T * 2 - 1
        result = rng.multivariate_normal(np.zeros(3), np.eye(3), size=num_points)
        result /= np.linalg.norm(result, axis=1)[:, np.newaxis]

    else:
        raise ValueError(f"Not implemented sample_type {sample_type}")

    return result
