from pathlib import Path
from numba import jit
import numpy as np


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_project_data() -> Path:
    return Path(__file__).parent.parent / "data"


@jit(nopython=True)
def decimal_to_base_3d(n, base):

    result = [0, 0, 0]
    if n == 0:
        return result

    counter = 0
    while n > 0:

        result[counter] = n % base
        n //= base
        counter += 1

    return result


def update_voxel_directions_dict(dict1, dict2):

    result_dict = dict()

    # Combine keys from both dictionaries
    all_keys = set(dict1.keys()).union(dict2.keys())

    for key in all_keys:
        if key in dict1 and key in dict2:
            # Stack 2D arrays if both dictionaries have the same key
            stacked_array = np.vstack((dict1[key], dict2[key]))
            result_dict[key] = stacked_array
        elif key in dict1:
            result_dict[key] = dict1[key]
        elif key in dict2:
            result_dict[key] = dict2[key]

    return result_dict
