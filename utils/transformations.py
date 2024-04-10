import numpy as np


def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry


def rotate_vectors(vectors, from_vector, to_vector):
    # Normalize input vectors
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)

    # Calculate the rotation axis and angle
    axis = np.cross(from_vector, to_vector)
    angle = np.arccos(np.dot(from_vector, to_vector))

    # Construct the rotation matrix
    rotation_matrix = rotation_matrix_from_axis_angle(axis, angle)

    # Apply the rotation to each vector
    rotated_vectors = np.dot(rotation_matrix, vectors.T).T

    return rotated_vectors


def rotation_matrix_from_axis_angle(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis
    rotation_matrix = np.array([
        [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])

    return rotation_matrix
