import numpy as np
from view_calculations import find_3d_line_plane_intersection

import open3d as o3d
import pathlib
from datetime import datetime
from collections import Counter

from view_calculations import are_arrays_parallel


class Camera:
    """
    Class representing a virtual camera with parameters for field of view (fov), position, and orientation.

    Attributes:
    - fov_hor_deg (float): Horizontal field of view in degrees.
    - fov_ver_deg (float): Vertical field of view in degrees.
    - center (numpy.ndarray): Camera center coordinates.
    - eye (numpy.ndarray): Camera eye coordinates.
    - up (numpy.ndarray): Up vector of the camera.
    - width_px (int): Width of the image in pixels.
    - height_px (int): Height of the image in pixels.
    - hor_1_pos (numpy.ndarray): Position of the first horizontal crosshair.
    - hor_2_pos (numpy.ndarray): Position of the second horizontal crosshair.
    - ver_1_pos (numpy.ndarray): Position of the first vertical crosshair.
    - ver_2_pos (numpy.ndarray): Position of the second vertical crosshair.
    - corner_11_pos (numpy.ndarray): Position of the first corner of the bounding frustum.
    - corner_12_pos (numpy.ndarray): Position of the second corner of the bounding frustum.
    - corner_21_pos (numpy.ndarray): Position of the third corner of the bounding frustum.
    - corner_22_pos (numpy.ndarray): Position of the fourth corner of the bounding frustum.
    """
    def __init__(self):
        """
        Initializes a Camera object with default attribute values.
        """
        self.fov_hor_deg = None
        self.fov_ver_deg = None
        self.center = None
        self.eye = None
        self.up = None
        self.width_px = None
        self.height_px = None

        self.hor_1_pos = None
        self.hor_2_pos = None
        self.ver_1_pos = None
        self.ver_2_pos = None

        self.corner_11_pos = None
        self.corner_12_pos = None
        self.corner_21_pos = None
        self.corner_22_pos = None

    def set_params(self, fov_deg=90, center=(0, 0, 0), eye=(1, 1, 1), width_px=640, height_px=480, up=(0, 1, 0)):
        """
        Sets the parameters for the Camera object.

        Parameters:
        - fov_deg (float, optional): Horizontal field of view in degrees (default is 90).
        - center (tuple or list, optional): Camera center coordinates (default is (0, 0, 0)).
        - eye (tuple or list, optional): Camera eye coordinates (default is (1, 1, 1)).
        - width_px (int, optional): Width of the image in pixels (default is 640).
        - height_px (int, optional): Height of the image in pixels (default is 480).
        - up (tuple or list, optional): Up vector of the camera (default is (0, 1, 0)).

        Returns:
        None
        """
        self.fov_hor_deg = fov_deg
        self.center = np.array(center)
        self.eye = np.array(eye)
        self.width_px = width_px
        self.height_px = height_px
        self.up = np.array(up)

        assert are_arrays_parallel(self.up, self.center - self.eye) is False, ("up vector parallel to "
                                                                               "eye-center direction")

        # Calculate horizontal cross-hairs
        hor_2_center_dir = -np.cross(self.center - self.eye, self.up)
        hor_1_center_dir = -hor_2_center_dir

        center_eye_length = np.linalg.norm(self.center - self.eye)
        hor_12_center_length = center_eye_length * np.tan(self.fov_hor_deg / 2 * np.pi / 180)

        hor_12_center_dir_multiplier = hor_12_center_length / np.linalg.norm(hor_1_center_dir)

        self.hor_1_pos = self.center + (hor_12_center_dir_multiplier * hor_1_center_dir)
        self.hor_2_pos = self.center + (hor_12_center_dir_multiplier * hor_2_center_dir)

        hor_length = np.linalg.norm(self.hor_1_pos - self.hor_2_pos)

        # Calculate vertical cross-hairs
        ver_length = self.height_px / self.width_px * hor_length

        self.fov_ver_deg = (
                (180 / np.pi)
                * (2 * np.arctan(ver_length / hor_length
                                 * np.tan(self.fov_hor_deg / 2 * np.pi / 180)
                                 )
                   )
        )

        ver_2_center_angled_dir = -np.cross(self.center - self.eye, self.hor_1_pos - self.center)
        ver_1_center_angled_dir = -ver_2_center_angled_dir

        # center_eye_length = np.linalg.norm(self.center - self.eye)
        ver_12_center_length = center_eye_length * np.tan(self.fov_ver_deg / 2 * np.pi / 180)

        ver_1_center_angled_dir_multiplier = ver_12_center_length / np.linalg.norm(ver_1_center_angled_dir)
        ver_2_center_angled_dir_multiplier = ver_12_center_length / np.linalg.norm(ver_2_center_angled_dir)

        ver_1_angled_pos = self.center + (ver_1_center_angled_dir_multiplier * ver_1_center_angled_dir)
        ver_2_angled_pos = self.center + (ver_2_center_angled_dir_multiplier * ver_2_center_angled_dir)

        self.ver_1_pos = find_3d_line_plane_intersection(self.eye,
                                                         ver_1_angled_pos,
                                                         self.center,
                                                         np.cross(self.hor_1_pos - self.center, self.up))

        self.ver_2_pos = find_3d_line_plane_intersection(self.eye,
                                                         ver_2_angled_pos,
                                                         self.center,
                                                         np.cross(self.hor_2_pos - self.center, self.up))

    def calculate_bounding_frustum(self):
        """
        Calculates the positions of the corners of the bounding frustum based on the camera parameters.

        Parameters:
        None

        Returns:
        None
        """
        self.corner_11_pos = self.hor_1_pos + self.ver_1_pos - self.center
        self.corner_12_pos = self.hor_1_pos + self.ver_2_pos - self.center
        self.corner_21_pos = self.hor_2_pos + self.ver_1_pos - self.center
        self.corner_22_pos = self.hor_2_pos + self.ver_2_pos - self.center

