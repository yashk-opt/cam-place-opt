import numpy as np
from utils.array_operations import find_3d_line_plane_intersection

import open3d as o3d
import pathlib
from datetime import datetime
from collections import Counter

from utils.array_operations import are_arrays_parallel


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

        self.corner_11_dir = None
        self.corner_12_dir = None
        self.corner_21_dir = None
        self.corner_22_dir = None

        self.ver_1_dir = None
        self.ver_2_dir = None
        self.hor_1_dir = None
        self.hor_2_dir = None

        self.bounding_frustum_dict = None

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
        self.fov_ver_deg = (2 * np.arctan(np.tan(self.fov_hor_deg / 2 * np.pi / 180) * self.width_px / self.height_px)
                            * 180 / np.pi)

        ver_length = self.height_px / self.width_px * hor_length

        ver_2_center_angled_dir = -np.cross(self.center - self.eye, self.hor_1_pos - self.center)
        ver_1_center_angled_dir = -ver_2_center_angled_dir

        # center_eye_length = np.linalg.norm(self.center - self.eye)
        ver_12_center_length = center_eye_length * np.tan(self.fov_ver_deg / 2 * np.pi / 180)

        ver_1_center_angled_dir_multiplier = ver_12_center_length / np.linalg.norm(ver_1_center_angled_dir)
        ver_2_center_angled_dir_multiplier = ver_12_center_length / np.linalg.norm(ver_2_center_angled_dir)

        ver_1_angled_pos = self.center + (ver_1_center_angled_dir_multiplier * ver_1_center_angled_dir)
        ver_2_angled_pos = self.center + (ver_2_center_angled_dir_multiplier * ver_2_center_angled_dir)

        self.ver_1_pos = ver_1_angled_pos
        self.ver_2_pos = ver_2_angled_pos

    def calculate_bounding_frustum(self):
        """
        Calculates the positions of the corners of the bounding frustum based on the camera parameters.

        Parameters:
        None

        Returns:
        None
        """
        corner_11_pos = self.hor_1_pos + self.ver_1_pos - self.center
        corner_11_dir = corner_11_pos - self.eye
        self.corner_11_dir = corner_11_dir / np.linalg.norm(corner_11_dir)

        corner_12_pos = self.hor_1_pos + self.ver_2_pos - self.center
        corner_12_dir = corner_12_pos - self.eye
        self.corner_12_dir = corner_12_dir / np.linalg.norm(corner_12_dir)

        corner_21_pos = self.hor_2_pos + self.ver_1_pos - self.center
        corner_21_dir = corner_21_pos - self.eye
        self.corner_21_dir = corner_21_dir / np.linalg.norm(corner_21_dir)

        corner_22_pos = self.hor_2_pos + self.ver_2_pos - self.center
        corner_22_dir = corner_22_pos - self.eye
        self.corner_22_dir = corner_22_dir / np.linalg.norm(corner_22_dir)

        self.ver_1_dir = self.ver_1_pos - self.eye
        self.ver_1_dir = self.ver_1_dir / np.linalg.norm(self.ver_1_dir)

        self.ver_2_dir = self.ver_2_pos - self.eye
        self.ver_2_dir = self.ver_2_dir / np.linalg.norm(self.ver_2_dir)

        self.hor_1_dir = self.hor_1_pos - self.eye
        self.hor_1_dir = self.hor_1_dir / np.linalg.norm(self.hor_1_dir)

        self.hor_2_dir = self.hor_2_pos - self.eye
        self.hor_2_dir = self.hor_2_dir / np.linalg.norm(self.hor_2_dir)

        self.bounding_frustum_dict = {
            "c11_unit_dir": self.corner_11_dir,
            "c12_unit_dir": self.corner_12_dir,
            "c21_unit_dir": self.corner_21_dir,
            "c22_unit_dir": self.corner_22_dir,
            "h1_unit_dir": self.hor_1_dir,
            "h2_unit_dir": self.hor_2_dir,
            "v1_unit_dir": self.ver_1_dir,
            "v2_unit_dir": self.ver_2_dir,
        }