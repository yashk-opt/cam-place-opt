import numpy as np
from view_calculations import intersect_line_plane_v3

import open3d as o3d
import pathlib
from datetime import datetime
from collections import Counter

from view_calculations import are_arrays_parallel

class Camera:
    def __init__(self):

        self.fov_hor_deg = None
        self.fov_ver_deg = None
        self.center = None
        self.eye = None
        self.up = None
        self.width_px = None
        self.height_px = None

    def set_params(self, fov_deg=90, center=(0, 0, 0), eye=(1, 1, 1), width_px=640, height_px=480, up=(0, 1, 0)):

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

        self.ver_1_pos = intersect_line_plane_v3(self.eye, ver_1_angled_pos, self.center, np.cross(self.hor_1_pos - self.center, self.up))
        self.ver_2_pos = intersect_line_plane_v3(self.eye, ver_2_angled_pos, self.center, np.cross(self.hor_2_pos - self.center, self.up))



    def calculate_bounding_frustum(self):

        self.corner_11 = self.hor_1_pos + self.ver_1_pos - self.center
        self.corner_12 = self.hor_1_pos + self.ver_2_pos - self.center
        self.corner_21 = self.hor_2_pos + self.ver_1_pos - self.center
        self.corner_22 = self.hor_2_pos + self.ver_2_pos - self.center

