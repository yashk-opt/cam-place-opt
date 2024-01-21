import numpy as np
import open3d as o3d
import pathlib
from datetime import datetime
from collections import Counter

from typing import Optional


class Scene:
    def __init__(self, filepath, obj_type="mesh"):
        """
        Initialize a Scene object.

        Parameters:
        - filepath (str): Path to the file containing the scene data.
        - obj_type (str, optional): Type of object to initialize ('mesh' by default).

        Returns:
        None
        """
        self.init_object = obj_type
        if self.init_object == "mesh":
            self.mesh = o3d.io.read_triangle_mesh(filepath, True)
        else:
            print(f"{obj_type} not implemented. Empty object initialized")
            self.mesh = None

        self.pcd = None
        self.voxel_grid = None
        self.voxel_grid_dim = None
        self.free_space_points = None
        self.voxel_tensor = None

        # TODO: There might be free space points beyond walls. One way to remove some of them would be to find
        #  the minimum bounding box (maybe convex hull) and pre-process to remove the free space points out of that
        #  bounding box

        self.pcd_center = None
        self.mesh_center = None
        self.voxel_size = None

        pass

    def initialize_objects(self, voxel_size=30):
        """
        Initializes objects based on the specified object type ('mesh' by default).
        Converts the mesh object to a point cloud, voxel grid, calculates voxel grid dimension, and finds free space
        points.

        Parameters:
        - voxel_size (float, optional): Size of the voxels in the voxel grid (default is 30).

        Returns:
        None
        """

        if self.init_object == "mesh":
            try:
                self.pcd = self.convert_mesh_pointcloud()
                print("point cloud implemented")

            except:
                print("point cloud not implemented")

            try:
                self.voxel_grid = self.convert_mesh_voxels(voxel_size=voxel_size)
                print("voxel_grid implemented")

            except:
                print("voxel_grid not implemented")

            try:
                self.voxel_grid_dim = self._voxel_grid_dim()
                print("voxel_grid_dimension calculated")

            except:
                print("voxel_grid_dimension not calculated")

            try:
                self.free_space_points = self.find_voxel_centers(negative_grid=True)
                print("free space calculated")
            except:
                print("free space not calculated")

    def camera_ready_mesh(self):

        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(
            np.vstack((np.asarray(self.mesh.vertices), self.free_space_points)))
        new_mesh.triangles = o3d.utility.Vector3iVector(self.mesh.triangles)

        return new_mesh

    def convert_mesh_pointcloud(self, filesave=False, filename: Optional[str] = None, filepath: Optional[str] = None):
        """
        Converts mesh object to point cloud and saves it, if required.

        Parameters:
        - filesave (bool, optional): Flag indicating whether to save the point cloud (default is False).
        - filename (str, optional): Name of the file to save the point cloud (used if filesave is True).
        - filepath (str, optional): Path to the directory where the point cloud file will be saved (used if filesave is
        True).

        Returns:
        o3d.geometry.PointCloud: Converted point cloud object.
        """

        pcd = o3d.geometry.PointCloud()
        pcd.points = self.mesh.vertices

        if self.mesh.has_vertex_colors():
            pcd.colors = self.mesh.vertex_colors
        elif self.mesh.has_vertex_colors():
            pcd.colors = self.mesh.vertex_normals

        pcd.normals = self.mesh.vertex_normals

        if filesave is True:
            point_cloud_array = np.hstack((pcd.points, pcd.colors, pcd.normals))
            self._save_object(point_cloud_array, filesave, filename, filepath)

        return pcd

    def convert_mesh_voxels(self, voxel_size, filesave=False, filename=None, filepath=None):
        """
        Converts mesh object to voxel grid and saves it if required.

        Parameters:
        - voxel_size (float): Size of the voxels in the grid.
        - filesave (bool, optional): Flag indicating whether to save the voxel grid (default is False).
        - filename (str or None, optional): Name of the file to save the voxel grid (used if filesave is True).
            If set to 'auto', the filename will be 'pointcloud_{datetime}'.
        - filepath (str or None, optional): Path to the directory where the voxel grid file will be saved (used if
        filesave is True).
            If set to 'auto', the file is stored in the data folder.

        Returns:
        o3d.geometry.VoxelGrid: Converted voxel grid object.
        """

        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(self.mesh, voxel_size=voxel_size)
        voxels_all = voxel_grid.get_voxels()
        self.voxel_size = voxel_size

        if filesave is True:
            self._save_object(voxels_all, filesave, filename, filepath)

        return voxel_grid

    def find_voxel_centers(self, negative_grid=False, save_voxel_tensor=True):
        """
        Provides coordinates of the voxel grid. If negative_grid is True, then it will find voxel centers of the free
        space.

        Parameters:
        - negative_grid (bool, optional): Flag indicating whether to find voxel centers of the free space (default is
        False).

        Returns:
        numpy.ndarray: Array containing coordinates of the voxel centers.
        """

        voxels_all = self.voxel_grid.get_voxels()
        voxel_size = self.voxel_grid.voxel_size
        voxel_tensor = np.zeros(tuple(self.voxel_grid_dim))

        for voxel in voxels_all:
            voxel_tensor[tuple(voxel.grid_index)] = 1

        if save_voxel_tensor is True:
            self.voxel_tensor = voxel_tensor

        if negative_grid is False:
            voxel_centers = np.array([voxel.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels_all])

        else:

            negative_grid_indices = list(zip(*np.where(voxel_tensor == 0)))

            for i, voxel in enumerate(voxels_all):
                if i == 0:
                    voxel_index_1 = voxel.grid_index
                    voxel_center_1 = self.voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

                elif i == 1:
                    voxel_index_2 = voxel.grid_index
                    voxel_center_2 = self.voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

                else:
                    break

            voxel_origin = voxel_center_1 - voxel_size * voxel_index_1
            assert max(voxel_origin - voxel_center_2 + voxel_size * voxel_index_2) < 0.001, "origin calculation error"

            voxel_centers = np.array([voxel_origin + voxel_size * np.array(negative_grid_index)
                                      for negative_grid_index in negative_grid_indices])

        return voxel_centers

    def _voxel_grid_dim(self):

        voxels_all = self.voxel_grid.get_voxels()
        index_array = np.array([voxel.grid_index for voxel in voxels_all])
        return np.max(index_array, axis=0) + 1

    def calculate_center(self):
        """
        Calculates the center of the point cloud and assigns it to the attributes pcd_center and mesh_center.

        Parameters:
        None

        Returns:
        None
        """
        self.pcd_center = np.mean(self.pcd.points, axis=0)
        self.mesh_center = self.pcd_center

    @staticmethod
    def _save_object(obj, filesave=False, filename=None, filepath=None):
        """
        Saves the given object to a file if filesave is True.

        Parameters:
        - obj: The object to be saved.
        - filesave (bool, optional): Flag indicating whether to save the object (default is False).
        - filename (str or None, optional): Name of the file to save the object (used if filesave is True).
        - filepath (str or None, optional): Path to the directory where the object file will be saved (used if filesave
        is True).

        Returns:
        None
        """
        if filesave is True:
            np.savetxt(f'{filepath}/{filename}', obj)
        return
