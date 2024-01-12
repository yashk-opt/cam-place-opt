import numpy as np
import open3d as o3d
import pathlib
from datetime import datetime
from collections import Counter


class Scene:
    def __init__(self, filepath, obj_type="mesh"):

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

        # TODO: There might be free space points beyond walls. One way to remove some of them would be to find
        #  the minimum bounding box (maybe convex hull) and pre-process to remove the free space points out of that
        #  bounding box

        pass

    def initialize_objects(self, voxel_size=30):

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

    def convert_mesh_pointcloud(self, filesave=False, filename=None, filepath=None):
        """Converts mesh object to pointcloud and saves it, if required."""

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
        """Converts mesh object to pointcloud and saves it, if required.
        filename is 'pointcloud_{datetime}' if filename is set to 'auto'.
        file is stored in the data folder if filepath set to auto.
        Assume first three columns are points, last three are normals.
        If colors_present is set to True, then middle three are taken as colors"""

        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(self.mesh, voxel_size=voxel_size)
        voxels_all = voxel_grid.get_voxels()
        self.voxel_size = voxel_size
        self._save_object(voxels_all, filesave, filename, filepath)

        return voxel_grid

    def find_voxel_centers(self, negative_grid=False):
        """Provides coordinates of the voxel_grid. If negative_grid is True, then it will find voxel
        centres of the free space"""

        voxels_all = self.voxel_grid.get_voxels()
        voxel_size = self.voxel_grid.voxel_size
        voxel_tensor = np.zeros(tuple(self.voxel_grid_dim))

        if negative_grid is False:
            voxel_centers = [voxel.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels_all]

        else:
            for voxel in voxels_all:
                voxel_tensor[tuple(voxel.grid_index)] = 1

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
        self.pcd_center = np.mean(self.pcd.points, axis=0)
        self.mesh_center = self.pcd_center

    @staticmethod
    def _save_object(obj, filesave=False, filename=None, filepath=None):

        if filesave is True:
            np.savetxt(f'{filepath}/{filename}', obj)
        return
