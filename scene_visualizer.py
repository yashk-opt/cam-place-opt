import numpy as np
import open3d as o3d
import time
from scene import Scene
import numpy as np
from camera import Camera
from view_calculations import calculate_camera_view


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


def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):

    assert not np.all(end == origin), "start and end point are same"
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size/17.5 * scale,
                                                  cone_height=size*0.2 * scale,
                                                  cylinder_radius=size/30 * scale,
                                                  cylinder_height=size*(1 - 0.2*scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return mesh


def create_coordinate_axes_mesh(origin, dir_x=np.array([1, 0, 0]), dir_y=np.array([0, 1, 0]), dir_z=np.array([0, 0, 1])):

    arrow_x = get_arrow(origin + dir_x)
    arrow_x.paint_uniform_color(np.array([1, 0, 0]))
    arrow_y = get_arrow(origin + dir_y)
    arrow_y.paint_uniform_color(np.array([0, 1, 0]))
    arrow_z = get_arrow(origin + dir_z)
    arrow_z.paint_uniform_color(np.array([0, 0, 1]))

    return arrow_x, arrow_y, arrow_z


def visualize_list(scene_list, **kwargs):

    vis = o3d.visualization.Visualizer()
    window_name = kwargs.get("window_name", "untitled")
    width = kwargs.get("window_width", 800)
    height = kwargs.get("window_height", 800)
    vis.create_window(window_name=window_name, width=width, height=height)

    for scene in scene_list:
        vis.add_geometry(scene)

    vis.run()
    vis.destroy_window()


def create_voxels_subset(voxel_grid, object_list, voxel_size=None, object_type="voxel", **kwargs):

    color_dict = {
        "turquoise": np.array([0, 255, 255])/255,
        "orange_red": np.array([255, 69, 0])/255,
        "lime_green": np.array([50, 205, 50])/255,
        "deep_pink": np.array([255, 20, 147])/255,
        "gold": np.array([255, 215, 0])/255,
        "royal_blue": np.array([65, 105, 225])/255,
        "dark_orchid": np.array([153, 50, 204])/255,
        "indian_red": np.array([205, 92, 92])/255,
        "spring_green": np.array([0, 255, 127])/255,
        "slate_blue": np.array([106, 90, 205])/255,
    }
    color = color_dict[kwargs.get("color", "dark_orchid")]

    if voxel_size is None:
        voxel_size = voxel_grid.voxel_size

    new_mesh = o3d.geometry.TriangleMesh()

    for object in object_list:
        # create a cube mesh with a size 1x1x1
        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        # paint it with the color of the current voxel
        cube.paint_uniform_color(color)
        # scale the box using the size of the voxel
        cube.scale(voxel_size, center=cube.get_center())
        # get the center of the current voxel
        if object_type == "voxel":
            voxel_center = voxel_grid.get_voxel_center_coordinate(object.grid_index)

        elif object_type == "point":
            voxel_center = object

        else:
            raise TypeError(f"{object_type} is not a a valid object type")

        # translate the box to the center of the voxel
        cube.translate(voxel_center, relative=False)
        # add the box to the TriangleMesh object
        new_mesh += cube

    return new_mesh
