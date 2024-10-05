import open3d as o3d
import numpy as np
import json

def visualize_list(scene_list, **kwargs):

    print("press p for screencamera.json, d for depthcapture.json, o for renderoption.json. Just use d")
    vis = o3d.visualization.Visualizer()
    json_path = kwargs.get("json_path", -1)

    if json_path == -1:
        window_name = kwargs.get("window_name", "untitled")
        width = kwargs.get("window_width", 800)
        height = kwargs.get("window_height", 800)
        vis.create_window(window_name=window_name, width=width, height=height)

    else:
        with open(json_path, 'r') as file:
            cam_dict = json.load(file)
        vis.create_window(height=cam_dict["intrinsic"]["height"], width=cam_dict["intrinsic"]["width"])
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(json_path)

    for scene in scene_list:
        vis.add_geometry(scene)

    if json_path != -1:
        ctr.convert_from_pinhole_camera_parameters(param, True)
        vis.get_render_option().load_from_json(json_path)

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
        "black": np.array([0, 0, 0])/255,
    }

    color = kwargs.get("color", "dark_orchid")
    if type(color) is str:
        color = color_dict[color]

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