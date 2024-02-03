import open3d as o3d
import numpy as np

from scene_visualizer import calculate_zy_rotation_for_arrow


def create_hollow_room(width, height, depth):
    room = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    center = room.get_center()
    room.translate(np.array([0, 0, 0]), relative=False)

    lines = []
    for triangle in np.array(room.triangles):
        lines.extend([(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])])

    vertices = np.array(room.vertices)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    room.triangles = o3d.cpu.pybind.utility.Vector3iVector(np.flip(np.array(room.triangles), axis=1))
    room.compute_triangle_normals()

    return room, line_set


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
