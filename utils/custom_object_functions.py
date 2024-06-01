import open3d as o3d
import numpy as np
import copy
from utils.transformations import calculate_zy_rotation_for_arrow
from visualization.visualization_utils import visualize_list


def create_hollow_room(width, height, depth, num_walls=0, wall_edge_ratio=0.8, wall_width=1, wall_normal="x",
                       wall_slices=1, seed=42, random_range=0.0):
    room = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    vertices = np.array(room.vertices)
    center = room.get_center()

    # reverse triangle orientation so that normals are pointing inwards
    room.triangles = o3d.cpu.pybind.utility.Vector3iVector(np.flip(np.array(room.triangles), axis=1))
    room.compute_triangle_normals()

    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    min_z, max_z = np.min(vertices[:, 2]), np.max(vertices[:, 2])

    if wall_normal == "x":
        # Design wall
        wall = o3d.geometry.TriangleMesh.create_box(width=wall_width,
                                                    height=wall_edge_ratio * height,
                                                    depth=wall_edge_ratio * height)

        for num in range(wall_slices):
            wall_slice = (o3d.geometry.TriangleMesh.create_box(width=(2 * num + 1) * wall_width / (2 * wall_slices + 1),
                                                               height=wall_edge_ratio * height,
                                                               depth=wall_edge_ratio * height)
                          .translate(wall.get_center(), relative=False)
                          )
            wall = wall + wall_slice

        # Place wall
        rng = np.random.default_rng(seed)
        x_nominal = np.linspace(min_x, max_x, num=num_walls + 2)[1:-1]
        x_nominal = [nom + (rng.random() * 2 - 1) * random_range * (max_x - min_x) for nom in x_nominal]
        y_nominal = [wall_edge_ratio * max_y * 0.5] * len(x_nominal)
        z_nominal_dir = np.random.randint(2, size=len(y_nominal))
        z_nominal = [wall_edge_ratio * max_z * 0.5 if direction == 0
                     else max_z - wall_edge_ratio * max_z * 0.5
                     for direction in z_nominal_dir]

        for count in range(num_walls):
            wall_new = copy.deepcopy(wall).translate((x_nominal[count], y_nominal[count], z_nominal[count]),
                                                     relative=False)
            room = room + wall_new

    room.translate(np.array(-center), relative=True)
    lines = []
    for triangle in np.array(room.triangles):
        lines.extend([(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])])

    line_set = o3d.geometry.LineSet()
    vertices = np.array(room.vertices)
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # line_set.translate(np.array(-center), relative=True)

    return room, line_set


def develop_mesh_line_set(mesh):
    lines = []
    for triangle in np.array(mesh.triangles):
        lines.extend([(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])])

    line_set = o3d.geometry.LineSet()
    vertices = np.array(mesh.vertices)
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set


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