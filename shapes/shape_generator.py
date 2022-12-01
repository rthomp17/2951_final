import trimesh
import random
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing


def make_box():
    box_mesh = trimesh.creation.box((random.uniform(0.03, 0.07), random.uniform(0.03, 0.07), 0.05))
    return box_mesh


def make_triangle_prism():
    s1_length = random.uniform(0.04, 0.06)

    ring = LinearRing([(0.0, 0.0), (s1_length, 0.0), (random.uniform(0.0, s1_length*1.2), random.uniform(0.04, 0.06))])
    triangle = Polygon(ring)

    prism_mesh = trimesh.creation.extrude_polygon(triangle, 0.05)
    return prism_mesh


def make_cylinder():
    cylinder_mesh = trimesh.creation.cylinder(random.uniform(0.015, 0.035), 0.05)
    return cylinder_mesh


def generate_shapes(n_cyl=3, n_tri=3, n_box=3):
    shapes = list()
    for _ in range(n_cyl):
        shapes.append(make_cylinder())
    for _ in range(n_tri):
        shapes.append(make_triangle_prism())
    for _ in range(n_box):
        shapes.append(make_box())
    return shapes


def generate_shape_images(shapes):
    scene = trimesh.Scene()
    i = 0

    for shape in shapes:
        scene.add_geometry(shape, node_name="zoink")
        scene.set_camera(distance=0.12)
        png = scene.save_image(resolution=(64, 64))
        f = open(f"shape_images/shape{i}.png", "wb")
        f.write(png)
        f.close()
        scene.delete_geometry("zoink")
        i += 1

