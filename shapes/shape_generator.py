import trimesh
import random
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing


def make_box():
    box_mesh = trimesh.creation.box((random.uniform(0.03, 0.07), random.uniform(0.03, 0.07), 0.05))
    return box_mesh


def make_triangle_prism():
    s1_length = random.uniform(0.04, 0.06)

    ring = LinearRing([(0.0, 0.0), (0.0, s1_length), (random.uniform(0.0, s1_length*1.2), random.uniform(0.04, 0.06))])
    triangle = Polygon(ring)

    prism_mesh = trimesh.creation.extrude_polygon(triangle, 0.05)
    return prism_mesh


def make_cylinder():
    cylinder_mesh = trimesh.creation.cylinder(random.uniform(0.015, 0.035), 0.05)
    return cylinder_mesh
