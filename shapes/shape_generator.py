import trimesh
import random
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing


def make_box():
    box_mesh = trimesh.creation.box((random.uniform(0.025, 0.08), random.uniform(0.025, 0.08), 0.05))
    return box_mesh


def make_triangle_prism():
    s1_length = random.uniform(0.025, 0.07)

    ring = LinearRing([(0.0, 0.0), (s1_length, 0.0), (random.uniform(-s1_length*0.4, s1_length*1.4), random.uniform(0.02, 0.06))])
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


def save_image(shape_name, png):
    f = open(f"shape_images/{shape_name}.png", "wb")
    f.write(png)
    f.close()


def save_obj(shape_name, mesh):
    f = open(f"shape_objs/shape/{shape_name}.stl", "wb")
    mesh.export(f, "stl")
    f.close()


def generate_shape_images(shapes, resolution=32):
    scene = trimesh.Scene()
    i = 0

    for shape in shapes:
        scene.add_geometry(shape, node_name="zoink")
        scene.set_camera(distance=0.15)
        png = scene.save_image(resolution=(resolution, resolution))
        save_image(f"shape{i}", png)
        scene.delete_geometry("zoink")
        i += 1


def save_shapes(shapes):
    i = 0
    for shape in shapes:
        save_obj(f"shape{i}", shape)
        i += 1


def main():
    shapes = generate_shapes(100, 100, 100)
    generate_shape_images(shapes)



if __name__ == "__main__":
    main()
