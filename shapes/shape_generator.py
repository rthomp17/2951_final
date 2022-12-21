import trimesh
import random
from shapely.geometry import Polygon
from shapely.geometry.polygon import LinearRing
import pickle
import numpy as np


def make_box(testing):
    if testing:
        param_a = random.uniform(0.010, 0.09)
        param_b = random.uniform(0.010, 0.09)
    else:
        param_a = random.uniform(0.025, 0.08)
        param_b = random.uniform(0.025, 0.08)
    box_mesh = trimesh.creation.box((param_a, param_b, 0.05))
    return box_mesh, (param_a, param_b)


def make_triangle_prism(testing):
    if testing:
        s1_length = random.uniform(0.015, 0.08)
        param_b = random.uniform(-s1_length * 0.4, s1_length * 1.4)
        param_c = random.uniform(0.01, 0.07)
    else:
        s1_length = random.uniform(0.025, 0.07)
        param_b = random.uniform(-s1_length*0.4, s1_length*1.4)
        param_c = random.uniform(0.02, 0.06)

    ring = LinearRing([(0.0, 0.0), (s1_length, 0.0), (param_b, param_c)])
    triangle = Polygon(ring)

    prism_mesh = trimesh.creation.extrude_polygon(triangle, 0.05)
    return prism_mesh, (s1_length, param_b, param_c)


def make_cylinder(testing):
    param_a = random.uniform(0.015, 0.035)
    cylinder_mesh = trimesh.creation.cylinder(param_a, 0.05)
    return cylinder_mesh, (param_a)


def generate_shapes(n_cyl=3, n_tri=3, n_box=3, testing=False):
    shapes = list()
    for _ in range(n_cyl):
        shapes.append(make_cylinder(testing))
    for _ in range(n_tri):
        shapes.append(make_triangle_prism(testing))
    for _ in range(n_box):
        shapes.append(make_box(testing))
    return shapes


def save_image(shape_name, png, shape_key='train'):
    f = open(f"shape_images/{shape_name}.png", "wb")
    f.write(png)
    f.close()

    if shape_key == 'train':
        f = open(f"vae_training_images/shape/{shape_name}.png", "wb")
        f.write(png)
        f.close()


def save_obj(shape_name, mesh):
    f = open(f"shape_objs/{shape_name}.stl", "wb")
    mesh.export(f, "stl")
    f.close()


def generate_shape_images(shapes, resolution=32, shape_key='train'):
    scene = trimesh.Scene()
    i = 0

    for shape in shapes:
        scene.add_geometry(shape, node_name="zoink")
        scene.set_camera(distance=0.15)
        png = scene.save_image(resolution=(resolution, resolution))
        save_image(f"{shape_key}-shape-{i}", png, shape_key)
        scene.delete_geometry("zoink")
        i += 1


def save_shapes(shapes, shape_key='train'):
    i = 0
    for shape in shapes:
        save_obj(f"{shape_key}-shape-{i}", shape)
        i += 1


def save_params(params, shape_key='train'):
    with open(f'shape_parameters_{shape_key}.txt', 'w') as f:
        i = 0
        for p in params:
            f.write(f"{shape_key}-shape-{i}:{p}\n")
            i += 1


def show_all_shapes(train_shapes, test_shapes):
    scene = trimesh.Scene()

    for i, shape in enumerate(train_shapes):
        transform = np.array([[1, 0, 0, ((0.1 * i)%1)*0.8],
                              [0, 1, 0, int(0.1 * i) * 0.09],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        scene.add_geometry(shape, node_name=f"zoink{i}", transform=transform)

    for i, shape in enumerate(test_shapes):
        transform = np.array([[1, 0, 0, ((0.1 * i)%1)*0.8],
                              [0, 1, 0, -int(0.1 * i) * 0.07 - 0.15],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        scene.add_geometry(shape, node_name=f"zoinky{i}", transform=transform)

    scene.set_camera(distance=0.85)

        # scene.delete_geometry("zoink")

    png = scene.save_image(resolution=(512, 420))

    f = open("all_objects.png", "wb")
    f.write(png)
    f.close()


def main():
    # shapes = generate_shapes(0, 20, 20, testing=False)
    # generate_shape_images([s[0] for s in shapes], shape_key='train')
    # save_shapes([s[0] for s in shapes], shape_key='train')
    # save_params([s[1] for s in shapes], shape_key='train')
    # pickle.dump(shapes, open('shape_parameters_train.pkl', 'wb'))
    #
    # shapes = generate_shapes(0, 10, 10, testing=True)
    # generate_shape_images([s[0] for s in shapes], shape_key='test')
    # save_shapes([s[0] for s in shapes], shape_key='test')
    # save_params([s[1] for s in shapes], shape_key='test')
    # pickle.dump(shapes, open('shape_parameters_test.pkl', 'wb'))

    train_shapes = pickle.load(open('shape_parameters_train.pkl', 'rb'))
    test_shapes = pickle.load(open('shape_parameters_test.pkl', 'rb'))

    show_all_shapes([s[0] for s in train_shapes], [s[0] for s in test_shapes])

    # shapes = generate_shapes(0, 20, 20, testing=False)
    #
    # shapes[0][0].show()


if __name__ == "__main__":
    main()
