
import tensorflow as tf

tf.random.set_seed(42)
tf.compat.v1.enable_eager_execution()

import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 1
NUM_SAMPLES = 32
EPOCHS = 20
POS_ENCODE_DIMS = 16
H = IMAGE_HEIGHT = 64
W = IMAGE_WIDTH = 64
#tf.enable_eager_execution()
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()

def process_dynamics(state):
    #concatenates all the data together into a thingy
    vector = []
    keys = ['cubeA', 'cubeB', 'cubeC', 'cubeD']
    for obj in keys:
        
        vector += [state[obj]['x'],
        state[obj]['y'],
        state[obj]['th']]

    vector += [state['eef']['x'], state['eef']['y']]
    return vector

def load_images():
    data_dir = './images'
    pickle_filenames = [name for name in os.listdir('./gt')]
    dynamics_dict = {}
    for f in pickle_filenames:
        rollout_num = f[8]
        dynamics_dict[rollout_num] = pickle.load(open(os.path.join('./gt', f), 'rb'))

    filenames = [name for name in os.listdir(data_dir)]
    BATCH_SIZE = len(filenames)
    images = np.zeros((len(filenames), 64, 64, 3))
    dynamics = np.zeros((len(filenames), 14))
    for i in range(len(filenames)):

        name = filenames[i]
        rollout_num = name[4]
        step_num = int(name[6:-4])
        dynamics[i] = process_dynamics(dynamics_dict[rollout_num][step_num])
        images[i] = imageio.imread(os.path.join(data_dir, filenames[i]))
        mask = np.load(os.path.join('./masks', filenames[i][:-3] + 'npy'))

        images[i] = np.where(mask>0, images[i], np.array([255,255,255]))/255
    #         coords[i] = torch.from_numpy(points)
    #         rgb[i] = torch.from_numpy(bilinear_interpolate(images[i].T.numpy(), points))

    #         # new_img = np.array(rgb[i]).reshape(sample_extent, sample_extent, 3)/255
    #         # new_img = np.swapaxes(new_img, 0, 1)
    #     data_dict[obj] = [images, coords, rgb]
    return (tf.convert_to_tensor(images, dtype=tf.uint8), tf.convert_to_tensor(dynamics, dtype=tf.float32))

camera_pose_map = {'tableview': [[ 0.,         -0.99998082,  0.,          0.,        ],
                                  [ 0.99998082,  0.,          0.,          0.,        ],
                                  [ 0.,          0.,          0.99998082,  1.5,       ],
                                  [ 0.,          0.,          0.,          1.,        ]]}
camera_focal_map = {'tableview':  0.25 * IMAGE_HEIGHT / np.tan(45 * np.pi / 360)}


images, dynamics_data= load_images()
num_images = len(images)
(poses, focal) = ([camera_pose_map['tableview'] for _ in range(num_images)], 
                  camera_focal_map['tableview'])

# # Plot a random image from the dataset for visualization.
# plt.imshow(images[np.random.randint(low=0, high=num_images)])
# plt.show()

def encode_position(x):
    """Encodes the position into its corresponding Fourier feature.

    Args:
        x: The input coordinate.

    Returns:
        Fourier features tensors of the position.
    """
    positions = [x]
    for i in range(POS_ENCODE_DIMS):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0 ** i * x))
    return tf.concat(positions, axis=-1)

def get_rays(height, width, focal, pose):
    """Computes origin point and direction vector of rays.

    Args:
        height: Height of the image.
        width: Width of the image.
        focal: The focal length between the images and the camera.
        pose: The pose matrix of the camera.

    Returns:
        Tuple of origin point and direction vector for rays.
    """
    # Build a meshgrid for the rays.
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)

    # Get the camera matrix.
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    # Return the origins and directions.
    return (ray_origins, ray_directions)


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    """Renders the rays and flattens it.

    Args:
        ray_origins: The origin points for rays.
        ray_directions: The direction unit vectors for the rays.
        near: The near bound of the volumetric scene.
        far: The far bound of the volumetric scene.
        num_samples: Number of sample points in a ray.
        rand: Choice for randomising the sampling strategy.

    Returns:
       Tuple of flattened rays and sample points on each rays.
    """
    # Compute 3D query points.
    # Equation: r(t) = o+td -> Building the "t" here.
    t_vals = tf.linspace(near, far, num_samples)
    if rand:
        # Inject uniform noise into sample space to make the sampling
        # continuous.
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    # Equation: r(t) = o + td -> Building the "r" here.
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)


def get_nerf_model(num_layers, num_pos, num_dynamics=14):
    """Generates the NeRF neural network.

    Args:
        num_layers: The number of MLP layers.
        num_pos: The number of dimensions of positional encoding.

    Returns:
        The [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) model.
    """
    input1 = keras.Input(shape=(num_pos, 2 * 3 * POS_ENCODE_DIMS + 3))
    # input1 = layers.Dense(units=64, activation="relu")(input1)
    input2 = keras.Input(shape=(num_pos, num_dynamics,))
    inputs = layers.concatenate([input1, input2])
    x = inputs
    for i in range(num_layers):
        x = layers.Dense(units=64, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            # Inject residual connection.
            x = layers.concatenate([x, inputs], axis=-1)
    outputs = layers.Dense(units=4)(x)
    return keras.Model(inputs=[input1, input2], outputs=outputs)


def render_rgb_depth(model, rays_flat, dynamics, t_vals, rand=True, train=True):
    """Generates the RGB image and depth map from model prediction.

    Args:
        model: The MLP model that is trained to predict the rgb and
            volume density of the volumetric scene.
        rays_flat: The flattened rays that serve as the input to
            the NeRF model.
        t_vals: The sample points for the rays.
        rand: Choice to randomise the sampling strategy.
        train: Whether the model is in the training or testing phase.

    Returns:
        Tuple of rgb image and depth map.
    """
    # Get the predictions from the nerf model and reshape it.
    new_dynamics = tf.expand_dims(dynamics, 1)
    if train:
        predictions = model((rays_flat, tf.repeat(new_dynamics, rays_flat.shape[1], axis=1)))
    else:
        predictions = model.predict((rays_flat, tf.repeat(new_dynamics, rays_flat.shape[1], axis=1)))
    predictions = tf.reshape(predictions, shape=(BATCH_SIZE, H, W, NUM_SAMPLES, 4))

    # Slice the predictions into rgb and sigma.
    rgb = tf.sigmoid(predictions[..., :-1])
    sigma_a = tf.nn.relu(predictions[..., -1])

    # Get the distance of adjacent intervals.
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    # delta shape = (num_samples)
    if rand:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, H, W, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta)
    else:
        delta = tf.concat(
            [delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, 1))], axis=-1
        )
        alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])

    # Get transmittance.
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
    weights = alpha * transmittance
    rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

    if rand:
        depth_map = tf.reduce_sum(weights * t_vals, axis=-1)
    else:
        depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
    return (rgb, depth_map)

def map_fn(pose):
    """Maps individual pose to flattened rays and sample points.

    Args:
        pose: The pose matrix of the camera.

    Returns:
        Tuple of flattened rays and sample points corresponding to the
        camera pose.
    """
    (ray_origins, ray_directions) = get_rays(height=H, width=W, focal=focal, pose=pose)
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=NUM_SAMPLES,
        rand=True,
    )
    return (rays_flat, t_vals)


class NeRF(keras.Model):
    def __init__(self, nerf_model):
        super().__init__()
        self.nerf_model = nerf_model

    def call(self, inputs):
        self.train_step(inputs)

    def __call__(self, inputs):
        self.train_step(inputs)


    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")

    def train_step(self, inputs):
        # Get the images and the rays.
        (images, rays, dynamics) = inputs
        (rays_flat, t_vals) = rays

        with tf.GradientTape() as tape:
            # Get the predictions from the model.
            rgb, _ = render_rgb_depth(
                model=self.nerf_model, rays_flat=rays_flat, dynamics=dynamics, t_vals=t_vals, rand=True
            )
            loss = self.loss_fn(images, rgb)
        
        # Get the trainable variables.
        trainable_variables = self.nerf_model.trainable_variables

        # Get the gradeints of the trainiable variables with respect to the loss.
        gradients = tape.gradient(loss, trainable_variables)

        
        # Apply the grads and optimize the model.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    def test_step(self, inputs):
        # Get the images and the rays.
        (images, rays, dynamics) = inputs
        (rays_flat, t_vals) = rays

        # Get the predictions from the model.
        rgb, _ = render_rgb_depth(
            model=self.nerf_model, rays_flat=rays_flat, dynamics=dynamics, t_vals=t_vals, rand=True
        )
        loss = self.loss_fn(images, rgb)

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric]

# Create the training split.
split_index = int(num_images * 0.8)

# Split the images into training and validation.
train_images = images[:split_index]
train_dynamics = dynamics_data[:split_index]
val_images = images[split_index:]
val_dynamics = dynamics_data[:split_index]

# Split the poses into training and validation.
train_poses = poses[:split_index]
val_poses = poses[split_index:]

# Make the training pipeline.
train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
train_dynamics_ds = tf.data.Dataset.from_tensor_slices(train_dynamics)
train_ray_ds = train_pose_ds.map(map_fn, num_parallel_calls=AUTO)
training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds, train_dynamics_ds))
train_ds = (
    training_ds.shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)

# Make the validation pipeline.
val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
val_dynamics_ds = tf.data.Dataset.from_tensor_slices(val_dynamics)
val_ray_ds = val_pose_ds.map(map_fn, num_parallel_calls=AUTO)
validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds, val_dynamics_ds))
val_ds = (
    validation_ds.shuffle(BATCH_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
    .prefetch(AUTO)
)


test_imgs, test_rays, test_dynamics = next(iter(train_ds))
test_rays_flat, test_t_vals = test_rays

loss_list = []
# print(test_imgs.shape)
# print(test_rays)
# exit(0)

class TrainMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs["loss"]
        loss_list.append(loss)
        test_recons_images, depth_maps = render_rgb_depth(
            model=self.model.nerf_model,
            rays_flat=test_rays_flat,
            dynamics=test_dynamics,
            t_vals=test_t_vals,
            rand=True,
            train=False,
        )
        


        if epoch % 1 == 0:
            # Plot the rgb, depth and the loss plot.
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
            ax[0].imshow(keras.preprocessing.image.array_to_img(test_recons_images[0]))
            ax[0].set_title(f"Predicted Image: {epoch:03d}")

            ax[1].imshow(keras.preprocessing.image.array_to_img(test_imgs[0]))
            ax[1].set_title(f"True Image: {epoch:03d}")

            ax[2].plot(loss_list)
            ax[2].set_xticks(np.arange(0, EPOCHS + 1, 5.0))
            ax[2].set_title(f"Loss Plot: {epoch:03d}")

            #fig.savefig(f"images/{epoch:03d}.png")
            plt.show()
            plt.close()


num_pos = H * W * NUM_SAMPLES
print(num_pos)
nerf_model = get_nerf_model(num_layers=4, num_pos=num_pos)


model = NeRF(nerf_model)
model.compile(
    optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError(),
)
# model.build((1, 256, 256, 3))
# print(model.summary())

# # Create a directory to save the images during training.
# if not os.path.exists("images"):
#     os.makedirs("images")

model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=BATCH_SIZE,
    epochs=1,
    callbacks=[TrainMonitor()],
    #steps_per_epoch=split_index // BATCH_SIZE,
)

keras.save_model( model, 'test_model')

def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)


create_gif("images/*.png", "training.gif")
