import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, BatchNorm, Concat, GaussianNoise
from config import flags
from utils import WeightNorm
import matplotlib.pyplot as plt
import PIL
import scipy

# Read MNIST data
x_train, _, _, _, _, _ = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
x_train = x_train.astype(np.float32)


def get_mnist_batch(batch_size=64):
    # Select random batch (WxHxC)
    idx = np.random.choice(x_train.shape[0], batch_size)
    batch_raw = x_train[idx, :, :, 0].reshape((batch_size, 28, 28, 1))
    # Resize (this is optional but results in a training set of larger images)
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in batch_raw])
    # Create a new placeholder variable for our batch
    batch = np.zeros((batch_size, 64, 64, 3))

    for i in range(batch_size):
        # Take a random crop of the Lena image (background)
        r, g, b = np.random.normal(0, 0.5, 3)
        #print(r, g, b)
        # batch_resized[i]: (64, 64, 1)
        color_r = batch_resized[i] * r + (batch_resized[i] - 1)
        color_g = batch_resized[i] * g + (batch_resized[i] - 1)
        color_b = batch_resized[i] * b + (batch_resized[i] - 1)
        batch_rgb = np.concatenate([color_r, color_g, color_b], axis=2)
        # batch_rgb: (64, 64, 3)
        noise = np.random.normal(scale=0.2, size=(64, 64, 3))
        batch_rgb += noise
        batch_rgb /= np.max(np.abs(batch_rgb), axis=0)
        #print(batch_rgb)
        batch[i] = batch_rgb
        # Invert the colors at the location of the number

    return batch


count = 60000
examples = get_mnist_batch(count)
length = len(examples)
for i in range(length):
    img_path = "data/cmnist_new/" +str(i) + ".jpg"
    tl.visualize.save_image(examples[i], img_path)


