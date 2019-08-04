import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import Lambda, Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, BatchNorm, Concat, GaussianNoise
from utils import WeightNorm
from train import args
from config import FLAGS_CMNIST, FLAGS_CIFAR


flags = FLAGS_CMNIST()
if args.dataset == 'CMNIST':
    flags = FLAGS_CMNIST()
elif args.dataset == 'CIFAR_10':
    flags = FLAGS_CIFAR_10()
else:
    print('model error')


def get_G_cmnist(shape_z):    # Dimension of gen filters in first conv layer. [64]
    # input: (100,)
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    nz = Input(shape_z)
    n = Dense(n_units=3136, act=tf.nn.relu, W_init=w_init)(nz)
    n = Reshape(shape=[-1, 14, 14, 16])(n)
    n = DeConv2d(64, (5, 5), strides=(2, 2), W_init=w_init, b_init=None)(n) # (1, 28, 28, 64)
    n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(n)
    n = DeConv2d(32, (5, 5), strides=(1, 1), padding="VALID", W_init=w_init, b_init=None)(n) # (1, 32, 32, 32)
    n = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(n)
    n = DeConv2d(flags.c_dim, (5, 5), strides=(2, 2), act=tf.nn.tanh, W_init=w_init)(n)  # (1, 64, 64, 3)
    return tl.models.Model(inputs=nz, outputs=n, name='generator_CMNIST')


def get_img_D_cmnist(shape):
    df_dim = 8
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    ni = Input(shape)
    n = Conv2d(df_dim, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 2, (5, 5), (1, 1), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 4, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (5, 5), (1, 1), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    nf = Flatten(name='flatten')(n)
    n = Dense(n_units=1, act=None, W_init=w_init)(nf)
    return tl.models.Model(inputs=ni, outputs=n, name='img_Discriminator_CMNIST')


def get_E_cmnist(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    ni = Input(shape)   # (1, 64, 64, 3)
    n = Conv2d(3, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)  # (1, 32, 32, 3)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(32, (5, 5), (1, 1), padding="VALID", act=None, W_init=w_init, b_init=None)(n)  # (1, 28, 28, 32)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(64, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)  # (1, 14, 14, 64)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Flatten(name='flatten')(n)
    nz = Dense(n_units=flags.z_dim, act=None, W_init=w_init)(n)
    return tl.models.Model(inputs=ni, outputs=nz, name='encoder_CMNIST')


def get_z_D_cmnist(shape_z):
    gamma_init = tf.random_normal_initializer(1., 0.02)
    w_init = tf.random_normal_initializer(stddev=0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    nz = Input(shape_z)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(nz)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=1, act=None, W_init=w_init)(n)
    return tl.models.Model(inputs=nz, outputs=n, name='z_Discriminator_CMNIST')


def get_G_cifar_10(shape_z):    # Dimension of gen filters in first conv layer. [64]
    # input: (flags.z_dim,)
    w_init = tf.random_normal_initializer(stddev=0.02)
    ni = Input(shape_z)
    n = Dense(n_units=128 * 4 * 4, act=tf.nn.relu, W_init=w_init)(ni)
    # res blocks
    nn = Conv2d(128, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = Conv2d(128, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
    n = Elementwise(tf.add)([n, nn])
    # res blocks
    nn = Conv2d(128, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = Conv2d(128, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
    n = Elementwise(tf.add)([n, nn])
    # res blocks
    nn = Conv2d(128, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=g_init)(nn)
    nn = Conv2d(128, (3, 3), (1, 1), act=None, W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=None, gamma_init=g_init)(nn)
    n = Elementwise(tf.add)([n, nn])
    n = Conv2d(3, (3, 3), (1, 1), act=tf.nn.relu, W_init=w_init)(n)
    return tl.models.Model(inputs=nz, outputs=n, name='generator_CIFAR10')


def get_img_D_cifar10(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    df_dim = 64
    ni = Input(shape)
    n = Lambda(lambda x: tf.image.random_crop(x, [flags.batch_size, 24, 24, 3]))(ni)  # patchGAN

    # need to be implemented

    n = Dense(n_units=1, act=None, W_init=w_init)(n)
    return tl.models.Model(inputs=ni, outputs=n, name='img_Discriminator_CIFAR10')


def get_E_cifar10(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    ni = Input(shape)   # (1, 64, 64, 3)
    n = Conv2d(3, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(ni)  # (1, 32, 32, 3)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(32, (5, 5), (1, 1), padding="VALID", act=None, W_init=w_init, b_init=None)(n)  # (1, 28, 28, 32)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(64, (5, 5), (2, 2), act=None, W_init=w_init, b_init=None)(n)  # (1, 14, 14, 64)
    n = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Flatten(name='flatten')(n)
    nz = Dense(n_units=flags.z_dim, act=None, W_init=w_init)(n)
    return tl.models.Model(inputs=ni, outputs=nz, name='encoder_CIFAR10')


def get_z_D_cifar10(shape_z):
    gamma_init = tf.random_normal_initializer(1., 0.02)
    w_init = tf.random_normal_initializer(stddev=0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)
    nz = Input(shape_z)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(nz)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=750, act=None, W_init=w_init, b_init=None)(n)
    n = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(n)
    n = Dense(n_units=1, act=None, W_init=w_init)(n)
    return tl.models.Model(inputs=nz, outputs=n, name='z_Discriminator_CIFAR10')



