import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import FLAGS_CMNIST, FLAGS_CIFAR
from train import args

flags = FLAGS_CMNIST()
if args.dataset == 'CMNIST':
    flags = FLAGS_CMNIST()
elif args.dataset == 'CIFAR_10':
    flags = FLAGS_CIFAR_10()
else:
    print('dataset error')


def get_CMNIST_train():
    images_path = []
    for i in range(flags.len_dataset):
        tmp_path = '/home/asus/Workspace/dataset/cmnist_label/' + str(i) + ".jpg"
        images_path.append(tmp_path)
        # images_path = tl.files.load_file_list(path='/home/asus/Workspace/dataset/cmnist_label/', regx='.*.jpg', keep_prefix=True, printable=False)

    targets = []
    with open('/home/asus/Workspace/dataset/label.txt', 'r') as f:
        numbers = f.readlines()
        for i in range(len(numbers)):
            targets.append(numbers[i])

    if len(images_path) != len(targets):
        raise AssertionError("The length of inputs and targets should be equal")

    def generator_train():
        for image_path, target in zip(images_path, targets):
            yield image_path.encode('utf-8'), target

    def _map_fn(image_path, target):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.image.crop_central(image, [FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])
        # image = tf.image.resize_images(image, FLAGS.output_size])
        # image = image[45:173, 25:153, :]  # central crop
        # image = tf.image.resize([image], (output_size, output_size))[0]
        # image = tf.image.crop_and_resize(image, boxes=[[]], crop_size=[64, 64])
        # image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.output_size, FLAGS.output_size) # central crop
        # image = tf.image.random_flip_left_right(image)
        image = image * 2 - 1
        target = tf.reshape(target, ())
        return image, target

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.string, tf.int32))
    ds = train_ds.shuffle(buffer_size=4096)
    # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
    n_step_epoch = int(flags.len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)
    ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(flags.batch_size_train)
    ds = ds.prefetch(buffer_size=2)
    return ds, images_path


def get_dataset_eval():
    if flags.dataset == 'MNIST':
        X_train, Y_train, _, _, X_test, _ = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1), path='../data')
        X_train = X_train * 2 - 1
    if flags.dataset == 'CIFAR_10':
        X_train, Y_train, X_test, _ = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), path='../data')
        X_train = X_train / 127.5 - 1

    def generator():
        for image, label in zip(X_train, Y_train):
            yield image, label

    def _map_fn(image, label):
        return image, label

    ds = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(flags.batch_size_eval)
    ds = ds.prefetch(buffer_size=4) # For concurrency
    return ds


def get_CIFAR10_train():
    X_train, y_train, _, _ = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

    def generator_train():
        inputs = X_train
        targets = y_train
        if len(inputs) != len(targets):
            raise AssertionError("The length of inputs and targets should be equal")
        for _input, _target in zip(inputs, targets):
            # yield _input.encode('utf-8'), _target.encode('utf-8')
            yield _input, _target

    def _map_fn_train(img, target):
        # 1. Randomly crop a [height, width] section of the image.
        # img = tf.image.random_crop(img, [24, 24, 3])
        # 2. Randomly flip the image horizontally.
        # img = tf.image.random_flip_left_right(img)
        # 3. Randomly change brightness.
        # img = tf.image.random_brightness(img, max_delta=63)
        # 4. Randomly change contrast.
        # img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        # img = tf.image.per_image_standardization(img)
        target = tf.reshape(target, ())
        return img, target

    ds = tf.data.Dataset.from_generator(
        generator_train, output_types=(tf.float32, tf.int32))  # , output_shapes=((32, 32, 3), (1)))
    ds = ds.shuffle(buffer_size=4096)
    n_step_epoch = int(flags.len_dataset // flags.batch_size_train)
    n_epoch = int(flags.step_num // n_step_epoch)
    ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn_train, num_parallel_calls=4)
    ds = ds.batch(flags.batch_size_train)
    ds = ds.prefetch(buffer_size=2)
    return ds
