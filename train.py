import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import FLAGS_CMNIST, FLAGS_CIFAR
from data import get_dataset_train, get_dataset_eval
from models import get_G, get_img_D, get_E, get_z_D
import random
import argparse
import math
import scipy.stats as stats
import sys

temp_out = sys.stdout  # 记录当前输出指向，默认是consle

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='DWGAN', help='train or eval')
parser.add_argument('--is_continue', type=bool, default=False, help='load weights from checkpoints?')
parser.add_argument('--dataset', type=str, default='CIFAR_10', help=['CMNIST', 'CIFAR_10'])
args = parser.parse_args()

def data_aug(images):
    z = np.random.normal(loc=0.0, scale=0.15,
                         size=[flags.batch_size_train, flags.img_size_h, flags.img_size_h, flags.c_dim]).astype(
        np.float32)

    return images + z


def KStest(real_z, fake_z):
    p_list = []
    for i in range(flags.batch_size_train):
        _, tmp_p = stats.ks_2samp(fake_z[i], real_z[i])
        p_list.append(tmp_p)
    return np.min(p_list), np.mean(p_list)


def train(con=False):
    dataset, len_dataset = get_dataset_train()
    len_dataset = flags.len_dataset
    G = get_G([None, flags.z_dim])
    D = get_img_D([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E = get_E([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    D_z = get_z_D([None, flags.z_dim])

    if con:
        G.load_weights('./checkpoint/G.npz')
        D.load_weights('./checkpoint/D.npz')
        E.load_weights('./checkpoint/E.npz')
        D_z.load_weights('./checkpoint/D_z.npz')

    G.train()
    D.train()
    E.train()
    D_z.train()

    n_step_epoch = int(len_dataset // flags.batch_size_train)
    n_epoch = flags.n_epoch

    # lr_G = flags.lr_G * flags.initial_scale
    # lr_E = flags.lr_E * flags.initial_scale
    # lr_D = flags.lr_D * flags.initial_scale
    # lr_Dz = flags.lr_Dz * flags.initial_scale

    lr_G = flags.lr_G
    lr_E = flags.lr_E
    lr_D = flags.lr_D
    lr_Dz = flags.lr_Dz

    # total_step = n_epoch * n_step_epoch
    # lr_decay_G = flags.lr_G * (flags.ending_scale - flags.initial_scale) / total_step
    # lr_decay_E = flags.lr_G * (flags.ending_scale - flags.initial_scale) / total_step
    # lr_decay_D = flags.lr_G * (flags.ending_scale - flags.initial_scale) / total_step
    # lr_decay_Dz = flags.lr_G * (flags.ending_scale - flags.initial_scale) / total_step

    d_optimizer = tf.optimizers.Adam(lr_D, beta_1=flags.beta1, beta_2=flags.beta2)
    g_optimizer = tf.optimizers.Adam(lr_G, beta_1=flags.beta1, beta_2=flags.beta2)
    e_optimizer = tf.optimizers.Adam(lr_E, beta_1=flags.beta1, beta_2=flags.beta2)
    dz_optimizer = tf.optimizers.Adam(lr_Dz, beta_1=flags.beta1, beta_2=flags.beta2)

    curr_lambda = flags.lambda_recon

    for step, batch_imgs_labels in enumerate(dataset):
        '''
        log = " ** new learning rate: %f (for GAN)" % (lr_v.tolist()[0])
        print(log)
        '''
        batch_imgs = batch_imgs_labels[0]
        # print("batch_imgs shape:")
        # print(batch_imgs.shape)  # (64, 64, 64, 3)
        batch_labels = batch_imgs_labels[1]
        # print("batch_labels shape:")
        # print(batch_labels.shape)  # (64,)
        epoch_num = step // n_step_epoch
        # for i in range(flags.batch_size_train):
        #    tl.visualize.save_image(batch_imgs[i].numpy(), 'train_{:02d}.png'.format(i))

        # # Updating recon lambda
        # if epoch_num <= 5:  # 50 --> 25
        #     curr_lambda -= 5
        # elif epoch_num <= 40:  # stay at 25
        #     curr_lambda = 25
        # else:  # 25 --> 10
        #     curr_lambda -= 0.25

        with tf.GradientTape(persistent=True) as tape:
            z = flags.scale * np.random.normal(loc=0.0, scale=flags.sigma * math.sqrt(flags.z_dim),
                                               size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            z += flags.scale * np.random.binomial(n=1, p=0.5,
                                                  size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            fake_z = E(batch_imgs)
            fake_imgs = G(fake_z)
            fake_logits = D(fake_imgs)
            real_logits = D(batch_imgs)
            fake_logits_z = D(G(z))
            real_z_logits = D_z(z)
            fake_z_logits = D_z(fake_z)

            e_loss_z = - tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.zeros_like(fake_z_logits)) + \
                       tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.ones_like(fake_z_logits))

            recon_loss = curr_lambda * tl.cost.absolute_difference_error(batch_imgs, fake_imgs)
            g_loss_x = - tl.cost.sigmoid_cross_entropy(fake_logits, tf.zeros_like(fake_logits)) + \
                       tl.cost.sigmoid_cross_entropy(fake_logits, tf.ones_like(fake_logits))
            g_loss_z = - tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.zeros_like(fake_logits_z)) + \
                       tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.ones_like(fake_logits_z))
            e_loss = recon_loss + e_loss_z
            g_loss = recon_loss + g_loss_x + g_loss_z

            d_loss = tl.cost.sigmoid_cross_entropy(real_logits, tf.ones_like(real_logits)) + \
                     tl.cost.sigmoid_cross_entropy(fake_logits, tf.zeros_like(fake_logits)) + \
                     tl.cost.sigmoid_cross_entropy(fake_logits_z, tf.zeros_like(fake_logits_z))

            dz_loss = tl.cost.sigmoid_cross_entropy(fake_z_logits, tf.zeros_like(fake_z_logits)) + \
                      tl.cost.sigmoid_cross_entropy(real_z_logits, tf.ones_like(real_z_logits))

        # Updating Encoder
        grad = tape.gradient(e_loss, E.trainable_weights)
        e_optimizer.apply_gradients(zip(grad, E.trainable_weights))

        # Updating Generator
        grad = tape.gradient(g_loss, G.trainable_weights)
        g_optimizer.apply_gradients(zip(grad, G.trainable_weights))

        # Updating Discriminator
        grad = tape.gradient(d_loss, D.trainable_weights)
        d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

        # Updating D_z & D_h
        grad = tape.gradient(dz_loss, D_z.trainable_weights)
        dz_optimizer.apply_gradients(zip(grad, D_z.trainable_weights))

        # # Updating lr
        # lr_G -= lr_decay_G
        # lr_E -= lr_decay_E
        # lr_D -= lr_decay_D
        # lr_Dz -= lr_decay_Dz

        # show current state
        if np.mod(step, flags.show_every_step) == 0:
            with open("log.txt", "a+") as f:
                p_min, p_avg = KStest(z, fake_z)
                sys.stdout = f  # 输出指向txt文件
                print("Epoch: [{}/{}] [{}/{}] curr_lambda: {:.5f}, recon_loss: {:.5f}, g_loss: {:.5f}, d_loss: {:.5f}, "
                      "e_loss: {:.5f}, dz_loss: {:.5f}, g_loss_x: {:.5f}, g_loss_z: {:.5f}, e_loss_z: {:.5f}".format
                      (epoch_num, flags.n_epoch, step - (epoch_num * n_step_epoch), n_step_epoch, curr_lambda,
                       recon_loss, g_loss, d_loss, e_loss, dz_loss, g_loss_x, g_loss_z, e_loss_z))
                print("kstest: min:{}, avg:{}".format(p_min, p_avg))
                
                sys.stdout = temp_out  # 输出重定向回console
                print("Epoch: [{}/{}] [{}/{}] curr_lambda: {:.5f}, recon_loss: {:.5f}, g_loss: {:.5f}, d_loss: {:.5f}, "
                      "e_loss: {:.5f}, dz_loss: {:.5f}, g_loss_x: {:.5f}, g_loss_z: {:.5f}, e_loss_z: {:.5f}".format
                      (epoch_num, flags.n_epoch, step - (epoch_num * n_step_epoch), n_step_epoch, curr_lambda,
                       recon_loss, g_loss, d_loss, e_loss, dz_loss, g_loss_x, g_loss_z, e_loss_z))
                print("kstest: min:{}, avg:{}".format(p_min, p_avg))

        if np.mod(step, n_step_epoch) == 0 and step != 0:
            G.save_weights('{}/{}/G.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D.save_weights('{}/{}/D.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            E.save_weights('{}/{}/E.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            D_z.save_weights('{}/{}/Dz.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
            # G.train()

        if np.mod(step, flags.eval_step) == 0:
            z = np.random.normal(loc=0.0, scale=1, size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
            G.eval()
            result = G(z)
            G.train()
            tl.visualize.save_images(result.numpy(), [8, 8],
                                     '{}/{}/train_{:02d}_{:04d}.png'.format(flags.sample_dir, flags.param_dir,
                                                                            step // n_step_epoch, step))
        del tape


class Retrival_Obj():
    def __init__(self, hash, label):
        self.label = label
        self.dist = 0
        list1 = [True if hash[i] == 1 else False for i in range(len(hash))]
        # convert bool list to bool array
        self.hash = np.array(list1)

    def __repr__(self):
        return repr((self.hash, self.label, self.dist))


# to calculate the hamming dist between obj1 & obj2


def hamming(obj1, obj2):
    res = obj1.hash ^ obj2.hash
    ans = 0
    for k in range(len(res)):
        if res[k] == True:
            ans += 1
    obj2.dist = ans


def take_ele(obj):
    return obj.dist


# to get 'nearest_num' nearest objs from 'image' in 'Gallery'
def get_nearest(image, Gallery, nearest_num):
    for obj in Gallery:
        hamming(image, obj)
    Gallery.sort(key=take_ele)
    ans = []
    cnt = 0
    for obj in Gallery:
        cnt += 1
        if cnt <= nearest_num:
            ans.append(obj)
        else:
            break

    return ans


# given retrivial_set, calc AP w.r.t. given label
def calc_ap(retrivial_set, label):
    total_num = 0
    ac_num = 0
    ans = 0
    result = []
    for obj in retrivial_set:
        total_num += 1
        if obj.label == label:
            ac_num += 1
        ans += ac_num / total_num
        result.append(ac_num / total_num)
    result = np.array(result)
    ans = np.mean(result)
    return ans


def Evaluate_mAP():
    print('Start Eval!')
    # load images & labels
    ds = get_dataset_eval()
    E = get_E([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E.load_weights('./checkpoint/E.npz')
    E.eval()

    # create (hash,label) gallery
    Gallery = []
    cnt = 0
    step_time1 = time.time()
    for batch, label in ds:
        cnt += 1
        if cnt % flags.eval_print_freq == 0:
            step_time2 = time.time()
            print("Now {} Imgs done, takes {:.3f} sec".format(cnt, step_time2 - step_time1))
            step_time1 = time.time()
        hash_fake, _ = E(batch)
        hash_fake = hash_fake.numpy()[0]
        hash_fake = ((tf.sign(hash_fake * 2 - 1, name=None) + 1) / 2).numpy()
        label = label.numpy()[0]
        Gallery.append(Retrival_Obj(hash_fake, label))
    print('Hash calc done, start split dataset')

    # sample 1000 from Gallery and bulid the Query set
    random.shuffle(Gallery)
    cnt = 0
    Queryset = []
    G = []
    for obj in Gallery:
        cnt += 1
        if cnt > flags.eval_sample:
            G.append(obj)
        else:
            Queryset.append(obj)
    Gallery = G
    print('split done, start eval')

    # Calculate mAP
    Final_mAP = 0
    step_time1 = time.time()
    for eval_epoch in range(flags.eval_epoch_num):
        result_list = []
        cnt = 0
        for obj in Queryset:
            cnt += 1
            if cnt % flags.retrieval_print_freq == 0:
                step_time2 = time.time()
                print("Now Steps {} done, takes {:.3f} sec".format(eval_epoch, cnt, step_time2 - step_time1))
                step_time1 = time.time()

            retrivial_set = get_nearest(obj, Gallery, flags.nearest_num)
            result = calc_ap(retrivial_set, obj.label)
            result_list.append(result)
        result_list = np.array(result_list)
        temp_res = np.mean(result_list)
        print("Query_num:{}, Eval_step:{}, Top_k_num:{}, AP:{:.3f}".format(flags.eval_sample, eval_epoch,
                                                                           flags.nearest_num, temp_res))
        Final_mAP += temp_res / flags.eval_epoch_num
    print('')
    print("Query_num:{}, Eval_num:{}, Top_k_num:{}, mAP:{:.3f}".format(flags.eval_sample, flags.eval_epoch_num,
                                                                       flags.nearest_num, Final_mAP))
    print('')


def Evaluate_Cluster():
    return 0


if __name__ == '__main__':
    # To choose flags
    flags = FLAGS_CMNIST()
    if args.dataset == 'CMNIST':
        flags = FLAGS_CMNIST()
    elif args.dataset == 'CIFAR_10':
        flags = FLAGS_CIFAR_10()
    else:
        print('dataset error')

    # To make sure path is legal
    tl.files.exists_or_mkdir(flags.checkpoint_dir + '/' + flags.param_dir)  # checkpoint path
    tl.files.exists_or_mkdir(flags.sample_dir + '/' + flags.param_dir)  # samples path

    tl.files.exists_or_mkdir(flags.checkpoint_dir)  # save model
    tl.files.exists_or_mkdir(flags.sample_dir)  # save generated image

    # Start training process
    train(con=args.is_continue)
