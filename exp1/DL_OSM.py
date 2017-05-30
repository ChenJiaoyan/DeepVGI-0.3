#! /usr/bin/python

import os
import sys
import random
import getopt
import gc
import numpy as np
from scipy import misc

sys.path.append("../lib")
import NN_Model
import FileIO
import OSM

def read_test_sample(n, test_imgs, osm_p_imgs, osm_n_imgs):
    img_X = np.zeros((n, 256, 256, 3))
    label = np.zeros((n, 2))
    img_dir = '../data/image_guinea/'
    random.shuffle(test_imgs)
    i = 0
    te_p, te_n = 0, 0
    for img in test_imgs:
        if i >= n:
            break
        if os.path.exists(os.path.join(img_dir, img)):
            img_X[i] = misc.imread(os.path.join(img_dir, img))
        i1, i2 = img.index('-'), img.index('.')
        task_x, task_y = img[0:i1], img[(i1 + 1):i2]
        if [int(task_x), int(task_y)] in osm_p_imgs:
            label[i, 1] = 1
            i += 1
            te_p += 1
        elif [int(task_x), int(task_y)] in osm_n_imgs:
            label[i, 0] = 1
            i += 1
            te_n += 1
    print 'positive testing samples: %d \n ' % te_p
    print 'negative testing samples: %d \n ' % te_n
    return img_X, label

def read_train_sample(n1, n0, train_imgs, osm_p_imgs, osm_n_imgs):
    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))
    label = np.zeros((n1 + n0, 2))
    img_dir = '../data/image_guinea/'

    osm_po_imgs, osm_ne_imgs = [], []
    for img in train_imgs:
        i1, i2 = img.index('-'), img.index('.')
        task_x, task_y = img[0:i1], img[(i1 + 1):i2]
        k =  [int(task_x), int(task_y)]
        if k in osm_p_imgs:
            osm_po_imgs.append(img)
        elif k in ms_n_imgs:
            osm_ne_imgs.append(img)

    print 'osm_po_imgs: %d \n' % len(osm_po_imgs)
    print 'osm_ne_imgs: %d \n' % len(osm_ne_imgs)

    osm_po_imgs = random.sample(osm_po_imgs, n1)
    for i, img in enumerate(osm_po_imgs):
        img_X1[i] = misc.imread(os.path.join(img_dir, img))
    label[0:n1, 1] = 1

    osm_ne_imgs = random.sample(osm_ne_imgs, n0)
    for i, img in enumerate(osm_ne_imgs):
        img_X0[i] = misc.imread(os.path.join(img_dir, img))
    label[n1:(n1 + n0), 0] = 1

    j = range(n1 + n0)
    random.shuffle(j)
    X = np.concatenate((img_X1, img_X0))
    return X[j], label[j]

def deal_args(my_argv):
    v, n1, n0, b, e, t, c, z = False, 100, 100, 30, 1000, 8, 0, 200
    m = 'lenet'
    try:
        opts, args = getopt.getopt(my_argv, "vhy:n:b:e:t:c:z:m:",
                                   ["p_sample_size=", "n_sample_size=", "batch_size=", "epoch_num=", "thread_num=",
                                    "cv_round=", 'test_size=', 'network_model='])
    except getopt.GetoptError:
        print 'DL_MS.py -v -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
              '-c <cv_round>, -z <test_size>, -m <network_model>'
        print 'default settings: v=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, c=%d, z=%d, m=%s' % (v, n1, n0, b, e, t, c, z, m)
    for opt, arg in opts:
        if opt == '-h':
            print 'DL_OSM.py -v -y <p_sample_size> -n <n_sample_size> -b <batch_size> -e <epoch_num> -t <thread_num>, ' \
                  '-c <cv_round>, -z <test_size>, -m <network_model>'
            sys.exit()
        elif opt == '-v':
            v = True
        elif opt in ("-y", "--p_sample_size"):
            n1 = int(arg)
        elif opt in ("-n", "--n_sample_size"):
            n0 = int(arg)
        elif opt in ("-b", "--batch_size"):
            b = int(arg)
        elif opt in ("-e", "--epoch_num"):
            e = int(arg)
        elif opt in ("-t", "--thread_num"):
            t = int(arg)
        elif opt in ("-c", "--cv_round"):
            c = int(arg)
        elif opt in ("-z", "--test_size"):
            z = int(arg)
        elif opt in ("-m", "--network_model"):
            m = arg
    print 'settings: v=%s, n1=%d, n0=%d, b=%d, e=%d, t=%d, c=%d, z=%d, m=%s' % (v, n1, n0, b, e, t, c, z, m)
    return v, n1, n0, b, e, t, c, z, m


if __name__ == '__main__':
    evaluate_only, tr_n1, tr_n0, tr_b, tr_e, tr_t, cv_i, te_n, nn = deal_args(sys.argv[1:])
    cv_n = 4

    print '--------------- Read Samples ---------------'
    client = OSM.MSClient()
    train_imgs, test_imgs = client.imgs_cross_validation(cv_i, cv_n)
    osm_p_imgs = client.read_p_images()
    osm_n_imgs = client.read_n_images()
    print 'train_imgs: %d \n' % len(train_imgs)
    print 'osm_p_imgs: %d \n' % len(osm_p_imgs)
    print 'osm_n_imgs: %d\n' % len(osm_n_imgs)
    img_X, Y = read_train_sample(tr_n1, tr_n0, train_imgs, osm_p_imgs, osm_n_imgs)
    m = NN_Model.Model(img_X, Y, nn + '_ZY')

    if not evaluate_only:
        print '--------------- Training on MS Labels---------------'
        m.set_batch_size(tr_b)
        m.set_epoch_num(tr_e)
        m.set_thread_num(tr_t)
        m.train(nn)
        print '--------------- Evaluation on Training Samples ---------------'
        m.evaluate()
    del img_X, Y, train_imgs
    gc.collect()

    print '--------------- Evaluation on MapSwipe Samples ---------------'
    osm_p_imgs = client.read_p_images()
    osm_n_imgs = client.read_n_images()
    img_X2, Y2 = read_test_sample(te_n, test_imgs, osm_p_imgs, osm_n_imgs)
    m.set_evaluation_input(img_X2, Y2)
    m.evaluate()

