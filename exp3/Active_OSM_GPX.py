#! /usr/bin/python

import os

if not os.getcwd().endswith('exp3'):
    os.chdir('exp3')

import gc
import sys
import random
import numpy as np
from scipy import misc

sys.path.append("../lib")
import NN_Model
import FileIO
import Parameters
import sample_client

sample_dir = '../samples0/'


def read_train_sample_OSM(n1, n0):
    client = sample_client.OSMclient()
    OSM_train_p = client.OSM_train_positive()
    OSM_train_n = client.MS_train_negative()

    print 'OSM_train_p: %d \n' % len(OSM_train_p)
    print 'OSM_train_n: %d \n' % len(OSM_train_n)

    if len(OSM_train_p) < n1:
        print 'n1 is set too large'
        sys.exit()

    if len(OSM_train_n) < n0:
        print 'n0 is set too large'
        sys.exit()

    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))
    label = np.zeros((n1 + n0, 2))

    OSM_train_p = random.sample(OSM_train_p, n1)
    for i, img in enumerate(OSM_train_p):
        img_X1[i] = misc.imread(os.path.join(sample_dir, 'train/MS_record/', img))
    label[0:n1, 1] = 1

    OSM_train_n = random.sample(OSM_train_n, n0)
    for i, img in enumerate(OSM_train_n):
        img_X0[i] = misc.imread(os.path.join(sample_dir, 'train/MS_negative/', img))
    label[n1:(n1 + n0), 0] = 1

    j = range(n1 + n0)
    random.shuffle(j)
    X = np.concatenate((img_X1, img_X0))
    return X[j], label[j]


def active_sampling_GPX(m0, act_n, t_up, t_low, active_cache):
    gpx_client = sample_client.GPXclient()
    GPX_train_p = gpx_client.GPX_train_positive()
    MS_train_n = gpx_client.MS_train_negative()

    osm_client = sample_client.OSMclient()
    OSM_train_p = osm_client.OSM_train_positive()

    GPX_diff_OSM_train_p = list(set(GPX_train_p).difference(OSM_train_p))
    print 'GPX_diff_OSM_train_p: %d' % len(GPX_diff_OSM_train_p)
    if act_n / 2 <= len(GPX_diff_OSM_train_p) < active_cache:
        print 'active_cache is too large, set active_cache to %d' % len(GPX_diff_OSM_train_p)
        active_cache = len(GPX_diff_OSM_train_p)
    if len(GPX_diff_OSM_train_p) < act_n / 2:
        print 'active_cache/active_size is too large, set active_cache and active_size/2 to %d' % len(
            GPX_diff_OSM_train_p)
        active_cache = len(GPX_diff_OSM_train_p)
        act_n = len(GPX_diff_OSM_train_p) * 2
    GPX_diff_OSM_train_p = random.sample(GPX_diff_OSM_train_p, active_cache)

    img_X = np.zeros((len(GPX_diff_OSM_train_p), 256, 256, 3))
    for i, img in enumerate(GPX_diff_OSM_train_p):
        img_X[i] = misc.imread(os.path.join(sample_dir, 'train/MS_record/', img))

    m0.set_prediction_input(img_X)
    scores, _ = m0.predict()

    indexes = np.where((scores < t_up) & (scores > t_low))[0]
    if indexes.shape[0] < act_n / 2:
        print 'act_n/2 is larger than uncertain samples (%d)' % indexes.shape[0]
        print 'act_n is set to %d' % (indexes.shape[0] * 2)
        act_n = indexes.shape[0] * 2
    uncertain_img_X = img_X[indexes]
    j = range(indexes.shape[0])
    random.shuffle(j)
    positive_img_X = uncertain_img_X[j][0:act_n / 2]

    label_p = np.zeros((act_n / 2, 2))
    label_p[:, 1] = 1

    active_n = random.sample(MS_train_n, act_n / 2)
    negative_img_X = np.zeros((act_n / 2, 256, 256, 3))
    for i, img in enumerate(active_n):
        negative_img_X[i] = misc.imread(os.path.join(sample_dir, 'train/MS_negative/', img))
    label_n = np.zeros((act_n / 2, 2))
    label_n[:, 0] = 1

    return np.concatenate((negative_img_X, positive_img_X)), np.concatenate((label_n, label_p))


if __name__ == '__main__':
    evaluate_only, tr_n1, tr_n0, tr_b, tr_e, tr_t, te_n, nn, act_n, t_up, t_low, a_c = \
        Parameters.deal_args_active(sys.argv[1:])

    print '--------------- Read Samples ---------------'
    img_X, Y = read_train_sample_OSM(tr_n1, tr_n0)

    if not evaluate_only:
        print '--------------- M0: Training on OSM Labels---------------'
        m = NN_Model.Model(img_X, Y, nn + '_active_jy')
        m.set_batch_size(tr_b)
        m.set_epoch_num(tr_e)
        m.set_thread_num(tr_t)
        m.train(nn)
        print '--------------- M0: Evaluation on Training Samples ---------------'
        m.evaluate()

        print '--------------- Ma: Actively Sampling ---------------'
        img_Xa, Ya = active_sampling_GPX(m, act_n, t_up, t_low, a_c)
        img_X = np.concatenate((img_X, img_Xa))
        Y = np.concatenate((Y, Ya))
        index = range(img_X.shape[0])
        random.shuffle(index)
        img_X = img_X[index]
        Y = Y[index]

        print '--------------- Ma: Re-Training ---------------'
        m.set_XY(img_X, Y)
        m.re_learn()
        print '--------------- Ma: Evaluation on Training Samples ---------------'
        m.evaluate()
    else:
        m = NN_Model.Model(img_X, Y, nn + '_active_jy')

    del img_X, Y
    gc.collect()

    print '--------------- Ma: Evaluation on Validation Samples ---------------'
    img_X2, Y2 = FileIO.read_valid_sample(te_n)
    m.set_evaluation_input(img_X2, Y2)
    m.evaluate()
    del img_X2, Y2
    gc.collect()
