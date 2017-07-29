#! /usr/bin/python

import os
if not os.getcwd().endswith('exp1'):
    os.chdir('exp1')

import sys
import gc
import random
import numpy as np
from scipy import misc

sys.path.append("../lib")
import NN_Model
import sample_client
import FileIO
import Parameters

sample_dir = '../samples0/'

def read_train_sample(n1, n0):
    client = sample_client.GPXclient()
    GPX_train_p = client.GPX_train_positive()
    GPX_train_n = client.MS_train_negative()

    print 'GPX_train_p: %d \n' % len(GPX_train_p)
    print 'GPX_train_n: %d \n' % len(GPX_train_n)

    if len(GPX_train_p) < n1:
        print 'n1 is set too large'
        sys.exit()

    if len(GPX_train_n) < n0:
        print 'n0 is set too large'
        sys.exit()

    img_X1, img_X0 = np.zeros((n1, 256, 256, 3)), np.zeros((n0, 256, 256, 3))
    label = np.zeros((n1 + n0, 2))

    GPX_train_p = random.sample(GPX_train_p, n1)
    for i, img in enumerate(GPX_train_p):
        img_X1[i] = misc.imread(os.path.join(sample_dir, 'train/MS_record/', img))
    label[0:n1, 1] = 1

    GPX_train_n = random.sample(GPX_train_n, n0)
    for i, img in enumerate(GPX_train_n):
        img_X0[i] = misc.imread(os.path.join(sample_dir, 'train/MS_negative/', img))
    label[n1:(n1 + n0), 0] = 1

    j = range(n1 + n0)
    random.shuffle(j)
    X = np.concatenate((img_X1, img_X0))
    return X[j], label[j]


if __name__ == '__main__':

    evaluate_only, tr_n1, tr_n0, tr_b, tr_e, tr_t, te_n, nn = Parameters.deal_args(sys.argv[1:])
    print '--------------- Read Samples ---------------'
    img_X, Y = read_train_sample(tr_n1, tr_n0)
    m = NN_Model.Model(img_X, Y, nn + '_ZY')

    if not evaluate_only:
        print '--------------- Training on GPX Labels---------------'
        m.set_batch_size(tr_b)
        m.set_epoch_num(tr_e)
        m.set_thread_num(tr_t)
        m.train(nn)
        print '--------------- Evaluation on Training Samples ---------------'
        m.evaluate()
    del img_X, Y
    gc.collect()

    print '--------------- Evaluation on Validation Samples ---------------'
    img_X2, Y2 = FileIO.read_gpx_valid_sample(te_n)
    m.set_evaluation_input(img_X2, Y2)
    m.evaluate()
    del img_X2, Y2
    gc.collect()
