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
import sample_client

sample_dir = '../samples0/'

def read_train_sample(n1, n0):
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
        img_X1[i] = misc.imread(os.path.join(sample_dir, img))
    label[0:n1, 1] = 1

    OSM_train_n = random.sample(OSM_train_n, n0)
    for i, img in enumerate(OSM_train_n):
        img_X0[i] = misc.imread(os.path.join(sample_dir, img))
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

    print '--------------- Read Samples ---------------'
    img_X, Y = read_train_sample(tr_n1, tr_n0)
    m = NN_Model.Model(img_X, Y, nn + '_ZY')

    if not evaluate_only:
        print '--------------- Training on OSM Labels---------------'
        m.set_batch_size(tr_b)
        m.set_epoch_num(tr_e)
        m.set_thread_num(tr_t)
        m.train(nn)
        print '--------------- Evaluation on Training Samples ---------------'
        m.evaluate()
    del img_X, Y
    gc.collect()

    print '--------------- Evaluation on Validation Samples ---------------'
    img_X2, Y2 = FileIO.read_valid_sample(te_n)
    m.set_evaluation_input(img_X2, Y2)
    m.evaluate()
    del img_X2, Y2
    gc.collect()


