#! /usr/bin/python

import csv
import os
from scipy import misc
import numpy as np
import sample_client
import random

def csv_reader(file_name):
    cf = open(file_name)
    reader = csv.DictReader(cf)
    return reader


def read_lines(file_name, start_line):
    f = open(file_name)
    lines = f.readlines()
    f.close()
    return lines[start_line:]


def save_lines(file_name, lines):
    f = open(file_name, 'w')
    f.writelines(lines)
    f.close()
    return len(lines)


def read_external_test_img():
    lines = read_lines("../data/test_imgs.csv", 0)
    lines_p = read_lines("../data/test_positive_imgs.csv", 0)
    imgs_p, imgs_n = [], []
    for line in lines_p:
        imgs_p.append(line.strip())
    for line in lines:
        if line.strip() not in imgs_p:
            imgs_n.append(line.strip())
    return imgs_p, imgs_n


def read_external_test_sample():
    lines = read_lines("../data/test_imgs.csv", 0)
    lines_p = read_lines("../data/test_positive_imgs.csv", 0)
    imgs_p, imgs_n = [], []
    for line in lines_p:
        imgs_p.append(line.strip())
    for line in lines:
        if line.strip() not in imgs_p:
            imgs_n.append(line.strip())
    n = len(imgs_p) + len(imgs_n)
    img_X = np.zeros((n, 256, 256, 3))
    label = np.zeros((n, 2))
    dir1 = '../data/imagery/'
    i = 0
    img_files = []
    for img in imgs_p:
        if os.path.exists(os.path.join(dir1, img)):
            img_X[i] = misc.imread(os.path.join(dir1, img))
            label[i, 1] = 1
            img_files.append(img)
            i += 1
    n_p = i
    print 'positive external testing samples: %d \n' % n_p
    for img in imgs_n:
        if os.path.exists(os.path.join(dir1, img)):
            img_X[i] = misc.imread(os.path.join(dir1, img))
            label[i, 0] = 1
            img_files.append(img)
            i += 1
    n_n = i - n_p
    print 'negative external testing samples: %d \n' % n_n
    return img_X[0:i], label[0:i], img_files


def read_valid_sample(n):   # valid positive according to osm positive
    client = sample_client.OSMclient()
    MS_valid_p = client.OSM_valid_positive()
    MS_valid_n = client.MS_valid_negative()

    print 'MS_valid_p: %d \n' % len(MS_valid_p)
    print 'MS_valid_n: %d \n' % len(MS_valid_n)

    if len(MS_valid_p) < n/2 or len(MS_valid_p) < n/2:
        print 'n is set too large; use all the samples for testing'
        n = len(MS_valid_p) * 2

    img_X1, img_X0 = np.zeros((n/2, 256, 256, 3)), np.zeros((n/2, 256, 256, 3))
    MS_valid_p = random.sample(MS_valid_p, n/2)
    for i, img in enumerate(MS_valid_p):
        img_X1[i] = misc.imread(os.path.join('../samples0/valid/MS_record/', img))

    MS_valid_n = random.sample(MS_valid_n, n/2)
    for i, img in enumerate(MS_valid_n):
        img_X0[i] = misc.imread(os.path.join('../samples0/valid/MS_negative/', img))

    X = np.concatenate((img_X1[0:n/2], img_X0[0:n/2]))

    label = np.zeros((n, 2))
    label[0:n/2, 1] = 1
    label[n/2:n, 0] = 1

    return X, label

def read_gpx_valid_sample(n):
    client = sample_client.GPXclient()
    GPX_valid_p = client.GPX_valid_positive()
    GPX_valid_n = client.MS_valid_negative()

    print 'GPX_valid_p: %d \n' % len(GPX_valid_p)
    print 'GPX_valid_n: %d \n' % len(GPX_valid_n)

    if len(GPX_valid_p) < n / 2 or len(GPX_valid_p) < n / 2:
        print 'n is set too large; use all the samples for testing'
        n = len(GPX_valid_p) * 2

    img_X1, img_X0 = np.zeros((n / 2, 256, 256, 3)), np.zeros((n / 2, 256, 256, 3))
    GPX_valid_p = random.sample(GPX_valid_p, n / 2)
    for i, img in enumerate(GPX_valid_p):
        img_X1[i] = misc.imread(os.path.join('../samples0/valid/MS_record/', img))

    GPX_valid_n = random.sample(GPX_valid_n, n / 2)
    for i, img in enumerate(GPX_valid_n):
        img_X0[i] = misc.imread(os.path.join('../samples0/valid/MS_negative/', img))

    X = np.concatenate((img_X1[0:n / 2], img_X0[0:n / 2]))

    label = np.zeros((n, 2))
    label[0:n / 2, 1] = 1
    label[n / 2:n, 0] = 1

    return X, label

def read_gRoad_valid_sample():
    client = sample_client.gRoadclient()
    gRoad_valid_p = client.valid_positive()
    gRoad_valid_n = client.valid_negative()

    print 'gRoad_valid_p: %d \n' % len(gRoad_valid_p)
    print 'gRoad_valid_n: %d \n' % len(gRoad_valid_n)

    if len(gRoad_valid_p) < n / 2 or len(gRoad_valid_p) < n / 2:
        print 'n is set too large; use all the samples for testing'
        n = len(gRoad_valid_p) * 2

    img_X1, img_X0 = np.zeros((n / 2, 256, 256, 3)), np.zeros((n / 2, 256, 256, 3))
    gRoad_valid_p = random.sample(gRoad_valid_p, n / 2)
    for i, img in enumerate(gRoad_valid_p):
        img_X1[i] = misc.imread(os.path.join('../samples0/valid/MS_record/', img))

    gRoad_valid_n = random.sample(gRoad_valid_n, n / 2)
    for i, img in enumerate(gRoad_valid_n):
        img_X0[i] = misc.imread(os.path.join('../samples0/valid/MS_negative/', img))

    X = np.concatenate((img_X1[0:n / 2], img_X0[0:n / 2]))

    label = np.zeros((n, 2))
    label[0:n / 2, 1] = 1
    label[n / 2:n, 0] = 1

    return X, label
