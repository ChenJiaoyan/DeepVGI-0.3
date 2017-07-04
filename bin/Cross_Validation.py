#! /usr/bin/python

import sys

sys.path.append("../lib")

import os
import random
import shutil
import FileIO

CV_i = 0
CV_n = 4
Record_N = 160000
Negative_N = 20000


def cv(imgs, N):
    random.shuffle(imgs)
    imgs = imgs[0:N]
    l = len(imgs)
    b = l / CV_n
    valid = imgs[CV_i * b: (CV_i + 1) * b]
    train = imgs[0:CV_i * b] + imgs[(CV_i + 1) * b: l]
    return train, valid


if __name__ == '__main__':
    img_dir = '../data/image_guinea/'
    MS_imgs = os.listdir(img_dir)

    nega_file = '../data/ms_negative.csv'
    lines = FileIO.csv_reader(nega_file)
    n_imgs = []
    for line in lines:
        task_x = line['task_x']
        task_y = line['task_y']
        n_imgs.append(task_x + '-' + task_y + '.jpeg')

    MS_negative, MS_records = [], []
    for img in MS_imgs:
        if img in n_imgs:
            MS_negative.append(img)
        else:
            MS_records.append(img)

    MS_records_train, MS_records_valid = cv(MS_records, Record_N)
    MS_negative_train, MS_negative_valid = cv(MS_negative, Negative_N)

    print 'moving file for train/MS_record'
    for img in MS_records_train:
        shutil.copy(os.path.join(img_dir, img), '../samples0/train/MS_record/')

    print 'moving file for train/MS_negative'
    for img in MS_negative_train:
        shutil.copy(os.path.join(img_dir, img), '../samples0/train/MS_negative/')

    print 'moving file for valid/MS_record'
    for img in MS_records_valid:
        shutil.copy(os.path.join(img_dir, img), '../samples0/valid/MS_record/')

    print 'moving file for valid/MS_negative'
    for img in MS_negative_valid:
        shutil.copy(os.path.join(img_dir, img), '../samples0/valid/MS_negative/')
