#! /usr/bin/python
# get ms images without labels and bad image == negative images

import os
import sys
import csv

sys.path.append("../lib")
import FileIO

ms_file = '../data/guinea_ms.csv'
lines = FileIO.csv_reader(ms_file)
ms_imgs = []
for line in lines:
    task_x = line['task_x'].strip()
    task_y = line['task_y'].strip()
    ms_imgs.append([int(task_x), int(task_y)])
print len(ms_imgs)

img_dir = '../data/image_guinea/'
imgs = os.listdir(img_dir)
img_file = []
for img in imgs:
    i1, i2 = img.index('-'), img.index('.')
    task_x, task_y = img[0:i1], img[(i1 + 1):i2]
    img_file.append([int(task_x), int(task_y)])

ms_set = set(tuple(element) for element in ms_imgs)
img_set = set(tuple(element) for element in img_file)
n_imgs_tuple = list(img_set - ms_set)
n_imgs = list(list(element) for element in n_imgs_tuple)

output = '../data/ms_negative.csv'
fields = ['id', 'task_x', 'task_y']
csvfile = open(output, 'wb')
writer = csv.writer(csvfile)
writer.writerow(fields)
id = 0
for n_img in n_imgs:
    task_x, task_y = n_img[0], n_img[1]
    row = [id, task_x, task_y]
    writer.writerow(row)
    id += 1
csvfile.close()
