#! /usr/bin/python

import FileIO
import os

class OSMclient:
    def __init__(self):
        self.sample_dir = '../samples0/'

    def MS_train_record(self):
        return os.listdir(os.path.join(self.sample_dir, 'train/MS_record'))

    def MS_train_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'train/MS_negative'))

    def MS_valid_record(self):
        return os.listdir(os.path.join(self.sample_dir, 'valid/MS_record'))

    def MS_valid_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'valid/MS_negative'))

    def OSM_positive(self):
        osm_file = '../data/guinea_highway.csv'
        lines = FileIO.csv_reader(osm_file)
        p_imgs_raw = []
        for line in lines:
            task_x = line['task_x']
            task_y = line['task_y']
            img = '%s-%s.jpeg' % (task_x, task_y)
            p_imgs_raw.append(img)
        p_imgs = [list(t) for t in set(tuple(element) for element in p_imgs_raw)]
        return p_imgs

    def MS_train_positive(self):
        record = os.listdir(os.path.join(self.sample_dir, 'train/MS_record'))
        return list(set(record).intersection(set(self.OSM_positive())))

    def MS_valid_positive(self):
        record = os.listdir(os.path.join(self.sample_dir, 'valid/MS_record'))
        return list(set(record).intersection(set(self.OSM_positive())))
