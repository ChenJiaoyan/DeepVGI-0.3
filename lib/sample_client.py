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

    def OSM_train_positive(self):
        record = os.listdir(os.path.join(self.sample_dir, 'train/MS_record'))
        return list(set(record).intersection(set(self.OSM_positive())))

    def OSM_valid_positive(self):
        record = os.listdir(os.path.join(self.sample_dir, 'valid/MS_record'))
        return list(set(record).intersection(set(self.OSM_positive())))

class GPXclient:
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

    def GPX_positive(self):
        gpx_file = '../data/gpx_nodes.csv'
        lines = FileIO.csv_reader(gpx_file)
        p_imgs_raw = []
        for line in lines:
            task_x = line['task_x']
            task_y = line['task_y']
            img = '%s-%s.jpeg' % (task_x, task_y)
            p_imgs_raw.append(img)
        p_imgs = [list(t) for t in set(tuple(element) for element in p_imgs_raw)]
        return p_imgs

    def GPX_train_positive(self):
        record = os.listdir(os.path.join(self.sample_dir, 'train/MS_record'))
        return list(set(record).intersection(set(self.GPX_positive())))

    def GPX_valid_positive(self):
        record = os.listdir(os.path.join(self.sample_dir, 'valid/MS_record'))
        return list(set(record).intersection(set(self.GPX_positive())))

class OSM_GPXclient:
    def __init__(self):
        self.sample_dir = '../samples0/'

    def train_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'train/MS_negative'))

    def valid_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'valid/MS_negative'))

    def train_valid_positive(self):
        osm = OSMclient()
        gpx = GPXclient()
        train_raw = osm.OSM_train_positive() + gpx.GPX_train_positive()
        train_positive = [list(t) for t in set(tuple(element) for element in train_raw)]
        valid_raw = osm.OSM_valid_positive() + gpx.GPX_valid_positive()
        valid_positive = [list(t) for t in set(tuple(element) for element in valid_raw)]
        return train_positive, valid_positive

class OSM_GPX_intClient:
    def __init__(self):
        self.sample_dir = '../samples0/'

    def train_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'train/MS_negative'))

    def valid_negative(self):
        return os.listdir(os.path.join(self.sample_dir, 'valid/MS_negative'))

    def train_valid_positive(self):
        osm = OSMclient()
        gpx = GPXclient()
        osm_train_p = osm.OSM_train_positive()
        gpx_train_p = gpx.GPX_train_positive()
        train_positive = list(set(osm_train_p).intersection(set(gpx_train_p)))

        osm_valid_p = osm.OSM_valid_positive()
        gpx_valid_p = gpx.GPX_valid_positive()
        valid_positive = list(set(osm_valid_p).intersection(set(gpx_valid_p)))
        return train_positive, valid_positive