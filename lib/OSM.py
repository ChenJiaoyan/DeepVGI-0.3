#! /usr/bin/python

import math
import FileIO
import os
import random


def cal_lat_lon(task_x, task_y):
    task_z = 18
    PixelX = task_x * 256
    PixelY = task_y * 256
    MapSize = 256 * math.pow(2, task_z)
    x = (PixelX / MapSize) - 0.5
    y = 0.5 - (PixelY / MapSize)
    lon_left = 360 * x
    lat_top = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi

    PixelX = (task_x + 1) * 256
    PixelY = (task_y + 1) * 256
    MapSize = 256 * math.pow(2, task_z)
    x = (PixelX / MapSize) - 0.5
    y = 0.5 - (PixelY / MapSize)
    lon_right = 360 * x
    lat_bottom = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi

    return lat_top, lon_left, lat_bottom, lon_right


def cal_pixel(lat, lon):
    task_z = 18
    sin_lat = math.sin(lat * math.pi / 180)
    x = ((lon + 180) / 360) * 256 * math.pow(2, task_z)
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * 256 * math.pow(2, task_z)
    task_x = int(math.floor(x / 256))
    task_y = int(math.floor(y / 256))
    pixel_x = int(x % 256 + 0.5)
    pixel_y = int(y % 256 + 0.5)
    return task_x, task_y, pixel_x, pixel_y


class MSClient:
    def __init__(self, project_id=5, name='Guinea'):
        self.project_id = project_id
        self.name = name

    def read_p_images(self):
        osm_file = '../data/guinea_highway.csv'
        lines = FileIO.csv_reader(osm_file)
        p_imgs_raw = []
        for line in lines:
            task_x = line['task_x']
            task_y = line['task_y']
            p_imgs_raw.append([int(task_x), int(task_y)])
        p_imgs = [list(t) for t in set(tuple(element) for element in p_imgs_raw)]  # remove duplicate
        return p_imgs

    def read_n_images(self):
        img_dir = '../data/image_guinea/'
        imgs = os.listdir(img_dir)
        img_file = []
        for img in imgs:
            i1, i2 = img.index('-'), img.index('.')
            task_x, task_y = img[0:i1], img[(i1 + 1):i2]
            img_file.append([int(task_x), int(task_y)])

        p_imgs = self.read_p_images()
        p_set = set(tuple(element) for element in p_imgs)
        img_set = set(tuple(element) for element in img_file)
        n_imgs_tuple = list(img_set - p_set)
        n_imgs = list(list(element) for element in n_imgs_tuple)
        return n_imgs

    def imgs_cross_validation(self, cv_i, cv_n):
        img_dir = '../data/image_guinea/'
        imgs = os.listdir(img_dir)
        random.shuffle(imgs)
        l = len(imgs)
        batch = l / cv_n
        test_imgs = imgs[cv_i * batch: (cv_i + 1) * batch]
        train_imgs = imgs[0:cv_i * batch] + imgs[(cv_i + 1) * batch: l]
        return train_imgs, test_imgs

class GPXclient:
    def __init__(self, project_id=5, name='Guinea'):
        self.project_id = project_id
        self.name = name

    def read_p_images(self):
        gpx_file = '../data/gpx_nodes.csv'
        lines = FileIO.csv_reader(gpx_file)
        p_imgs_raw = []
        for line in lines:
            task_x = line['task_x']
            task_y = line['task_y']
            p_imgs_raw.append([task_x, task_y])
        p_imgs = [list(t) for t in set(tuple(element) for element in p_imgs_raw)]
        return p_imgs

    def read_n_images(self):
        img_dir = '../data/image_guinea'
        imgs = os.listdir(img_dir)
        img_file = []
        for img in imgs:
            i1, i2 = img.index('-'), img.index('.')
            task_x, task_y = int(img[0:i1]), int(img[(i1 + 1):i2])
            img_file.append([int(task_x), int(task_y)])

        p_imgs = self.read_p_images()
        p_set = set(tuple(element) for element in p_imgs)
        img_set = set(tuple(element) for element in img_file)
        n_imgs_tuple = list(img_set - p_set)
        n_imgs = list(list(element) for element in n_imgs_tuple)
        return n_imgs

    def imgs_cross_validation(self, cv_i, cv_n):
        img_dir = '../data/image_guinea/'
        imgs = os.listdir(img_dir)
        random.shuffle(imgs)
        l = len(imgs)
        batch = l / cv_n
        test_imgs = imgs[cv_i * batch: (cv_i + 1) * batch]
        train_imgs = imgs[0:cv_i * batch] + imgs[(cv_i + 1) * batch: l]
        return train_imgs, test_imgs

class OSM_GPXclient:
    def __init__(self, project_id=5, name='Guinea'):
        self.project_id = project_id
        self.name = name

    def read_pn_images(self):
        osm = MSClient()
        gpx = GPXclient()
        p_imgs = osm.read_p_images() + gpx.read_p_images()
        n_imgs = osm.read_n_images() + gpx.read_n_images()
        return p_imgs, n_imgs

    def imgs_cross_validation(self, cv_i, cv_n):
        img_dir = '../data/image_guinea/'
        imgs = os.listdir(img_dir)
        random.shuffle(imgs)
        l = len(imgs)
        batch = l / cv_n
        test_imgs = imgs[cv_i * batch: (cv_i + 1) * batch]
        train_imgs = imgs[0:cv_i * batch] + imgs[(cv_i + 1) * batch: l]
        return train_imgs, test_imgs


