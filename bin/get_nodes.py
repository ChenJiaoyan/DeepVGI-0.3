#! /usr/bin/python

import urllib2
import xml.etree.ElementTree as ET
import csv
import math
from osgeo import ogr

def cal_pixel(lat, lon):
    task_z = 18
    sin_lat = math.sin(lat * math.pi / 180)
    x = ((lon + 180) / 360) * 256 * math.pow(2, task_z)
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * 256 * math.pow(2, task_z)
    task_x = int(math.floor(x / 256))
    task_y = int(math.floor(y / 256))
    return task_x, task_y

def get_nodes(shpfile):
    api = 'http://www.openstreetmap.org/api/0.6/way/'
    driver = ogr.GetDriverByName("ESRI Shapefile")
    source = driver.Open(shpfile, 0)
    layer = source.GetLayer()
    idall = []
    latall = []
    lonall = []

    for i in range(0, layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        osmid = feature.GetField("osm_id")
        if osmid != None:
            url = api + str(osmid) + '/full'
            req = urllib2.Request(url)
            try:
                response = urllib2.urlopen(req)
            except urllib2.HTTPError, e:
                print e.code
            else:
                tree = ET.parse(response)
                root = tree.getroot()
                for node in root.findall('node'):
                    idall.append(node.get('id'))
                    latall.append(node.get('lat'))
                    lonall.append(node.get('lon'))

    return idall, latall, lonall


if __name__ == "__main__":

    shapefile = '../data/shp/select4.shp'
    output = '../data/select4.csv'
    fields = ['osm_id', 'task_x', 'task_y']
    csvfile = open(output, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(fields)

    idall, latall, lonall = get_nodes(shapefile)
    for id, lat, lon in zip(idall, latall, lonall):
        task_x, task_y = cal_pixel(float(lat), float(lon))
        row = [id, task_x, task_y]
        writer.writerow(row)
    csvfile.close()
