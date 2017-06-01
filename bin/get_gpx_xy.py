#! /usr/bin/python

import get_nodes
import csv
import math
from osgeo import ogr

shpfile = '../data/shp/guinea_tracknodes.shp'
output = '../data/gpx_nodes.csv'

fields = ['GPX_id', 'task_x', 'task_y']
csvfile = open(output, 'wb')
writer = csv.writer(csvfile)
writer.writerow(fields)

driver = ogr.GetDriverByName("ESRI Shapefile")
source = driver.Open(shpfile, 0)
layer = source.GetLayer()
i = 0
for feature in layer:
    geometry = feature.GetGeometryRef()
    lon = geometry.GetX()
    lat = geometry.GetY()
    task_x, task_y = get_nodes.cal_pixel(lat, lon)
    row = [i, task_x, task_y]
    writer.writerow(row)
    i += 1

csvfile.close()

