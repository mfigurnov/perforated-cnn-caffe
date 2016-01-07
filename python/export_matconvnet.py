# coding: utf-8

from __future__ import print_function
import sys
import os, errno

import numpy as np
import scipy.io

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf

ORIGINAL_MAT_FILE = sys.argv[1]
CAFFEMODEL_FILE = sys.argv[2]
OUTPUT_MAT_FILE = CAFFEMODEL_FILE + '.mat'

mat_file = scipy.io.loadmat(ORIGINAL_MAT_FILE)
mnet = mat_file['net']
layers = mnet[0, 0]['layers']

net_data = caffe_pb2.NetParameter()
caffe_data = open(CAFFEMODEL_FILE)
net_data.MergeFromString(caffe_data.read())
layers_data = {x.name: x for x in net_data.layer}

first_conv_layer = True
conv_idx = 0

for layer_idx in range(layers.shape[1]):
    mat_layer = layers[0, layer_idx]
    t = mat_layer['type'][0, 0][0]
    if t != u'conv':
        continue
    conv_idx += 1
    name = mat_layer['name'][0, 0][0]
    print(name)

    shape = mat_layer['filters'][0, 0].shape
    filters = np.array(layers_data[name].blobs[0].data, dtype='float32').reshape(shape[3], shape[2], shape[0], shape[1]).transpose()
    filters = filters.transpose((1, 0, 2, 3))  # transpose spatial dimensions
    if first_conv_layer:
        # BGR -> RGB
        filters = filters[:,:,::-1,:]
        first_conv_layer = False
    mat_layer['filters'][0, 0] = filters

    biases = np.array(layers_data[name].blobs[1].data, dtype='float32')
    mat_layer['biases'][0, 0][:] = biases

scipy.io.savemat(OUTPUT_MAT_FILE, mat_file)
