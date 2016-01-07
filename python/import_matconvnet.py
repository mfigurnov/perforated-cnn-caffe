# coding: utf-8

from __future__ import print_function
import sys
import os, errno

import numpy as np
import scipy.io

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf

model = sys.argv[1]
inputFile = sys.argv[2]
outputFolder = sys.argv[3]

# Copy the weights of the model?
COPY_DATA = False

# Upgraded caffemodels are obtained by running tools/upgrade_net_proto_binary
if model == 'caffenet':
    PARAM_FILE = '/home/mfigurnov/caffe/models/bvlc_reference_caffenet/train_val.prototxt'
    CAFFEMODEL_FILE = '/home/mfigurnov/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet_upgraded.caffemodel'
elif model == 'vgg16':
    PARAM_FILE = '/home/mfigurnov/caffe/models/vgg16/vgg_train_val_upgraded.prototxt'
    CAFFEMODEL_FILE = '/home/mfigurnov/caffe/models/vgg16/VGG_ILSVRC_16_layers_upgraded.caffemodel'

MAT_FILE = inputFile
OUTPUT_DIR = outputFolder

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

mkdir_p(OUTPUT_DIR)

mat_file = scipy.io.loadmat(MAT_FILE)
mnet = mat_file['net']
layers = mnet[0, 0]['layers']

net_param = caffe_pb2.NetParameter()
caffe_param = open(PARAM_FILE)
google.protobuf.text_format.Merge(caffe_param.read(), net_param)

if COPY_DATA:
    net_data = caffe_pb2.NetParameter()
    caffe_data = open(CAFFEMODEL_FILE)
    net_data.MergeFromString(caffe_data.read())
    layers_data = {x.name: x for x in net_data.layer}

layers_param = {x.name: x for x in net_param.layer}

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

    if COPY_DATA:
        data = mat_layer['filters'][0, 0]
        data = data.transpose((1, 0, 2, 3))  # transpose spatial dimensions
        if first_conv_layer:
            # RGB -> BGR
            data = data[:,:,::-1,:]
            first_conv_layer = False
        data = data.transpose()  # F-order -> C-order
        data = data.ravel()
        assert len(layers_data[name].blobs[0].data) == data.shape[0]
        del layers_data[name].blobs[0].data[:]
        layers_data[name].blobs[0].data.extend(data.tolist())

        data = mat_layer['biases'][0, 0].ravel().astype(float)
        assert len(layers_data[name].blobs[1].data) == data.shape[0]
        del layers_data[name].blobs[1].data[:]
        layers_data[name].blobs[1].data.extend(data.tolist())

    try:
        non_perforated_indices = mat_layer['nonPerforatedIndices'][0, 0]
        interpolation_indices = mat_layer['interpolationIndicesOut'][0, 0]
        perforation_rate = mat_layer['rate'][0, 0][0, 0]
        print('Layer is perforated')
        perforated = True
    except ValueError as e:
        print('Layer is not perforated')
        perforated = False

    if perforated:
        layers_param[name].type = 'ConvolutionPerforated'

        perf_param = layers_param[name].perf_param

        # The non-perforated positions indices and interpolation indices
        # must be transposed, because Caffe and MatConvNet use different orderings of spatial
        # dimensions.
        non_perforated_indices = non_perforated_indices.ravel()

        perf_mask = -1 * np.ones(np.prod(interpolation_indices.shape), dtype=int)
        perf_mask[non_perforated_indices] = np.arange(np.prod(interpolation_indices.shape))
        perf_mask = perf_mask.reshape(interpolation_indices.shape, order='F')
        non_perforated_indices_transposed = np.flatnonzero(perf_mask != -1)

        indices_mapping = perf_mask.ravel()[non_perforated_indices_transposed]
        indices_rev_mapping = np.zeros_like(indices_mapping)
        indices_rev_mapping[indices_mapping] = np.arange(indices_mapping.shape[0])
        interpolation_indices_transposed = indices_rev_mapping[interpolation_indices]

        non_perforated_indices_transposed = non_perforated_indices_transposed.tolist()
        perf_param.non_perforated_indices.extend(non_perforated_indices_transposed)

        interpolation_indices_transposed = interpolation_indices_transposed.ravel().tolist()
        perf_param.interpolation_indices.extend(interpolation_indices_transposed)

        micro_batch_size = int(1.0 / perforation_rate)
        perf_param.micro_batch_size = micro_batch_size

with open(OUTPUT_DIR + 'train_val.prototxt', 'w') as f:
    print(net_param, file=f)

if COPY_DATA:
    with open(OUTPUT_DIR + 'model.caffemodel', 'w') as f:
        f.write(net_data.SerializeToString())

