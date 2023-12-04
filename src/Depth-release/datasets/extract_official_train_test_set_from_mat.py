#!/usr/bin/env python
# -*- coding: utf-8 -*-
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#
# Helper script to convert the NYU Depth v2 dataset Matlab file into a set of
# PNG and JPEG images.
#
# See https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset

from __future__ import print_function

import os
import sys

import cv2
import h5py
import numpy as np
import scipy.io
import torch
from tqdm import tqdm


def halve(mask):
    tensor = torch.Tensor([[mask]])
    tensor = torch.nn.functional.interpolate(tensor, scale_factor=0.5, recompute_scale_factor=False)
    return np.array(tensor[0][0], dtype=mask.dtype)


def get_contours(instances, labels):
    ibb = np.zeros(np.array(instances.shape) + np.array((2, 2)), dtype=instances.dtype)
    ibb[1:-1, 1:-1] = instances
    lbb = np.zeros(np.array(labels.shape) + np.array((2, 2)), dtype=labels.dtype)
    lbb[1:-1, 1:-1] = labels
    contours = np.zeros(instances.shape, dtype=np.uint8)
    for i in range(1, contours.shape[0] + 1):
        for j in range(1, contours.shape[1] + 1):
            for ii, jj in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                if ((ibb[i, j] == 0 or (ibb[i, j] != 0 and ibb[i + ii, j + jj] != 0)) and ibb[i, j] != ibb[
                    i + ii, j + jj]) or (
                        (lbb[i, j] == 0 or (lbb[i, j] != 0 and lbb[i + ii, j + jj] != 0)) and lbb[i, j] != lbb[
                    i + ii, j + jj]):
                    contours[i - 1, j - 1] = 255
                    break
    return contours


def convert_image(i, scene, depth_raw, image, depth_dense, instances, labels):
    idx = int(i) + 1
    if idx in train_images:
        train_test = "train"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "test"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    dense_folder = folder + '/dense'
    instances_folder = folder + '/instances'
    labels_folder = folder + '/labels'
    contours_folder = folder + '/contours'
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(dense_folder):
        os.makedirs(dense_folder)
    if not os.path.exists(instances_folder):
        os.makedirs(instances_folder)
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
    if not os.path.exists(contours_folder):
        os.makedirs(contours_folder)

    img_depth = depth_raw * 1000.0
    img_depth_uint16 = img_depth.astype(np.uint16)
    dense_depth = depth_dense * 1000.0
    dense_depth_uint16 = dense_depth.astype(np.uint16)

    cv2.imwrite("%s/sync_depth_%05d.png" % (folder, i), img_depth_uint16)
    cv2.imwrite("%s/sync_depth_dense_%05d.png" % (dense_folder, i), dense_depth_uint16)

    instances_black_boundary = np.zeros((480, 640), dtype=np.uint8)
    instances_black_boundary[7:474, 7:632] = instances[7:474, 7:632]
    cv2.imwrite("%s/instances_%05d.jpg" % (instances_folder, i), instances_black_boundary)

    labels_black_boundary = np.zeros((480, 640), dtype=np.uint16)
    labels_black_boundary[7:474, 7:632] = labels[7:474, 7:632]
    cv2.imwrite("%s/labels_%05d.png" % (labels_folder, i), labels_black_boundary)

    instances0 = instances[7:7 + 467, 7:7 + 625].astype(np.uint8)
    labels0 = labels[7:7 + 467, 7:7 + 625].astype(np.uint16)
    contours0_black_boundary = np.zeros((480, 640), dtype=np.uint8)
    contours0_black_boundary[7:7 + 467, 7:7 + 625] = get_contours(instances0, labels0)
    cv2.imwrite("%s/contours0_%05d.jpg" % (contours_folder, i), contours0_black_boundary)

    instances1 = halve(instances0)
    labels1 = halve(labels0)
    contours1_black_boundary = np.zeros((240, 320), dtype=np.uint8)
    contours1_black_boundary[4:4 + 233, 4:4 + 312] = get_contours(instances1, labels1)
    cv2.imwrite("%s/contours1_%05d.jpg" % (contours_folder, i), contours1_black_boundary)

    instances2 = halve(instances1)
    labels2 = halve(labels1)
    contours2_black_boundary = np.zeros((120, 160), dtype=np.uint8)
    contours2_black_boundary[2:2 + 116, 2:2 + 156] = get_contours(instances2, labels2)
    cv2.imwrite("%s/contours2_%05d.jpg" % (contours_folder, i), contours2_black_boundary)

    instances3 = halve(instances2)
    labels3 = halve(labels2)
    contours3_black_boundary = np.zeros((60, 80), dtype=np.uint8)
    contours3_black_boundary[1:1 + 58, 1:1 + 78] = get_contours(instances3, labels3)
    cv2.imwrite("%s/contours3_%05d.jpg" % (contours_folder, i), contours3_black_boundary)

    image = image[:, :, ::-1]
    image_black_boundary = np.zeros((480, 640, 3), dtype=np.uint8)
    image_black_boundary[7:474, 7:632, :] = image[7:474, 7:632, :]
    cv2.imwrite("%s/rgb_%05d.jpg" % (folder, i), image_black_boundary)


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <h5_file> <train_test_split> <out_folder>" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    h5_file = h5py.File(sys.argv[1], "r")
    # h5py is not able to open that file. but scipy is
    train_test = scipy.io.loadmat(sys.argv[2])
    out_folder = sys.argv[3]

    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    depth_raw = h5_file['rawDepths']
    depth_dense = h5_file['depths']

    instances = h5_file['instances']
    labels = h5_file['labels']

    print("reading", sys.argv[1])

    images = h5_file['images']
    scenes = [''.join(chr(c[0]) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]

    print("processing images")
    for i, image in tqdm(enumerate(images)):
        # print("image", i + 1, "/", len(images))
        convert_image(i, scenes[i], depth_raw[i, :, :].T, image.T, depth_dense[i, :, :].T, instances[i, :, :].T,
                      labels[i, :, :].T)

    print("Finished")
