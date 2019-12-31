#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import cv2
import tensorflow as tf

from tensorpack import TowerContext
from tensorpack.tfutils import get_model_loader
from tensorpack.dataflow.dataset import ILSVRCMeta

import nets

"""
A small inference example for attackers to play with.
"""


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--depth', help='ResNet depth',
                    type=int, default=152, choices=[50, 101, 152])
parser.add_argument('--arch', help='Name of architectures defined in nets.py',
                    default='ResNetDenoise')
parser.add_argument('--load', help='path to checkpoint')
parser.add_argument('--input', help='path to input image')
args = parser.parse_args()

model = getattr(nets, args.arch + 'Model')(args)

input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
image = input / 127.5 - 1.0
image = tf.transpose(image, [0, 3, 1, 2])
with TowerContext('', is_training=False):
    logits = model.get_logits(image)

sess = tf.Session()
get_model_loader(args.load).init(sess)

sample = cv2.imread(args.input)  # this is a BGR image, not RGB
# imagenet evaluation uses standard imagenet pre-processing
# (resize shortest edge to 256 + center crop 224).
# However, for images of unknown sources, let's just do a naive resize.
sample = cv2.resize(sample, (224, 224))

prob = sess.run(logits, feed_dict={input: np.array([sample])})
print("Prediction: ", prob.argmax())

synset = ILSVRCMeta().get_synset_words_1000()
print("Top 5: ", [synset[k] for k in prob[0].argsort()[-5:][::-1]])
