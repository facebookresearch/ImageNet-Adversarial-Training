#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serve-data.py

import argparse
import os
import multiprocessing as mp
import socket

from tensorpack.dataflow import (
    send_dataflow_zmq, MapData, TestDataSpeed, FakeData, dataset,
    AugmentImageComponent, BatchData, PrefetchDataZMQ)
from tensorpack.utils import logger
from imagenet_utils import fbresnet_augmentor

from zmq_ops import dump_arrays


def get_data(batch, augmentors):
    """
    Sec 3, Remark 4:
    Use a single random shuffling of the training data (per epoch) that is divided amongst all k workers.

    NOTE: Here we do not follow the paper, but it makes little differences.
    """
    ds = dataset.ILSVRC12(args.data, 'train', shuffle=True)
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    ds = BatchData(ds, batch, remainder=False)
    ds = PrefetchDataZMQ(ds, min(50, mp.cpu_count()))
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--fake', action='store_true')
    parser.add_argument('--batch', help='per-GPU batch size',
                        default=32, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--no-zmq-ops', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.fake:
        ds = FakeData(
            [[args.batch, 224, 224, 3], [args.batch]],
            1000, random=False, dtype=['uint8', 'int32'])
    else:
        augs = fbresnet_augmentor(True)
        ds = get_data(args.batch, augs)

    logger.info("Serving data on {}".format(socket.gethostname()))

    if args.benchmark:
        ds = MapData(ds, dump_arrays)
        TestDataSpeed(ds, warmup=300).start()
    else:
        format = None if args.no_zmq_ops else 'zmq_ops'
        send_dataflow_zmq(
            ds, 'ipc://@imagenet-train-b{}'.format(args.batch),
            hwm=150, format=format, bind=True)
