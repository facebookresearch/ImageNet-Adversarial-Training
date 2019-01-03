#-*- coding: utf-8 -*-
#File:

from adv_model import AdvImageNetModel
from resnet_model import (
    resnet_group, resnet_bottleneck, resnet_backbone, resnet_denoising_backbone)
from resnet_model import denoising


class ResNetModel(AdvImageNetModel):
    def __init__(self, args):
        self.num_blocks = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[args.depth]

    def get_logits(self, image):
        return resnet_backbone(image, self.num_blocks, resnet_group, resnet_bottleneck)


class ResNetDenoiseModel(AdvImageNetModel):
    def __init__(self, args):
        self.num_blocks = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[args.depth]

    def get_logits(self, image):
        return resnet_denoising_backbone(
            image, self.num_blocks, resnet_group, resnet_bottleneck, denoising)
