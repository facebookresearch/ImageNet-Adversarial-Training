#-*- coding: utf-8 -*-
#File:

from adv_model import AdvImageNetModel
from resnet_model import (
    resnet_group, resnet_bottleneck, resnet_backbone, resnet_denoising_backbone)


class ResNetModel(AdvImageNetModel):
    def __init__(self, depth):
        self.num_blocks = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]

    def get_logits(self, image):
        return resnet_backbone(image, self.num_blocks, resnet_group, resnet_bottleneck)


class ResNetDenoiselModel(AdvImageNetModel):
    def __init__(self, depth, denoise_func_str):
        self.num_blocks = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        self.denoise_func_str = denoise_func_str

    def get_logits(self, image):
        return resnet_denoising_backbone(image, self.num_blocks, resnet_group, resnet_bottleneck, self.denoise_func_str)
