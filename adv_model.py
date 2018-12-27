#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py


import tensorflow as tf

from tensorpack.models import regularize_cost, BatchNorm
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import argscope
from tensorpack.tfutils.tower import get_current_tower_context, TowerFuncWrapper
from tensorpack.utils import logger
from tensorpack.tfutils.collection import freeze_collection

from third_party.imagenet_utils import ImageNetModel


IMAGE_SCALE = 2.0 / 255


class NoOpAttacker():
    """
    A placeholder attacker which does nothing.
    """
    def attack(self, image, label, model_func):
        return image


class PGDAttacker():
    """
    A PGD white-box attacker with random target label.
    """
    def __init__(self, num_iter, epsilon, step_size, prob_start_from_clean=0.0):
        """
        Args:
            num_iter (int):
            epsilon (float):
            step_size (int):
        """
        step_size = max(step_size, epsilon / num_iter)
        """
        TODO note
        """
        self.num_iter = num_iter
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size * IMAGE_SCALE
        self.prob_start_from_clean = prob_start_from_clean

    def _create_random_target(self, label):
        """ Only support random target for now"""
        label_offset = tf.random_uniform(tf.shape(label), minval=1, maxval=999, dtype=tf.int32)
        return tf.floormod(label + label_offset, tf.constant(1000, tf.int32))

    def attack(self, image_clean, label, model_func):
        target_label = self._create_random_target(label)

        def one_step_attack(adv):
            logits = model_func(adv)
            # Note we don't add any summaries here when creating losses, because
            # summaries don't work in conditionals
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=target_label)
            g, = tf.gradients(losses, adv)
            adv = tf.clip_by_value(adv - tf.sign(g) * self.step_size, lower_bound, upper_bound)
            return adv

        # rescale the attack epsilon and attack step size
        lower_bound = tf.clip_by_value(image_clean - self.epsilon, -1., 1.)
        upper_bound = tf.clip_by_value(image_clean + self.epsilon, -1., 1.)

        init_start = tf.random_uniform(tf.shape(image_clean), minval=-self.epsilon, maxval=self.epsilon)
        # A probability to use random initialized example or the clean example to start
        start_from_noise_index = tf.cast(tf.greater(tf.random_uniform(tf.shape(label)), self.prob_start_from_clean), tf.float32)
        start_adv = image_clean + tf.reshape(start_from_noise_index, [tf.shape(label)[0], 1, 1, 1]) * init_start

        with tf.name_scope('attack_loop'):
            adv_final = tf.while_loop(
                lambda _: True, one_step_attack,
                [start_adv],
                back_prop=False,
                maximum_iterations=self.num_iter,
                parallel_iterations=1)
        return adv_final


class AdvImageNetModel(ImageNetModel):

    label_smoothing = 0.1

    def set_attacker(self, attacker):
        self.attacker = attacker

    def build_graph(self, image, label):
        image = self.image_preprocess(image)
        assert self.data_format == 'NCHW'
        image = tf.transpose(image, [0, 3, 1, 2])
        ctx = get_current_tower_context()

        if not ctx.is_training:
            logits = self.get_logits(image)
        else:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                # BatchNorm always give you trouble
                with freeze_collection([tf.GraphKeys.UPDATE_OPS]), argscope(BatchNorm, training=False):
                    image = self.attacker.attack(image, label, self.get_logits)
                    image = tf.stop_gradient(image, name='adv_training_sample')

                logits = self.get_logits(image)

        loss = ImageNetModel.compute_loss_and_error(
            logits, label, label_smoothing=self.label_smoothing)

        wd_loss = regularize_cost(self.weight_decay_pattern,
                                  tf.contrib.layers.l2_regularizer(self.weight_decay),
                                  name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)
        total_cost = tf.add_n([loss, wd_loss], name='cost')

        if self.loss_scale != 1.:
            logger.info("Scaling the total loss by {} ...".format(self.loss_scale))
            return total_cost * self.loss_scale
        else:
            return total_cost

    def get_inference_func(self, attacker):
        """
        Returns a tower function to be used for inference. It generates adv
        images with the given attacker and runs classification on it.
        """

        def tower_func(image, label):
            assert not get_current_tower_context().is_training
            image = self.image_preprocess(image)
            image = tf.transpose(image, [0, 3, 1, 2])
            image = attacker.attack(image, label, self.get_logits)
            logits = self.get_logits(image)
            _ = ImageNetModel.compute_loss_and_error(logits, label)  # compute top-1 and top-5

        return TowerFuncWrapper(tower_func, self.get_inputs_desc())

    def image_preprocess(self, image):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            # For the purpose of adversarial training, normalize images to [-1,1]
            image = image * IMAGE_SCALE - 1.0
            return image
