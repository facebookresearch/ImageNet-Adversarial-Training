import numpy as np
import argparse
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import argscope
from tensorpack.tfutils.tower import get_current_tower_context, TowerFuncWrapper
from tensorpack.utils import logger
from tensorpack.tfutils.collection import freeze_collection
from tensorpack.utils.gpu import get_num_gpu

CLASS_NUM = 10

WEIGHT_DECAY = 1e-4

FILTER_SIZES = [64, 128, 256, 512]


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def residual_block(l, ch_out, stride=1):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    out = l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))
    return tf.nn.relu(out)


def nonlocal_dot(l):
    channel = l.shape[1]
    H, W = l.shape[2], l.shape[3]
    q, k, v = l, l, l

    if channel > H * W:
        a = tf.einsum('niab,nicd->nabcd', q, k) # N, H, W, H, W
        a = tf.einsum('nabcd,nicd->niab', a, v)
    else:
        a = tf.einsum('nihw,njhw->nij', q, k) # N, C, C
        a = tf.einsum('nij,nihw->njhw', a, v)

    a = a * (1.0 / int(H * W))
    # go through an additional 1x1 conv
    a = Conv2D('conv', a, channel, 1, strides=1, activation=get_bn(zero_init=True))
    return l + a


class ResNet_Cifar(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 32, 32, 3], 'input'),
                tf.placeholder(tf.float32, [None, CLASS_NUM], 'label')]

    def build_graph(self, image, label):
        assert tf.test.is_gpu_available()
        image = image * (2.0 / 255) - 1.0 # use simpler pre-processing
        image = tf.transpose(image, [0, 3, 1, 2])

        ctx = get_current_tower_context()
        if not ctx.is_training:
            logits = self.get_logits(image)
        else:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                with freeze_collection([tf.GraphKeys.UPDATE_OPS]), argscope(BatchNorm, training=False):
                    image = self.attacker.attack(image, label, self.get_logits)
                    image = tf.stop_gradient(image, name='adv_training_sample')

                logits = self.get_logits(image)

        ce_cost = self.compute_loss_and_error(logits, label)
        add_param_summary(('.*/W', ['histogram']))

        # weight decay on all W matrixes. including convolutional layers
        wd_cost = tf.multiply(WEIGHT_DECAY, regularize_cost('.*', tf.nn.l2_loss), name='wd_cost')

        return tf.add_n([ce_cost, wd_cost], name='cost')

    def compute_loss_and_error(self, logits, label, label_smoothing=0.):
        ce_cost = tf.losses.softmax_cross_entropy(label, logits, reduction=tf.losses.Reduction.NONE)
        ce_cost = tf.reduce_mean(ce_cost, name='cross_entropy_loss')

        single_label = tf.to_int32(tf.argmax(label, axis=1))
        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, single_label, 1)), name='wrong_vector')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'), ce_cost)
        return ce_cost

    def get_logits(self, image):
        pytorch_default_init = tf.variance_scaling_initializer(scale=1.0 / 3, mode='fan_in', distribution='uniform')
        with argscope([Conv2D, BatchNorm, GlobalAvgPooling], data_format='channels_first'), \
                argscope(Conv2D, kernel_initializer=pytorch_default_init):
            net = Conv2D('conv0', image, 64, kernel_size=3, strides=1, use_bias=False)
            net = nonlocal_dot(net)
            for i, blocks_in_module in enumerate(4*[self.module_size]):
                for j in range(blocks_in_module):
                    stride = 2 if j == 0 and i > 0 else 1
                    with tf.variable_scope("res%d.%d" % (i, j)):
                        net = residual_block(net, FILTER_SIZES[i], stride)
                if self.denoising_str.startswith('nonlocal_dot'):
                    with tf.variable_scope("res%d.denoising" % (i)):
                        net = nonlocal_dot(net)

            net = GlobalAvgPooling('gap', net)
            logits = FullyConnected('linear', net, CLASS_NUM,
                                    kernel_initializer=tf.random_normal_initializer(stddev=1e-3))
        return logits

    def get_inference_func(self, attacker):
        """
        Returns a tower function to be used for inference. It generates adv
        images with the given attacker and runs classification on it.
        """
        def tower_func(image, label):
            assert not get_current_tower_context().is_training
            image = image * (2.0 / 255) - 1.0 # use simpler pre-processing
            image = tf.transpose(image, [0, 3, 1, 2])
            image = attacker.attack(image, label, self.get_logits)
            logits = self.get_logits(image)
            _ = self.compute_loss_and_error(logits, label)

        return TowerFuncWrapper(tower_func, self.get_inputs_desc())

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test, batch):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
        ]
        ds = AugmentImageComponent(ds, augmentors)

    ds = BatchData(ds, batch, remainder=not isTrain)

    def f(dp):
        images, labels = dp
        one_hot_labels = np.eye(CLASS_NUM)[labels]  # one hot coding
        return [images, one_hot_labels]

    ds = MapData(ds, f)
    return ds


def get_config(model, dataset_train, dataset_test):
    START_LR = 0.1
    BASE_LR = START_LR * (model.batch_size / 128.0)
    callbacks=[
        ModelSaver(),
        EstimatedTimeLeft(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, min(START_LR, BASE_LR)), (100, BASE_LR * 1e-1), (150, BASE_LR * 1e-2)]),
    ]
    max_epoch = 200

    if BASE_LR > START_LR:
        callbacks.append(
            ScheduledHyperParamSetter(
                'learning_rate', [(0, START_LR), (5 * len(dataset_train), BASE_LR)],
                interp='linear', step_based=True))

    def add_eval(name, attacker, condition):
        """
        name (str): a prefix
        attacker:
        condition: a function(epoch number) that returns whether this epoch should evaluate or not
        """
        # For simple validation tasks such as image classification, distributed validation is possible.
        infs = [ClassificationError('wrong_vector', '{}-error'.format(name))]
        # infs = [ScalarStats('{}-cost'.format(name)), ClassificationError('{}-wrong_vector'.format(name))]
        nr_tower = max(get_num_gpu(), 1)
        if nr_tower == 1:
            cb = InferenceRunner(
                    dataset_test, infs,
                    tower_name=name,
                    tower_func=model.get_inference_func(attacker))
        else:
            cb = DataParallelInferenceRunner(
                    dataset_test, infs, list(range(nr_tower)),
                    tower_name=name,
                    tower_func=model.get_inference_func(attacker))

        cb = EnableCallbackIf(
            cb,
            # always eval in the last epoch no matter what
            lambda self: condition(self.epoch_num) or self.epoch_num >= max_epoch - 5)
        callbacks.append(cb)

    add_eval('eval-clean', NoOpAttacker(), lambda e: True)
    if not model.clean:
        add_eval('eval-10step', PGDAttacker(
            10, args.attack_epsilon, max(args.attack_step_size, args.attack_epsilon/10)), lambda e: True)
        add_eval('eval-50step', PGDAttacker(
            50, args.attack_epsilon, max(args.attack_step_size, args.attack_epsilon/50)), lambda e: e % 10 == 0)
        add_eval('eval-100step', PGDAttacker(
            100, args.attack_epsilon, max(args.attack_step_size, args.attack_epsilon/100)), lambda e: e % 10 == 0)
        for k in [20, 30, 40, 60, 70, 80, 90]:
            add_eval('eval-{}step'.format(k), PGDAttacker(
                k, args.attack_epsilon, max(args.attack_step_size, args.attack_epsilon/k)), lambda e: False)

    return TrainConfig(
        model=model,
        data=QueueInput(dataset_train),
        callbacks=callbacks,
        steps_per_epoch=len(dataset_train),
        max_epoch=max_epoch,
    )


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
    def __init__(self, num_iter, epsilon, step_size):
        self.num_iter = num_iter
        self.epsilon = epsilon * (2.0 / 255)
        self.step_size = step_size * (2.0 / 255)

    def attack(self, image_clean, label, model_func):
        def one_step_attack(adv):
            logits = model_func(adv)
            # Note we don't add any summaries here when creating losses, because
            # summaries don't work in conditionals
            losses = tf.losses.softmax_cross_entropy(label, logits)
            g, = tf.gradients(losses, adv)
            adv = tf.clip_by_value(adv + tf.sign(g) * self.step_size, lower_bound, upper_bound)
            return adv

        # rescale the attack epsilon and attack step size
        lower_bound = tf.clip_by_value(image_clean - self.epsilon, -1., 1.)
        upper_bound = tf.clip_by_value(image_clean + self.epsilon, -1., 1.)
        start_adv = image_clean + tf.random_uniform(tf.shape(image_clean), minval=-self.epsilon, maxval=self.epsilon)

        with tf.name_scope('attack_loop'):
            if self.num_iter > 1:
                adv_final = tf.while_loop(
                    lambda _: True, one_step_attack,
                    [start_adv],
                    back_prop=False, # TODO
                    maximum_iterations=self.num_iter,
                    parallel_iterations=1)
            else:
                adv_final = one_step_attack(start_adv)
        return adv_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.',
                        default='0,1,2,3,4,5,6,7')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--attack_iter', help='adversarial attack iteration',
                        type=int, default=10)
    parser.add_argument('--attack_epsilon', help='adversarial attack maximal perturbation',
                        type=float, default=8.0)
    parser.add_argument('--attack_step_size', help='adversarial attack step size',
                        type=float, default=2.0)
    parser.add_argument('--denoising_str', help='which denoising function to use',
                        type=str, default='None')
    parser.add_argument('--clean', help='run clean image training', action='store_true')
    parser.add_argument('--batch_size', help='batch size for training and testing',
                        type=int, default=128)
    parser.add_argument('--module_size', help='module size of each residual group',
                        type=int, default=2)
    args = parser.parse_args()

    if args.batch_size > 128:
        print("batch_size larger than 128 may produce much worse adversarial robustness!!")

    num_gpu = max(get_num_gpu(), 1)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.clean:
        log_folder = 'train_log/clean-cifar10-{}-module-{}-GPU-{}-batch-{}'.format(
                     args.denoising_str, args.module_size, num_gpu, args.batch_size)
    else:
        log_folder = 'train_log/adv-cifar10-{}-module-{}-GPU-{}-batch-{}-iter{}-epsilon{}-step{}'.format(
                      args.denoising_str, args.module_size, num_gpu, args.batch_size, args.attack_iter, args.attack_epsilon, args.attack_step_size)
    logger.set_logger_dir(os.path.join(log_folder))

    dataset_train = get_data('train', args.batch_size)
    dataset_test = get_data('test', args.batch_size)

    if args.clean:
        args.attacker = NoOpAttacker()
    else:
        args.attacker = PGDAttacker(
                args.attack_iter, args.attack_epsilon, args.attack_step_size)

    model = ResNet_Cifar()
    model.clean = args.clean
    model.batch_size = args.batch_size
    model.module_size = args.module_size
    model.attacker = args.attacker
    model.denoising_str = args.denoising_str

    config = get_config(model, dataset_train, dataset_test)
    config.session_init = None
    if num_gpu == 1:
        launch_train_with_config(config, SimpleTrainer())
    else:
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))
