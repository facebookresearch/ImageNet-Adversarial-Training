import numpy as np
import horovod.tensorflow as hvd
import tensorflow as tf

from tensorpack.callbacks import Inferencer
from tensorpack.utils.stats import RatioCounter


class HorovodClassificationError(Inferencer):
    """
    Like ClassificationError, it evaluates total samples & count of incorrect or correct samples.
    But in the end we aggregate the total&count by horovod.
    """
    def __init__(self, wrong_tensor_name, summary_name='validation_error'):
        """
        Args:
            wrong_tensor_name(str): name of the ``wrong`` binary vector tensor.
            summary_name(str): the name to log the error with.
        """
        self.wrong_tensor_name = wrong_tensor_name
        self.summary_name = summary_name

    def _setup_graph(self):
        self._placeholder = tf.placeholder(tf.float32, shape=[2], name='to_be_reduced')
        self._reduced = hvd.allreduce(self._placeholder, average=False)

    def _before_inference(self):
        self.err_stat = RatioCounter()

    def _get_fetches(self):
        return [self.wrong_tensor_name]

    def _on_fetches(self, outputs):
        vec = outputs[0]
        batch_size = len(vec)
        wrong = np.sum(vec)
        self.err_stat.feed(wrong, batch_size)
        # Uncomment this to monitor the metric during evaluation
        # print(self.summary_name, self.err_stat.ratio)

    def _after_inference(self):
        tot = self.err_stat.total
        cnt = self.err_stat.count
        tot, cnt = self._reduced.eval(feed_dict={self._placeholder: [tot, cnt]})
        return {self.summary_name: cnt * 1. / tot}
