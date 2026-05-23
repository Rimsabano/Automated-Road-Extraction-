%%writefile utils.py
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import backend as K


class CustomIoU(Metric):

    def __init__(self, num_classes=2, name="custom_iou", **kwargs):
        super(CustomIoU, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes

        self.total_cm = self.add_weight(
            name="total_confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true > 0.5, tf.int32)
        y_pred = tf.cast(y_pred > 0.5, tf.int32)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )

        self.total_cm.assign_add(current_cm)

    def result(self):

        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)

        true_positives = tf.linalg.diag_part(self.total_cm)

        denominator = (
            sum_over_row +
            sum_over_col -
            true_positives
        )

        iou = tf.math.divide_no_nan(
            true_positives,
            denominator
        )

        return tf.reduce_mean(iou)

    def reset_state(self):

        K.set_value(
            self.total_cm,
            tf.zeros_like(self.total_cm)
        )


class F1Score(Metric):

    def __init__(self, name="f1_score", **kwargs):

        super(F1Score, self).__init__(
            name=name,
            **kwargs
        )

        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):

        self.precision.update_state(
            y_true,
            y_pred,
            sample_weight
        )

        self.recall.update_state(
            y_true,
            y_pred,
            sample_weight
        )

    def result(self):

        precision = self.precision.result()
        recall = self.recall.result()

        return 2 * (
            (precision * recall) /
            (precision + recall + K.epsilon())
        )

    def reset_state(self):

        self.precision.reset_state()
        self.recall.reset_state()
