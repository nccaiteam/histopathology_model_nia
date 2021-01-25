import keras.backend as K
from keras.layers import *
import tensorflow as tf
import config as c

class FixedDropout(tf.keras.layers.Dropout):
    """ Custom dropout to fix the problem of 'Unknown layer: FixedDropout' when loading EfficientNet models.
        
        Issue:
            https://github.com/tensorflow/tensorflow/issues/30946
    """
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

def true_positives(y_true, y_pred):
    """ Computes the number of true positives in a given batch.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            The number of true positive values.
    """
    return  K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

def true_negatives(y_true, y_pred):
    """ Computes the number of true negatives in a given batch.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            The number of true negative values.
    """
    return K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))

def false_positives(y_true, y_pred):
    """ Computes the number of false positives in a given batch.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            The number of false positive values.
    """
    return K.sum(K.round(K.clip((1-y_true) * y_pred, 0, 1)))

def false_negatives(y_true, y_pred):
    """ Computes the number of false negatives in a given batch.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            The number of false negative values.
    """
    return K.sum(K.round(K.clip(y_true * (1-y_pred), 0, 1)))

def npv(y_true, y_pred):
    """ Computes a negative predictive value of the predictions with respect to the labels.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            Negative predictive value.
    """
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    predicted_negatives = K.sum(K.round(K.clip(1 - y_pred, 0, 1)))
    return true_negatives / (predicted_negatives + K.epsilon())

def precision(y_true, y_pred):
    """ Computes a precision value of the predictions with respect to the labels.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            Precision.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def specificity(y_true, y_pred):
    """ Computes a specificity value of the predictions with respect to the labels.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            Specificity.
    """
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def recall(y_true, y_pred):
    """ Computes a recall value (sensitivity value) of the predictions with respect to the labels.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            Recall.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f_beta(y_true, y_pred):
    """ Computes a f-beta score of the predictions with respect to the labels.
        The default beta value here is 1.

        Args:
            y_true: the ground truth values.
            y_pred: the predicted values.

        Returns:
            F-beta score.
    """
    beta = 1
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    numerator = (1 + beta ** 2) * (precision_value * recall_value)
    denominator = ((beta ** 2) * precision_value) + recall_value + K.epsilon()
    return numerator / denominator
