""" Implementation of Grad-CAM.
    [RR Selvaraju et al., 2016. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization]
    https://arxiv.org/abs/1610.02391
"""
from keras.models import Model
import tensorflow as tf
import numpy as np
import keras.backend as K
import cv2
import config as c

def get_last_conv_layer(model):
    """ Retreives the last convolutional layer to use for creating Grad-CAM.
        Grad-CAM Gradient class activation maps are a visualization technique for deep learning networks.

        Args:
            model: a Keras model that computes Grad-CAM.

        Returns:
            layer: the last convolutional layer to use for creating Grad-CAM.
    """
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer
    raise Exception('- There is no convolutional layer for Grad-CAM to use!')

def get_grad_cam(img_arr, model):
    """ Gets a Grad-CAM of an input image.

        Args:
            model: a Keras model that computes Grad-CAM.
            img_arr: an input image array which the model creates a Grad-CAM of.

        Returns:
            grad_cam: an numpy array of Grad-CAM.
            pred_num: the sigmoid prediction value of the input image.
            pred_label: the predicted label of the input image after thresholding (threshold = 0.5).
    """
    conv_layer = get_last_conv_layer(model)
    grad_model = Model([model.input], [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        img_arr_float = tf.cast(img_arr, 'float32')
        conv_outputs, predictions = grad_model(img_arr_float)
        pred = predictions[:, 0]
        pred_num = K.eval(pred)[0]
    
    if pred_num < c.THRESHOLD:
        pred_label = 0
    else:
        pred_label = 1

    grads = tape.gradient(pred, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    gate_f = tf.cast(conv_outputs > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    grad_cam = np.zeros(dtype=np.float32, shape=(conv_outputs.shape[:2]))
    for i, w in enumerate(weights):
        grad_cam += w * conv_outputs[:, :, i]

    grad_cam = cv2.resize(grad_cam.numpy(), (img_arr.shape[2], img_arr.shape[1]))
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)

    return grad_cam, pred_num, pred_label

def get_heatmap_overlay(img_arr, model):
    """ Overlays the Grad-CAM on the input image.

        Args:
            model: a Keras model that computes Grad-CAM.
            img_arr: an input image array which the model creates a Grad-CAM of.

        Returns:
            heatmap_overlay: an numpy array of the input image with the Grad-CAM heatmap overlaid.
            pred_num: a sigmoid prediction value of the input image.
            pred_label: a predicted label of the input image after thresholding (threshold = 0.5).
    """
    img_arr_modi = img_arr / 255.
    img_arr_modi = np.expand_dims(img_arr_modi, axis=0)

    grad_cam, pred_num, pred_label = get_grad_cam(img_arr_modi, model)

    heatmap = np.uint8(255 * np.clip(grad_cam, 0, 1))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap[np.where(grad_cam == 0)] = 0

    heatmap_gray = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(heatmap_gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contoured_image = cv2.drawContours(img_arr, contours, -1, (255, 255, 255), 4)
    heatmap_overlay = cv2.addWeighted(contoured_image, 0.9, heatmap, 0.5, 0)

    return heatmap_overlay, pred_num, pred_label