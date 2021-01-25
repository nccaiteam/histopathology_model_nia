from metrics import FixedDropout
from keras.models import load_model
from PIL import Image
import config as c
import os

def save_img(img_arr, img_path):
    """ Saves a numpy image array as a image file.

        Args:
            img_arr: an numpy image array.
            img_path: the file path to store the image array.
    """
    img = Image.fromarray(img_arr)
    img.save(img_path)

def get_model(name, custom_objects=c.CUSTOM_OBJECTS, pretrained_weights=None, show_summary=False):
    """ Loads the model and the weights.

        Args:
            name: the name of the Keras model file.
            custom_objects: a dictionary that contains custom metrics used in the model.
        
        Returns:
            model: the Keras model.
    """
    model_name = os.path.join(c.MODEL_PATH, name + c.MODEL_FORMAT)
    
    if custom_objects:
        if 'EfficientNet' in name:
            custom_objects['FixedDropout'] = FixedDropout
        model = load_model(model_name, custom_objects=custom_objects)
    else:
        model = load_model(model_name)
    
    if pretrained_weights:
        weight_name = os.path.join(c.WEIGHT_PATH, name + c.WEIGHT_FORMAT)
        model.load_weights(weight_name)
        
    if show_summary:
        model.summary()
        
    return model