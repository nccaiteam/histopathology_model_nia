import config as c
from keras.applications import *
import efficientnet.keras as efn
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
def modify_network(backbone, optimizer=c.OPTIMIZER, loss=c.LOSS, metrics=c.METRICS, 
                show_summary=False, regularization=False):
    """ Modifies the backbone/ base network.

        Args:
            backbone: the Keras model of the base network.
        
        Returns:
            model: the modified Keras model.
    """
    unpacked_backbone = Model(inputs=backbone.input, outputs=backbone.output)
    backbone_out = unpacked_backbone.output
    layer = Dense(512, activation='relu')(backbone_out)
    out = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=unpacked_backbone.input, outputs=out)
    if regularization:
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                setattr(layer, 'kernel_regularizer', l2)
    
    if show_summary:
        model.summary()
        
        print('- Checking kernal regularizers ...')
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                print(layer.name, getattr(layer, 'kernel_regularizer'))
    
    model.compile(optimizer=c.OPTIMIZER,
                  loss=c.LOSS,
                  metrics=c.METRICS)
    return model

def get_backbone(name):
    """ Chooses a backbone/ base network.

        Args:
            name: the name of the base network.

        Returns:
            backbone: the Keras model of the chosen network.
    """
    if name == 'EfficientNetB0':
        backbone = efn.EfficientNetB0(include_top=c.INCLUDE_TOP,
                                    weights=c.WEIGHTS,
                                    input_shape=c.INPUT_SHAPE,
                                    pooling=c.POOLING)
    elif name == 'EfficientNetB1':
        backbone = efn.EfficientNetB1(include_top=c.INCLUDE_TOP,
                                    weights=c.WEIGHTS,
                                    input_shape=c.INPUT_SHAPE,
                                    pooling=c.POOLING)
    elif name == 'EfficientNetB2':
        backbone = efn.EfficientNetB2(include_top=c.INCLUDE_TOP,
                                    weights=c.WEIGHTS,
                                    input_shape=c.INPUT_SHAPE,
                                    pooling=c.POOLING)
    elif name == 'EfficientNetB3':
        backbone = efn.EfficientNetB3(include_top=c.INCLUDE_TOP,
                                    weights=c.WEIGHTS,
                                    input_shape=c.INPUT_SHAPE,
                                    pooling=c.POOLING)
    elif name == 'EfficientNetB4':
        backbone = efn.EfficientNetB4(include_top=c.INCLUDE_TOP,
                                    weights=c.WEIGHTS,
                                    input_shape=c.INPUT_SHAPE,
                                    pooling=c.POOLING)
    elif name == 'EfficientNetB5':
        backbone = efn.EfficientNetB5(include_top=c.INCLUDE_TOP,
                                    weights=c.WEIGHTS,
                                    input_shape=c.INPUT_SHAPE,
                                    pooling=c.POOLING)
    elif name == 'EfficientNetB6':
        backbone = efn.EfficientNetB6(include_top=c.INCLUDE_TOP,
                                    weights=c.WEIGHTS,
                                    input_shape=c.INPUT_SHAPE,
                                    pooling=c.POOLING)
    elif name == 'EfficientNetB7':
        backbone = efn.EfficientNetB7(include_top=c.INCLUDE_TOP,
                                    weights=c.WEIGHTS,
                                    input_shape=c.INPUT_SHAPE,
                                    pooling=c.POOLING)
    elif name == 'VGG16':
        backbone = VGG16(weights=c.WEIGHTS,
                    include_top=c.INCLUDE_TOP,
                    input_shape=c.INPUT_SHAPE,
                    pooling=c.POOLING)
    elif name == 'ResNet50':
        backbone = ResNet50(include_top=c.INCLUDE_TOP,
                        weights=c.WEIGHTS,
                        input_shape=c.INPUT_SHAPE,
                        pooling=c.POOLING)
    elif name == 'InceptionV3':
        backbone = InceptionV3(include_top=c.INCLUDE_TOP,
                            weights=c.WEIGHTS,
                            input_shape=c.INPUT_SHAPE,
                            pooling=c.POOLING)
    elif name == 'DenseNet201':
        backbone = DenseNet201(weights=c.WEIGHTS,
                            include_top=c.INCLUDE_TOP,
                            input_shape=c.INPUT_SHAPE,
                            pooling=c.POOLING)
    else:
        backbone = None
    try:
        backbone.trainable = True
        return backbone
    except Exception as e:
        print(str(e))