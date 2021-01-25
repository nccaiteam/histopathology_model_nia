import os
import networks
import pandas as pd
import config as c
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint, TensorBoard

def k_fold_train(name, k_fold, save_name, model=None,
           batch_size=c.BATCH_SIZE, epochs=c.EPOCHS,
           optimizer=c.OPTIMIZER, loss=c.LOSS, metrics=c.METRICS,
           show_summary=False, regularization=False,
           train_all=False):
    """ Trains k-fold models.

        Args:
            name: the name of the model.
            k_fold: nth k-fold.
            model: a chosen model to train.
            batch_size: batch size for training.
            epochs: number of epochs for training.
            train_all: it decides whether it will iterate through every k-fold model for training or only use a nth k-fold model for training. 
    """
    dirs = [c.MODEL_PATH, c.WEIGHT_PATH, c.CSVLOG_PATH, c.TENSORBOARD_PATH]
    for dir in dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)

    if k_fold == c.K_FOLD and train_all:
        start_k = 1
        end_k = k_fold + 1
    else:
        start_k = k_fold
        end_k = k_fold + 1

    for i in range(start_k, end_k):
        print('- Training K-fold = %d'%i)
        
        if not model:
            backbone = networks.get_backbone(name)
            model = networks.modify_network(backbone, optimizer=optimizer, loss=loss, metrics=metrics,
                                        show_summary=show_summary, regularization=regularization)
        else:
            if show_summary:
                model.summary()

        train_datagen = ImageDataGenerator(rescale=c.RESCALE,
                                           rotation_range=c.ROTATION_RANGE,
                                           horizontal_flip=c.HORIZONTAL_FLIP,
                                           vertical_flip=c.VERTICAL_FLIP,
                                           channel_shift_range=c.CHANNEL_SHIFT_RANGE,
                                           brightness_range=c.BRIGHTNESS_RANGE,
                                           fill_mode=c.FILL_MODE)
                      
        val_datagen = ImageDataGenerator(rescale=c.RESCALE)

        train_df = pd.read_csv(c.TRAIN_PATH + str(i) + c.CSV_FORMAT, dtype=str)
        print(train_df.info())
        val_df = pd.read_csv(c.VAL_PATH + str(i) + c.CSV_FORMAT, dtype=str)
        print(val_df.info())
        train_generator = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                            directory=c.ORIGINAL_DATA_PATH,
                                                            batch_size=batch_size,
                                                            x_col='id',
                                                            y_col='label',
                                                            shuffle=c.SHUFFLE,
                                                            class_mode=c.CLASS_MODE,
                                                            target_size=c.TARGET_SIZE,
                                                            color_mode=c.COLOR_MODE)

        validation_generator = val_datagen.flow_from_dataframe(dataframe=val_df,
                                                            directory=c.ORIGINAL_DATA_PATH,
                                                            batch_size=batch_size,
                                                            x_col='id',
                                                            y_col='label',
                                                            shuffle=c.SHUFFLE,
                                                            class_mode=c.CLASS_MODE,
                                                            target_size=c.TARGET_SIZE,
                                                            color_mode=c.COLOR_MODE)
        new_name = save_name + '_' + str(i)
        _callbacks = get_callbacks(new_name)
        model_path = os.path.join(c.MODEL_PATH, new_name + c.MODEL_FORMAT)
        try:
            model.fit_generator(train_generator,
                                steps_per_epoch=train_generator.n//batch_size,
                                validation_data=validation_generator,
                                validation_steps=validation_generator.n//batch_size,
                                epochs=epochs,
                                verbose=c.VERBOSE,
                                max_queue_size=c.MAX_QUEUE_SIZE,
                                workers=c.WORKERS,
                                class_weight=c.CLASS_WEIGHT,
                                callbacks=_callbacks)
            model.save(model_path)
        except KeyboardInterrupt:
            model.save(model_path)

def get_callbacks(name):
    """ Training callbacks.

        Args:
            name: the name of callbacks files which matches to its corresponding model name.
        
        Returns:
            a list of callbacks to use in training.
    """
    checkpoint = ModelCheckpoint(os.path.join(c.WEIGHT_PATH, '%s_{epoch:02d}'%name + c.WEIGHT_FORMAT),
                                 monitor=c.MONITOR_MC,
                                 verbose=c.VERBOSE_MC,
                                 save_best_only=c.SAVE_BEST_ONLY_MC,
                                 save_weights_only=c.SAVE_WEIGHTS_ONLY_MC,
                                 mode=c.MODE_MC,
                                 period=c.PERIOD)

    lr_reducer = ReduceLROnPlateau(monitor=c.MONITOR_LR,
                                   factor=c.FACTOR_LR,
                                   patience=c.PATIENCE_LR,
                                   min_lr=c.MIN_LR_LR,
                                   mode=c.MODE_LR)

    csv_logger = CSVLogger(os.path.join(c.CSVLOG_PATH, name + c.CSV_FORMAT))

    tensorboard = TensorBoard(log_dir=os.path.join(c.TENSORBOARD_PATH, name),
                              histogram_freq=c.HISTOGRAM_FREQ)

    return [checkpoint, lr_reducer, csv_logger, tensorboard]
