import keras.backend as K
import tensorflow as tf
import sys
import config as c
from keras.optimizers import *


def main(k_fold, condition, gpu_i, _all=False):

    K.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[gpu_i], 'GPU')
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        ################## for organizing dataset ##################
        # import dataset
        # dataset.get_k_folds_dataset(shuffle=False)
        # dataset.debug_dataset()
        ############################################################

        ################### for k-fold training ####################
        if condition == 'train':
            import train
            train.k_fold_train(name=c.DEFAULT_NAME, save_name=c.DEFAULT_NAME,
                            k_fold=k_fold, regularization=True, show_summary=True,
                            train_all=_all)
        elif condition == 'test':
            import test
            test.k_fold_test(name=c.DEFAULT_NAME,
                            k_fold=k_fold,
                            test_all=_all)
        else:
            print('- Wrong arguments!')
        ############################################################

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 6 and sys.argv[-1] == 'all':
        k_fold = int(sys.argv[2])
        main(k_fold, condition=str(sys.argv[-2]), gpu_i=int(sys.argv[-3]), _all=True)
    else:
        k_fold = int(sys.argv[2])
        main(k_fold, condition=str(sys.argv[-1]), gpu_i=int(sys.argv[-2]))
