from keras.optimizers import *
from metrics import *
import os

K_FOLD = 10
DEFAULT_NAME = 'EfficientNetB0'
BASE_PATH = os.pardir

ORIGINAL_DATA_PATH = os.path.join(BASE_PATH, 'original_data')
DATA_CSV_PATH = os.path.join(BASE_PATH, 'nia-2020_midterm_data.csv')
DATA_PATH = os.path.join(BASE_PATH, 'data')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VAL_PATH = os.path.join(DATA_PATH, 'validation')
TEST_PATH = os.path.join(BASE_PATH, 'test')
TEST_SET_CSV_PATH = os.path.join(DATA_PATH, 'test.csv')

OLD_MODEL_PATH = os.path.join(BASE_PATH, 'models')
OLD_WEIGHT_PATH = os.path.join(BASE_PATH, 'weights')

MODEL_PATH = os.path.join(BASE_PATH, 'models')
WEIGHT_PATH = os.path.join(BASE_PATH, 'weights')

CSVLOG_PATH = os.path.join(BASE_PATH, 'csvlogs')
RESULT_PATH = os.path.join(BASE_PATH, 'results')
HEATMAP_PATH = os.path.join(BASE_PATH, 'heatmaps')
TENSORBOARD_PATH = os.path.join(BASE_PATH, 'tensorboards')

TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2

OPTIMIZER = Adam()
LOSS = 'binary_crossentropy'

METRICS = ['acc', 'AUC', recall, specificity, f_beta, precision, npv, true_positives, true_negatives, false_positives, false_negatives]
CUSTOM_OBJECTS = {'recall': recall, 'specificity': specificity, 'f_beta': f_beta, 'precision': precision, 'npv':npv, 'true_positives':true_positives, 'true_negatives':true_negatives, 'false_positives':false_positives, 'false_negatives':false_negatives}
				
HEIGHT = 512
WIDTH = 512
INPUT_SHAPE = (HEIGHT, WIDTH, 3)
WEIGHTS = 'imagenet'
INCLUDE_TOP = False
POOLING = 'avg'

BATCH_SIZE = 8
RESCALE = 1./255
EPOCHS = 200
ROTATION_RANGE = 20
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
FILL_MODE = 'nearest'
CHANNEL_SHIFT_RANGE = 100.0
BRIGHTNESS_RANGE = [0.8, 1.2]

TARGET_SIZE = (512, 512)
COLOR_MODE = 'rgb'
SHUFFLE = True
CLASS_MODE = 'binary'

MAX_QUEUE_SIZE = 8
WORKERS = 5
VERBOSE = 1

TEST_SHUFFLE = False
THRESHOLD = 0.5
LABELS = ['benign', 'malign']

##################################################################
# n_samples / (n_classes * np.bincount(y))
# scaling by total / number of classes helps keep the loss to a similar magnitude.
# the sum of the weights of all examples stays the same.
##################################################################

weight_for_0 = 8000/(2.0*2000)
weight_for_1 = 8000/(2.0*6000)
CLASS_WEIGHT = {0: weight_for_0, 1: weight_for_1}
OUTPUT_DICT = True

# callback
MONITOR_MC = 'val_loss'
VERBOSE_MC = 1
SAVE_BEST_ONLY_MC = False
SAVE_WEIGHTS_ONLY_MC = False
MODE_MC = 'min'
PERIOD = 10

# reduce learning rate on plateau (LR)
MONITOR_LR = 'val_loss'
FACTOR_LR = 0.1
PATIENCE_LR = 8
MIN_LR_LR = 1e-7
MODE_LR = 'min'

# tensorboard
HISTOGRAM_FREQ = 1

MODEL_FORMAT = '.h5'
WEIGHT_FORMAT = '.h5'
CSV_FORMAT = '.csv'
IMG_FORMAT = '.png'
TEXT_FORMAT = '.txt'