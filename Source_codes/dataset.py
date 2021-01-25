from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os, shutil, random
import config as c

def debug_dataset():
    """ Double-checks if the dataset is split properly.
        Train, validation, and test dataset should not overlap.
    """
    for i in range(1, c.K_FOLD + 1):
        train_df = pd.read_csv(c.TRAIN_PATH + str(i) + c.CSV_FORMAT)
        val_df = pd.read_csv(c.VAL_PATH + str(i) + c.CSV_FORMAT)
        test_df = pd.read_csv(c.TEST_SET_CSV_PATH)

        train_ids = train_df['id'].tolist()
        val_ids = val_df['id'].tolist()
        test_ids = test_df['id'].tolist()

        set_ids = set(train_ids + val_ids + test_ids)
        list_ids = train_ids + val_ids + test_ids

        if len(set_ids) == len(list_ids):
            pass
        else:
            print('- Warning! Dataset is split incorretly!')

def load_dataset(data_type='new', shuffle=False, seed=1):
    """ Organizes the dataset using its labels.

        Args:
            shuffle: randomly shuffles the dataset if it is set to 'True'
        
        Returns:
            benign_imgs: a list of ids of the benign pathology images
            malign_imgs: a list of ids of the malign pathology images
    """
    benign_imgs = []
    malign_imgs = []

    if data_type == 'new':
        for subfolder in os.listdir(os.path.join(c.ORIGINAL_DATA_PATH, 'benign')):
            subpath = os.path.join(os.path.join(c.ORIGINAL_DATA_PATH, 'benign'), subfolder)
            for img in os.listdir(subpath):
                filename = 'benign/' + subfolder + '/' + img
                benign_imgs.append(filename)

        for subfolder in os.listdir(os.path.join(c.ORIGINAL_DATA_PATH, 'malignant')):
            subpath = os.path.join(os.path.join(c.ORIGINAL_DATA_PATH, 'malignant'), subfolder)
            for img in os.listdir(subpath):
                filename = 'malignant/' + subfolder + '/' + img
                malign_imgs.append(filename)

    elif data_type == 'old':
        df = pd.read_csv(c.DATA_CSV_PATH)
        
        for index, row in df.iterrows():
            if row['used'] == 1: 
                if row['label'] == 0:
                    img_path = str(row['save_id']) + c.IMG_FORMAT
                    benign_imgs.append(img_path)
                elif row['label'] == 1:
                    img_path = str(row['save_id']) + c.IMG_FORMAT
                    malign_imgs.append(img_path)

    if shuffle:
        seed = seed
        random.Random(seed).shuffle(benign_imgs)
        random.Random(seed).shuffle(malign_imgs)

    print('- Number of benign images: ' + str(len(benign_imgs)))
    print('- Number of malign images: ' + str(len(malign_imgs)))

    return benign_imgs, malign_imgs

def get_k_folds_dataset(train_split=c.TRAIN_SPLIT, test_split=c.TEST_SPLIT, data_type='new', shuffle=False):
    """ Splits the dataset into train, validation, and test set for k-fold cross validation.
        Saves the list of image ids for each k-fold's train, validation and test set into seperate CSV files.
        The folds are made by preserving the percentage of samples for each class.

        Args:
            train_split: the proportion for the train dataset.
            test_split: the proportion for the test dataset.
    """
    if train_split + test_split != 1.0:
        raise ('- Warning! Train split percentage or test split percentage is wrong!')

    benign_imgs, malign_imgs = load_dataset(data_type=data_type, shuffle=shuffle)

    test_path = os.path.join(c.TEST_PATH, 'all')
    for path in [c.DATA_PATH, c.TEST_PATH, test_path]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)

    benign_data = benign_imgs[:int(round(len(benign_imgs) * train_split))]
    malign_data = malign_imgs[:int(round(len(malign_imgs) * train_split))]

    benign_test = benign_imgs[int(round(len(benign_imgs) * train_split)):]
    malign_test = malign_imgs[int(round(len(malign_imgs) * train_split)):]

    for benign_test_img in benign_test:
        src = os.path.join(c.ORIGINAL_DATA_PATH, benign_test_img)
        if data_type == 'new':
            _names = benign_test_img.split('/')
            img_id = _names[1]
            file_id = _names[2].split('.')[0]
            new_name = img_id + '_' + file_id + '_0' + c.IMG_FORMAT
            dst = os.path.join(test_path, new_name)
            shutil.copyfile(src, dst)
        elif data_type == 'old':
            dst = os.path.join(test_path, benign_test_img.split('.')[0] + '_0' + c.IMG_FORMAT)
            shutil.copyfile(src, dst)

    for malign_test_img in malign_test:
        src = os.path.join(c.ORIGINAL_DATA_PATH, malign_test_img) 
        if data_type == 'new':
            _names = malign_test_img.split('/')
            img_id = _names[1]
            file_id = _names[2].split('.')[0]
            new_name = img_id + '_' + file_id + '_1' + c.IMG_FORMAT
            dst = os.path.join(test_path, new_name)
            shutil.copyfile(src, dst)
        elif data_type == 'old':
            dst = os.path.join(test_path, malign_test_img.split('.')[0] + '_1' + c.IMG_FORMAT)
            shutil.copyfile(src, dst)

    id_test = benign_test + malign_test
    label_test = [0 for i in range(len(benign_test))] + [1 for i in range(len(malign_test))]
    test_df = pd.DataFrame({
        'id': id_test,
        'label': label_test,
    })
    test_df.to_csv(c.TEST_SET_CSV_PATH, index=False)
    print('- Number of files in test dataset: ' + str(len(id_test)))

    for i in range(c.K_FOLD):

        size_0 = int(len(benign_data) / c.K_FOLD)
        size_1 = int(len(malign_data) / c.K_FOLD)
        if i < c.K_FOLD - 1:
            benign_train = benign_data[0:i * size_0] + benign_data[(i + 1) * size_0:]
            malign_train = malign_data[0: i * size_1] + malign_data[(i + 1) * size_1:]
            benign_val = benign_data[i * size_0: (i + 1) * size_0]
            malign_val = malign_data[i * size_1: (i + 1) * size_1]
        else:
            benign_train = benign_data[:i * size_0]
            malign_train = malign_data[:i * size_1]
            benign_val = benign_data[i * size_0:]
            malign_val = malign_data[i * size_1:]

        n = i+1
        print('- Number of files in benign train%d dataset: '%n + str(len(benign_train)))
        print('- Number of files in malign train%d dataset: '%n + str(len(malign_train)))

        print('- Number of files in benign validation%d dataset: '%n + str(len(benign_val)))
        print('- Number of files in malign validation%d dataset: '%n + str(len(malign_val)))

        id_train = benign_train + malign_train
        label_train = [0 for i in range(len(benign_train))] + [1 for i in range(len(malign_train))]

        id_val = benign_val + malign_val
        label_val = [0 for i in range(len(benign_val))] + [1 for i in range(len(malign_val))]

        train_df = pd.DataFrame(
            {'id': id_train,
             'label': label_train
            })
        train_df.to_csv(c.TRAIN_PATH + str(i+1) + c.CSV_FORMAT, index=False)
        val_df = pd.DataFrame(
            {'id': id_val,
             'label': label_val
            })
        val_df.to_csv(c.VAL_PATH + str(i+1) + c.CSV_FORMAT, index=False)
