import json
from os.path import join
import os
import random

if __name__ == '__main__':
    # root is the path to your code, which is current directory
    root = 'fdst_dataset'
    # root_data is where you download the FDST dataset
    root_data = ''
    train_folders = join(root_data, 'train_data')
    test_folders = join(root_data, 'test_data')
    output_train_all = join(root, 'train_all.json')
    output_train = join(root, 'train.json')
    output_val = join(root, 'val.json')
    output_test = join(root, 'test.json')

    random.seed(42)

    train_img_list = []
    val_img_list = []
    test_img_list = []

    dirs = next(os.walk(train_folders))[1]
    train_dirs = random.sample(dirs, int(len(dirs) * 0.8))

    for dir_name in train_dirs:
        path = join(train_folders, dir_name)
        for _, _, files in os.walk(path):
            for file_name in files:
                if file_name.endswith('.jpg'):
                    train_img_list.append(join(path, file_name))

    val_dirs = list(set(dirs).difference(train_dirs))

    for dir_name in val_dirs:
        path = join(train_folders, dir_name)
        for _, _, files in os.walk(path):
            for file_name in files:
                if file_name.endswith('.jpg'):
                    val_img_list.append(join(path, file_name))

    """for root, dirs, files in os.walk(train_folders):
        for file_name in files:
            if file_name.endswith('.jpg'):
                train_all_img_list.append(join(root, file_name))"""

    for root, dirs, files in os.walk(test_folders):
        for file_name in files:
            if file_name.endswith('.jpg'):
                test_img_list.append(join(root, file_name))

    random.shuffle(train_img_list)

    with open(output_train_all, 'w') as f:
        json.dump(train_img_list, f)

    with open(output_train, 'w') as f:
        json.dump(train_img_list, f)

    with open(output_val, 'w') as f:
        json.dump(val_img_list, f)

    with open(output_test, 'w') as f:
        json.dump(test_img_list, f)