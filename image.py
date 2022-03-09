import os
from PIL import Image
import numpy as np
import h5py
import cv2


def load_data(img_path, train=True, shape=None):
    WIDTH, HEIGHT = shape
    img_folder = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)

    ext = img_name.split('.')[1]
    fname = img_name.split('.')[0]
    if img_name.startswith('vidf'):
        # handle ucsd
        index = int(fname[-3:])
        prefix = fname[:-3]
        max_frame = 200
    else:   # TODO: handle other possible datasets! the choise should be done on an explicit dataset variable (not startswith or crap like that)
        # fdst
        index = int(fname)
        prefix = ""
        max_frame = 150

    prev_index = int(max(1, index - 5))
    post_index = int(min(max_frame, index + 5))

    prev_img_path = os.path.join(img_folder, prefix+'%03d.%s' % (prev_index, ext))
    post_img_path = os.path.join(img_folder, prefix+'%03d.%s' % (post_index, ext))

    prev_gt_path = prev_img_path.replace('.{}'.format(ext), '_resize.h5')
    gt_path = img_path.replace('.{}'.format(ext), '_resize.h5')
    post_gt_path = post_img_path.replace('.{}'.format(ext), '_resize.h5')

    prev_img = Image.open(prev_img_path).convert('RGB')
    img = Image.open(img_path).convert('RGB')
    post_img = Image.open(post_img_path).convert('RGB')

    prev_img = prev_img.resize((WIDTH, HEIGHT))
    img = img.resize((WIDTH, HEIGHT))
    post_img = post_img.resize((WIDTH, HEIGHT))

    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    gt_file.close()

    prev_gt_file = h5py.File(prev_gt_path)
    prev_target = np.asarray(prev_gt_file['density'])
    prev_gt_file.close()

    post_gt_file = h5py.File(post_gt_path)
    post_target = np.asarray(post_gt_file['density'])
    post_gt_file.close()

    return prev_img, img, post_img, prev_target, target, post_target
