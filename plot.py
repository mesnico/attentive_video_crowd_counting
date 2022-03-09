import argparse
import json

import cv2
import skimage
import torch
import torch.nn.functional as F
import yaml
import scipy.io
from matplotlib import cm
from torch.autograd import Variable
from torchvision import transforms

import dataset
from image import *
from model import SACANNet2s, XACANNet2s, CANNet2s
from variables import PATCH_SIZE_PF, MEAN, STD


def plotDensity(density, plot_path):
    '''
    @density: np array of corresponding density map
    @plot_path: path to save the plot
    '''
    density = density * 255.0

    # plot with overlay
    colormap_i = cm.jet(density)[:, :, 0:3]

    overlay_i = colormap_i

    new_map = overlay_i.copy()
    new_map[:, :, 0] = overlay_i[:, :, 2]
    new_map[:, :, 2] = overlay_i[:, :, 0]

    try:
        os.mkdir(os.path.dirname(plot_path))
    except:
        pass

    cv2.imwrite(plot_path, new_map * 255)

mask_outputs = False
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                std=STD),
])

parser = argparse.ArgumentParser(description='PyTorch SACANNet2s')

parser.add_argument('--config', type=str, default=None, help='path to the configuration yaml config file')
parser.add_argument('checkpoint', type=str, default='test', help='Experiment name')
args = parser.parse_args()

# modify the path of saved checkpoint if necessary
checkpoint = torch.load(args.checkpoint, map_location='cpu')
assert 'config' in checkpoint or args.config is not None, "If the configuration is not into the checkpoint, it should be passed with the --config argument"
if 'config' in checkpoint:
    configs = checkpoint['config']
    print('Config loaded from the checkpoint!')
else:
    # load configuration file into globals
    with open(args.config, 'r') as ymlfile:
        configs = yaml.safe_load(ymlfile)

WIDTH = configs['width']
HEIGHT = configs['height']
# the json file contains path of test images
test_json_path = configs['test_json']

# the folder to output density map and flow maps
output_folder = os.path.join('plot', MODEL_NAME)

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

model_fn = eval(configs['model'])
model = model_fn(load_weights=False)

model = model.cuda()

model.load_state_dict(checkpoint['state_dict'])

model.eval()

pred = []
gt = []

test_dataset = dataset.listDataset(img_paths,
                                   shuffle=False,
                                   transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                                            std=STD),
                            ]),
                                   train=False,
                                   shape=(WIDTH, HEIGHT),
                                   output_original_target=True)

roi_path = 'datasets/ucsd/vidf-cvpr/vidf1_33_roi_mainwalkway.mat'
roi = scipy.io.loadmat(roi_path)['roi'][0]
roi = roi.item(0)[-1]
roi = skimage.transform.resize(roi, (test_dataset[0][4].shape[0], test_dataset[0][4].shape[1]), order=0)

try:
    os.mkdir(os.path.dirname('plot/'))
except:
    pass

try:
    os.mkdir(os.path.dirname(os.path.join('plot', MODEL_NAME + '/')))
except:
    pass

with torch.no_grad():
    for i in range(0, len(img_paths), 100):
        img_path = img_paths[i]
        prev_img, img, post_img, prev_target, _, target, post_target, _ = test_dataset[i]

        prev_img = prev_img.cuda()
        img = img.cuda()

        img = img.unsqueeze(0)
        prev_img = prev_img.unsqueeze(0)

        with torch.no_grad():
            prev_flow = model(prev_img, img)
            prev_flow_inverse = model(img, prev_img)

        mask_boundry = torch.zeros(prev_flow.shape[2:])
        mask_boundry[0, :] = 1.0
        mask_boundry[-1, :] = 1.0
        mask_boundry[:, 0] = 1.0
        mask_boundry[:, -1] = 1.0

        mask_boundry = Variable(mask_boundry.cuda())

        reconstruction_from_prev = F.pad(prev_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(prev_flow[0, 1, 1:, :],
                                                                                        (0, 0, 0, 1)) + F.pad(
            prev_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(prev_flow[0, 3, :, 1:], (0, 1, 0, 0)) + prev_flow[0, 4, :,
                                                                                                    :] + F.pad(
            prev_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + F.pad(
            prev_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + prev_flow[0, 9, :,
                                                                                                      :] * mask_boundry

        reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :,
                                                                                              :] * mask_boundry

        overall = (reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0
        overall = overall.cpu().numpy()
        if mask_outputs:
            overall *= roi

        base_name = os.path.basename(img_path)
        ext = os.path.splitext(img_path)[1]
        print(base_name)
        folder_name = os.path.dirname(img_path).split('/')[-1]
        print(folder_name)
        gt_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_gt'+ext)
        print(gt_path)
        density_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_pred'+ext)
        flow_1_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_1'+ext)
        flow_2_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_2'+ext)
        flow_3_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_3'+ext)
        flow_4_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_4'+ext)
        flow_5_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_5'+ext)
        flow_6_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_6'+ext)
        flow_7_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_7'+ext)
        flow_8_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_8'+ext)
        flow_9_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_flow_9'+ext)

        pred = cv2.resize(overall, (overall.shape[1] * PATCH_SIZE_PF, overall.shape[0] * PATCH_SIZE_PF),
                          interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        #target = cv2.resize(target, (target.shape[1] * PATCH_SIZE_PF, target.shape[0] * PATCH_SIZE_PF),
        #                  interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        prev_flow = prev_flow.data.cpu().numpy()[0]
        flow_1 = cv2.resize(prev_flow[0], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        flow_2 = cv2.resize(prev_flow[1], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        flow_3 = cv2.resize(prev_flow[2], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        flow_4 = cv2.resize(prev_flow[3], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        flow_5 = cv2.resize(prev_flow[4], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        flow_6 = cv2.resize(prev_flow[5], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        flow_7 = cv2.resize(prev_flow[6], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        flow_8 = cv2.resize(prev_flow[7], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
        flow_9 = cv2.resize(prev_flow[8], (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)

        plotDensity(pred, density_path)
        plotDensity(target, gt_path)
        plotDensity(flow_1, flow_1_path)
        plotDensity(flow_2, flow_2_path)
        plotDensity(flow_3, flow_3_path)
        plotDensity(flow_4, flow_4_path)
        plotDensity(flow_5, flow_5_path)
        plotDensity(flow_6, flow_6_path)
        plotDensity(flow_7, flow_7_path)
        plotDensity(flow_8, flow_8_path)
        plotDensity(flow_9, flow_9_path)

        roi_resized = roi #cv2.resize(roi, (roi.shape[1] * PATCH_SIZE_PF, roi.shape[0] * PATCH_SIZE_PF),
                      #      interpolation=cv2.INTER_CUBIC)
        roi_path = os.path.join(output_folder, folder_name, base_name).replace(ext, '_roi' + ext)
        plotDensity(roi_resized, roi_path)
