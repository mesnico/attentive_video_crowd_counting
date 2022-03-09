import argparse
import csv
import json

import cv2
import scipy.io
import skimage
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.autograd import Variable
from torchinfo import summary
from torchvision import transforms
import yaml

import dataset
from image import *
from model import CANNet2s, XACANNet2s
from variables import MEAN, STD
from variables import PATCH_SIZE_PF

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                std=STD),
])

parser = argparse.ArgumentParser(description='PyTorch SACANNet2s')

parser.add_argument('--config', type=str, default=None, help='path to the configuration yaml config file')
parser.add_argument('--test_config', type=str, default=None, help='set an explicit json to perform inference on')
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

# the json file contains path of test images
if args.test_config is None:
    test_json_path = configs['test_json']
    WIDTH = configs['width']
    HEIGHT = configs['height']
    mask_outputs = configs['use_mask'] if 'use_mask' in configs else False
    dataset_name = configs['dataset']
else:
    with open(args.test_config, 'r') as ymlfile:
        test_configs = yaml.safe_load(ymlfile)
    test_json_path = test_configs['test_json']
    WIDTH = test_configs['width']
    HEIGHT = test_configs['height']
    mask_outputs = test_configs['use_mask'] if 'use_mask' in test_configs else False
    dataset_name = test_configs['dataset']
    print('Using {} configuration for testing!'.format(dataset_name))


with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

model_fn = eval(configs['model'])
model = model_fn(load_weights=False)

model = model.cuda()

summary(model, input_size=((1, 3, HEIGHT, WIDTH), (1, 3, HEIGHT, WIDTH)))

model.load_state_dict(checkpoint['state_dict'], strict=True)

model.eval()

pred = []
gt = []
errs = []
game = 0

test_dataset = dataset.listDataset(img_paths,
                                        shuffle=False,
                                        transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                                            std=STD),
                            ]),
                                        train=False,
                                        shape=(WIDTH, HEIGHT))

if mask_outputs:
    if dataset_name == 'ucsd':
        roi_path = 'datasets/ucsd/vidf-cvpr/vidf1_33_roi_mainwalkway.mat'
        roi = scipy.io.loadmat(roi_path)['roi'][0]
        roi = roi.item(0)[-1]
        roi = skimage.transform.resize(roi, (test_dataset[0][4].shape[0], test_dataset[0][4].shape[1]), order=0)
else:
    roi = None

for i in range(len(img_paths)):
    print(str(i) + "/" + str(len(img_paths)))
    img_path = img_paths[i]
    prev_img, img, post_img, prev_target, target, post_target, _ = test_dataset[i]

    prev_img = prev_img.cuda()
    img = img.cuda()

    # prev_img = prev_img.cuda()
    # prev_img = Variable(prev_img)
    #
    # img = img.cuda()
    # img = Variable(img)

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
        target *= roi

    pred_sum = overall.sum()
    print("PRED = " + str(pred_sum))
    pred.append(pred_sum)
    gt.append(np.sum(target))
    print("GT = " + str(np.sum(target)))
    errs.append(abs(np.sum(target) - pred_sum))

    # target = cv2.resize(target, (int(target.shape[1] / PATCH_SIZE_PF), int(target.shape[0] / PATCH_SIZE_PF)),
    #                     interpolation=cv2.INTER_CUBIC) * (PATCH_SIZE_PF ** 2)

    for k in range(target.shape[0]):
        for j in range(target.shape[1]):
            game += abs(overall[k][j] - target[k][j])

    print('MAE: ', mean_absolute_error(pred, gt))
    print('RMSE: ', np.sqrt(mean_squared_error(pred, gt)))
    print("GAME: " + str(game / (i + 1)) + "\n")

mae = mean_absolute_error(pred, gt)
rmse = np.sqrt(mean_squared_error(pred, gt))
game = game / len(pred)

print("FINAL RESULT")
print('MAE: ', mae)
print('RMSE: ', rmse)
print('GAME: ', game)

results = zip(errs, gt, pred)

header = ["Error", "GT", "Prediction"]

try:
    os.mkdir(os.path.dirname("results/"))
except:
    pass

# with open("results/model_best_" + MODEL_NAME + ".csv", "w") as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     for row in results:
#         writer.writerow(row)
