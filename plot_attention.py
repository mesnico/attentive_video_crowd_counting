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
from model import SACANNet2s, XACANNet2s, CANNet2s, XARelCANNet2s
from variables import PATCH_SIZE_PF, MEAN, STD
from matplotlib import pyplot as plt
import tqdm


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

with open(test_json_path, 'r') as outfile:
    img_paths = json.load(outfile)

model_fn = eval(configs['model'])
model = model_fn(load_weights=False)

model = model.cuda()

model.load_state_dict(checkpoint['state_dict'])

model.eval()

test_dataset = dataset.listDataset(img_paths,
                                   shuffle=False,
                                   transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                                            std=STD),
                            ]),
                                   train=False,
                                   shape=(WIDTH, HEIGHT),
                                   output_original_target=True)

try:
    os.mkdir('attentions/')
except:
    pass

i = 3000    # 100
with torch.no_grad():
    img_path = img_paths[i]
    prev_img, img, post_img, prev_target, _, target, post_target, _ = test_dataset[i]

    prev_img = prev_img.cuda()
    img = img.cuda()

    img = img.unsqueeze(0)
    prev_img = prev_img.unsqueeze(0)

    prev_flow, attention = model(prev_img, img, return_att=True)
    # prev_flow_inverse = model(img, prev_img)

    img = Image.open(img_path).convert('RGB')
    fact = 8

    xy = np.mgrid[4:80:4, 4:45:4].reshape(2, -1).T
    for coord in tqdm.tqdm(xy):
        x_point, y_point = coord
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 20))
        np_img = np.array(img)
        scale = img.height / HEIGHT
        mask1 = attention[0][0, 0, y_point, x_point, :, :].cpu().numpy()
        # mask2 = attention[1][0, 0, y_point, x_point, :, :].cpu().numpy()
        ax1.imshow(np_img)
        ax1.add_patch(plt.Circle((x_point * fact * scale, y_point * fact * scale), 16, color='r'))
        ax2.imshow(mask1, cmap='cividis', interpolation='nearest')
        plt.tight_layout()
        ax1.axis('off')
        ax2.axis('off')
        # ax3.imshow(mask2, cmap='cividis', interpolation='nearest')
        plt.savefig('attentions/att_{}_{}.jpg'.format(x_point, y_point))

