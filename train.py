import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import skimage
import scipy.io
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, cm
from torch import nn
from torch.autograd import Variable
from torchinfo import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import yaml

import dataset
from model import SACANNet2s, CANNet2s, XACANNet2s, XARelCANNet2s, XAAFCANNet2s
from utils import save_checkpoint
from variables import MEAN, STD, PATCH_SIZE_PF

parser = argparse.ArgumentParser(description='PyTorch SACANNet2s')

parser.add_argument('config', type=str, default='configs/fdst.yaml', help='path to the configuration yaml config file')
parser.add_argument('--experiment', type=str, default='test', help='Experiment name')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size (look also --virtual_batch_size if you dont have enough GPU memory)')
parser.add_argument('--virtual_batch_size', type=int, default=10, help='Batch size for batch accumulation')


def plotDensity(density, axarr, k):
    '''
    @density: np array of corresponding density map
    '''
    density = density * 255.0

    # plot with overlay
    colormap_i = cm.jet(density)[:, :, 0:3]

    overlay_i = colormap_i

    new_map = overlay_i.copy()
    new_map[:, :, 0] = overlay_i[:, :, 2]
    new_map[:, :, 2] = overlay_i[:, :, 0]

    axarr[k].imshow(255 * new_map.astype(np.uint8))


class Criterion(nn.Module):
    def __init__(self, uncertainty=False):
        super().__init__()
        self.uncertainty = uncertainty
        if uncertainty:
            self.base_var = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x, target, unc=True):
        if self.uncertainty and unc:
            mean, var = x[0, ...], x[1, ...]
            var = var + self.base_var
            nonzero_var_idxs = (var != 0)
            loss1 = ((1 / var[nonzero_var_idxs]) * (mean[nonzero_var_idxs] - target[nonzero_var_idxs]) ** 2).sum()
            loss2 = torch.log(var[nonzero_var_idxs]).sum()

            return 0.5 * (loss1 + loss2)
        else:
            return F.mse_loss(x, target, reduction='sum')

HEIGHT = 0
WIDTH = 0
MODEL_NAME = ""

def main():
    global args, HEIGHT, WIDTH, MODEL_NAME

    args = parser.parse_args()
    args.best_prec1 = 1e6
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.start_frame = 0
    args.workers = 4
    args.seed = int(time.time())

    # load configuration file into globals
    with open(args.config, 'r') as ymlfile:
        configs = yaml.safe_load(ymlfile)

    with open(configs['train_json'], 'r') as outfile:
        args.train_list = json.load(outfile)
    with open(configs['val_json'], 'r') as outfile:
        args.val_list = json.load(outfile)

    args.uncertainty = True if 'uncertainty' in configs and configs['uncertainty'] else False

    MODEL_NAME = args.experiment # configs['experiment_name']
    configs['experiment_name'] = MODEL_NAME
    WIDTH = configs['width']
    HEIGHT = configs['height']
    args.lr = configs['lr']
    args.print_freq = configs['log_freq']
    args.log_freg = configs['val_freq']


    if 'use_mask' in configs and configs['use_mask']:
        if configs['dataset'] == 'ucsd':
            # load ucsd mask
            roi_path = 'datasets/ucsd/vidf-cvpr/vidf1_33_roi_mainwalkway.mat'
            roi = scipy.io.loadmat(roi_path)['roi'][0]
            roi = roi.item(0)[-1]
            frame_mask = skimage.transform.resize(roi, (HEIGHT, WIDTH), order=0)

    else:
        frame_mask = None

    torch.cuda.manual_seed(args.seed)

    model_fn = eval(configs['model'])
    model = model_fn(load_weights=False, uncertainty=args.uncertainty) # SACANNet2s(load_weights=False, fine_tuning=False)

    model = model.cuda()

    criterion = Criterion(uncertainty=args.uncertainty)
    criterion = criterion.cuda()
    criterion_no_uncr = Criterion(uncertainty=False)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 weight_decay=args.decay)

    # summary(model, input_size=((args.batch_size, 3, HEIGHT, WIDTH), (args.batch_size, 3, HEIGHT, WIDTH)))

    # modify the path of saved checkpoint if necessary
    try:
        checkpoint = torch.load('models/' + MODEL_NAME + '.pth.tar', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(optimizer)
        args.start_epoch = checkpoint['epoch']
        args.start_frame = checkpoint['start_frame']
        try:
            args.best_prec1 = checkpoint['best_prec'].item()
        except:
            args.best_prec1 = checkpoint['best_prec']
        print("Train model " + MODEL_NAME + " from epoch " + str(args.start_epoch) + " with best prec = " + str(
            args.best_prec1) + "...")
    except:
        print("Train model " + MODEL_NAME + "...")

    for epoch in range(args.start_epoch, args.epochs):
        train(configs, args.train_list, model, criterion, optimizer, epoch, frame_mask_orig=frame_mask)
        prec1 = validate(configs, args.val_list, model, criterion, frame_mask_orig=frame_mask)

        is_best = prec1 < args.best_prec1
        args.best_prec1 = min(prec1, args.best_prec1)
        args.start_frame = 0
        print(' * best MAE {mse:.3f} '
              .format(mse=args.best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'start_frame': 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec': args.best_prec1,
            'config': configs
        }, is_best, model_name=MODEL_NAME)


def compute_densities_from_flows(prev_flow, post_flow, prev_flow_inverse, post_flow_inverse):
    # mask the boundary locations where people can move in/out between regions outside image plane
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

    reconstruction_from_post = torch.sum(post_flow[0, :9, :, :], dim=0) + post_flow[0, 9, :, :] * mask_boundry

    reconstruction_from_prev_inverse = torch.sum(prev_flow_inverse[0, :9, :, :], dim=0) + prev_flow_inverse[0, 9, :,
                                                                                          :] * mask_boundry

    reconstruction_from_post_inverse = F.pad(post_flow_inverse[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(
        post_flow_inverse[0, 1, 1:, :], (0, 0, 0, 1)) + F.pad(post_flow_inverse[0, 2, 1:, :-1],
                                                              (1, 0, 0, 1)) + F.pad(post_flow_inverse[0, 3, :, 1:],
                                                                                    (0, 1, 0,
                                                                                     0)) + post_flow_inverse[0, 4,
                                                                                           :, :] + F.pad(
        post_flow_inverse[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(post_flow_inverse[0, 6, :-1, 1:],
                                                               (0, 1, 1, 0)) + F.pad(
        post_flow_inverse[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(post_flow_inverse[0, 8, :-1, :-1],
                                                               (1, 0, 1, 0)) + post_flow_inverse[0, 9, :,
                                                                               :] * mask_boundry

    prev_density_reconstruction = torch.sum(prev_flow[0, :9, :, :], dim=0) + prev_flow[0, 9, :, :] * mask_boundry
    prev_density_reconstruction_inverse = F.pad(prev_flow_inverse[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(
        prev_flow_inverse[0, 1, 1:, :], (0, 0, 0, 1)) + F.pad(prev_flow_inverse[0, 2, 1:, :-1],
                                                              (1, 0, 0, 1)) + F.pad(prev_flow_inverse[0, 3, :, 1:],
                                                                                    (0, 1, 0,
                                                                                     0)) + prev_flow_inverse[0, 4,
                                                                                           :, :] + F.pad(
        prev_flow_inverse[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(prev_flow_inverse[0, 6, :-1, 1:],
                                                               (0, 1, 1, 0)) + F.pad(
        prev_flow_inverse[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(prev_flow_inverse[0, 8, :-1, :-1],
                                                               (1, 0, 1, 0)) + prev_flow_inverse[0, 9, :,
                                                                               :] * mask_boundry

    post_density_reconstruction_inverse = torch.sum(post_flow_inverse[0, :9, :, :], dim=0) + post_flow_inverse[0, 9,
                                                                                             :, :] * mask_boundry
    post_density_reconstruction = F.pad(post_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(post_flow[0, 1, 1:, :],
                                                                                       (0, 0, 0, 1)) + F.pad(
        post_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(post_flow[0, 3, :, 1:], (0, 1, 0, 0)) + post_flow[0, 4, :,
                                                                                                :] + F.pad(
        post_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(post_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + F.pad(
        post_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(post_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + post_flow[0, 9, :,
                                                                                                  :] * mask_boundry

    prev_reconstruction_from_prev = torch.sum(prev_flow[0, :9, :, :], dim=0) + prev_flow[0, 9, :, :] * mask_boundry
    post_reconstruction_from_post = F.pad(post_flow[0, 0, 1:, 1:], (0, 1, 0, 1)) + F.pad(post_flow[0, 1, 1:, :],
                                                                                         (0, 0, 0, 1)) + F.pad(
        post_flow[0, 2, 1:, :-1], (1, 0, 0, 1)) + F.pad(post_flow[0, 3, :, 1:], (0, 1, 0, 0)) + post_flow[0, 4, :,
                                                                                                :] + F.pad(
        post_flow[0, 5, :, :-1], (1, 0, 0, 0)) + F.pad(post_flow[0, 6, :-1, 1:], (0, 1, 1, 0)) + F.pad(
        post_flow[0, 7, :-1, :], (0, 0, 1, 0)) + F.pad(post_flow[0, 8, :-1, :-1], (1, 0, 1, 0)) + post_flow[0, 9, :,
                                                                                                  :] * mask_boundry

    return reconstruction_from_prev, reconstruction_from_post, reconstruction_from_prev_inverse, reconstruction_from_post_inverse, prev_reconstruction_from_prev, post_reconstruction_from_post


def train(config, train_list, model, criterion, optimizer, epoch, frame_mask_orig=None):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    alb_transforms = A.Compose([
        A.RandomResizedCrop(HEIGHT, WIDTH, scale=(0.75, 1.0), ratio=(0.95, 1.05), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=MEAN, std=STD)
    ],
        additional_targets={'image1': 'image', 'image2': 'image', 'density': 'mask', 'density1': 'mask', 'density2': 'mask', 'mask': 'mask'}
    )

    # alb_transforms = transforms.Compose([
    #                             transforms.ToTensor(), transforms.Normalize(mean=MEAN,
    #                                                                         std=STD),
    #                         ])

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            shuffle=True,
                            transform=alb_transforms,
                            train=True,
                            batch_size=args.batch_size,
                            num_workers=args.workers,
                            shape=(WIDTH, HEIGHT),
                            mask=frame_mask_orig),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (
        epoch, epoch * len(train_loader.dataset) + args.start_frame, args.lr))

    model.train()
    end = time.time()

    optimizer.zero_grad()
    for i, (prev_img, img, post_img, prev_target, target, post_target, frame_mask) in enumerate(train_loader):
        # if i + 1 <= args.start_frame:
        #     if (i + 1) % args.print_freq == 0:
        #         print(i + 1)
        #     continue
        data_time.update(time.time() - end)

        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        post_img = post_img.cuda()
        post_img = Variable(post_img)

        frame_mask = frame_mask.cuda()

        prev_flow = model(prev_img, img)
        post_flow = model(img, post_img)

        prev_flow_inverse = model(img, prev_img)
        post_flow_inverse = model(post_img, img)

        if 'use_mask' in config and config['use_mask']:
            prev_flow = prev_flow * frame_mask
            post_flow = post_flow * frame_mask
            prev_flow_inverse = prev_flow_inverse * frame_mask
            post_flow_inverse = post_flow_inverse * frame_mask

        target = target.type(torch.FloatTensor)[0].cuda()
        target = Variable(target)

        prev_target = prev_target.type(torch.FloatTensor)[0].cuda()
        prev_target = Variable(prev_target)

        post_target = post_target.type(torch.FloatTensor)[0].cuda()
        post_target = Variable(post_target)

        reconstruction_from_prev, reconstruction_from_post, reconstruction_from_prev_inverse, reconstruction_from_post_inverse, prev_reconstruction_from_prev, post_reconstruction_from_post = \
            compute_densities_from_flows(prev_flow[:, :10, :, :], post_flow[:, :10, :, :],
                                         prev_flow_inverse[:, :10, :, :], post_flow_inverse[:, :10, :, :])

        if args.uncertainty:
            reconstruction_from_prev_var, reconstruction_from_post_var, reconstruction_from_prev_inverse_var, reconstruction_from_post_inverse_var, prev_reconstruction_from_prev_var, post_reconstruction_from_post_var = \
                compute_densities_from_flows(prev_flow[:, 10:, :, :], post_flow[:, 10:, :, :],
                                             prev_flow_inverse[:, 10:, :, :], post_flow_inverse[:, 10:, :, :])

            # merge with the flows (on dimension 0)
            reconstruction_from_prev = torch.stack([reconstruction_from_prev, reconstruction_from_prev_var])
            reconstruction_from_post = torch.stack([reconstruction_from_post, reconstruction_from_post_var])
            reconstruction_from_prev_inverse = torch.stack([reconstruction_from_prev_inverse, reconstruction_from_prev_inverse_var])
            reconstruction_from_post_inverse = torch.stack([reconstruction_from_post_inverse, reconstruction_from_post_inverse_var])
            prev_reconstruction_from_prev = torch.stack([prev_reconstruction_from_prev, prev_reconstruction_from_prev_var])
            post_reconstruction_from_post = torch.stack([post_reconstruction_from_post, post_reconstruction_from_post_var])

        if 'use_mask' in config and config['use_mask']:
            frame_mask = frame_mask.squeeze(0)
            # reconstruction_from_prev = reconstruction_from_prev * frame_mask
            # reconstruction_from_post = reconstruction_from_post * frame_mask
            # reconstruction_from_prev_inverse = reconstruction_from_prev_inverse * frame_mask
            # reconstruction_from_post_inverse = reconstruction_from_post_inverse * frame_mask
            # prev_reconstruction_from_prev = prev_reconstruction_from_prev * frame_mask
            # post_reconstruction_from_post = post_reconstruction_from_post * frame_mask

            target = target * frame_mask
            prev_target = prev_target * frame_mask
            post_target = post_target * frame_mask

        loss_prev_flow = criterion(reconstruction_from_prev, target)
        loss_post_flow = criterion(reconstruction_from_post, target)
        loss_prev_flow_inverse = criterion(reconstruction_from_prev_inverse, target)
        loss_post_flow_inverse = criterion(reconstruction_from_post_inverse, target)
        loss_prev = criterion(prev_reconstruction_from_prev, prev_target)
        loss_post = criterion(post_reconstruction_from_post, post_target)

        # cycle consistency
        loss_prev_consistency = criterion(prev_flow[0, 0, 1:, 1:], prev_flow_inverse[0, 8, :-1, :-1], unc=False) + criterion(
            prev_flow[0, 1, 1:, :], prev_flow_inverse[0, 7, :-1, :], unc=False) + criterion(prev_flow[0, 2, 1:, :-1],
                                                                                 prev_flow_inverse[0, 6, :-1,
                                                                                 1:], unc=False) + criterion(
            prev_flow[0, 3, :, 1:], prev_flow_inverse[0, 5, :, :-1], unc=False) + criterion(prev_flow[0, 4, :, :],
                                                                                 prev_flow_inverse[0, 4, :,
                                                                                 :], unc=False) + criterion(
            prev_flow[0, 5, :, :-1], prev_flow_inverse[0, 3, :, 1:], unc=False) + criterion(prev_flow[0, 6, :-1, 1:],
                                                                                 prev_flow_inverse[0, 2, 1:,
                                                                                 :-1], unc=False) + criterion(
            prev_flow[0, 7, :-1, :], prev_flow_inverse[0, 1, 1:, :], unc=False) + criterion(prev_flow[0, 8, :-1, :-1],
                                                                                 prev_flow_inverse[0, 0, 1:, 1:], unc=False)

        loss_post_consistency = criterion(post_flow[0, 0, 1:, 1:], post_flow_inverse[0, 8, :-1, :-1], unc=False) + criterion(
            post_flow[0, 1, 1:, :], post_flow_inverse[0, 7, :-1, :], unc=False) + criterion(post_flow[0, 2, 1:, :-1],
                                                                                 post_flow_inverse[0, 6, :-1,
                                                                                 1:], unc=False) + criterion(
            post_flow[0, 3, :, 1:], post_flow_inverse[0, 5, :, :-1], unc=False) + criterion(post_flow[0, 4, :, :],
                                                                                 post_flow_inverse[0, 4, :,
                                                                                 :], unc=False) + criterion(
            post_flow[0, 5, :, :-1], post_flow_inverse[0, 3, :, 1:], unc=False) + criterion(post_flow[0, 6, :-1, 1:],
                                                                                 post_flow_inverse[0, 2, 1:,
                                                                                 :-1], unc=False) + criterion(
            post_flow[0, 7, :-1, :], post_flow_inverse[0, 1, 1:, :], unc=False) + criterion(post_flow[0, 8, :-1, :-1],
                                                                                 post_flow_inverse[0, 0, 1:, 1:], unc=False)

        loss = loss_prev_flow + loss_post_flow + loss_prev_flow_inverse + loss_post_flow_inverse + loss_prev + loss_post + loss_prev_consistency + loss_post_consistency

        losses.update(loss.item(), img.size(0))
        loss.backward()
        if (i + 1) % args.virtual_batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print("\nTarget = " + str(torch.sum(target)))
            if len(reconstruction_from_prev.shape) == 3:
                reconstruction_from_prev = reconstruction_from_prev[0]
                reconstruction_from_post = reconstruction_from_post[0]
                reconstruction_from_prev_inverse = reconstruction_from_prev_inverse[0]
                reconstruction_from_post_inverse = reconstruction_from_post_inverse[0]

            overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0).data.cpu().numpy()
            pred_sum = overall.sum()
            print("Pred = " + str(pred_sum))
            print("Reconstruction from prev = " + str(torch.sum(reconstruction_from_prev)))
            print("Reconstruction from post = " + str(torch.sum(reconstruction_from_post)))
            print("Reconstruction from prev inverse = " + str(torch.sum(reconstruction_from_prev_inverse)))
            print("Reconstruction from post inverse = " + str(torch.sum(reconstruction_from_post_inverse)))
            print("Prev Target = " + str(torch.sum(prev_target)))
            print("Prev Reconstruction from prev = " + str(torch.sum(reconstruction_from_prev)))
            print("Post Target = " + str(torch.sum(post_target)))
            print("Post Reconstruction from post = " + str(torch.sum(reconstruction_from_post)) + "\n")

            print("loss_prev_flow = " + str(loss_prev_flow))
            print("loss_post_flow = " + str(loss_post_flow))
            print("loss_prev_flow_inverse = " + str(loss_prev_flow_inverse))
            print("loss_post_flow_inverse = " + str(loss_post_flow_inverse))
            print("loss_prev = " + str(loss_prev))
            print("loss_post = " + str(loss_post))
            print("loss_prev_consistency = " + str(loss_prev_consistency))
            print("loss_post_consistency = " + str(loss_post_consistency))

            pred = cv2.resize(overall, (overall.shape[1] * PATCH_SIZE_PF, overall.shape[0] * PATCH_SIZE_PF),
                              interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)

            target = cv2.resize(target.cpu().detach().numpy(),
                                (target.shape[1] * PATCH_SIZE_PF, target.shape[0] * PATCH_SIZE_PF),
                                interpolation=cv2.INTER_CUBIC) / (PATCH_SIZE_PF ** 2)
            fig, axarr = plt.subplots(1, 2)
            plotDensity(pred, axarr, 0)
            plotDensity(target, axarr, 1)
            plt.show()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        if ((i + 1) % args.log_freg == 0) & ((i + 1) != len(train_loader)):
            prec1 = validate(config, args.val_list, model, criterion, frame_mask_orig=frame_mask_orig)

            is_best = prec1 < args.best_prec1
            args.best_prec1 = min(prec1, args.best_prec1)
            print(' * best MAE {mae:.3f} '
                  .format(mae=args.best_prec1))
            save_checkpoint({
                'epoch': epoch,
                'start_frame': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec': args.best_prec1,
                'config': config
            }, is_best, model_name=MODEL_NAME)


def validate(config, val_list, model, criterion, frame_mask_orig=None):
    print('begin val')
    val_loader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=MEAN,
                                                                            std=STD),
                            ]),
                            train=False,
                            shape=(WIDTH, HEIGHT),
                            mask=frame_mask_orig),
        batch_size=args.batch_size)

    model.eval()

    mae = 0.0
    mse = 0.0

    for i, (prev_img, img, post_img, _, target, _, frame_mask) in enumerate(val_loader):
        # only use previous frame in inference time, as in real-time application scenario, future frame is not available
        prev_img = prev_img.cuda()
        prev_img = Variable(prev_img)

        img = img.cuda()
        img = Variable(img)

        frame_mask = frame_mask.cuda()

        with torch.no_grad():
            prev_flow = model(prev_img, img)[:, :10, ...]
            prev_flow_inverse = model(img, prev_img)[:, :10, ...]

        target = target.type(torch.FloatTensor)[0].cuda()
        target = Variable(target)

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

        overall = ((reconstruction_from_prev + reconstruction_from_prev_inverse) / 2.0)
        if 'use_mask' in config and config['use_mask']:
            overall *= frame_mask.squeeze(0)
            target *= frame_mask.squeeze(0)

        overall = overall.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)

        if i % args.print_freq == 0:
            print("PRED = " + str(overall.data.sum()))
            print("GT = " + str(target.sum()))
        mae += abs(overall.data.sum() - target.sum())
        mse += abs(overall.data.sum() - target.sum()) * abs(overall.data.sum() - target.sum())

    mae = mae / len(val_loader)
    mse = math.sqrt(mse / len(val_loader))
    print(' * MAE {mae:.3f} '
          .format(mae=mae))
    print(' * MSE {mse:.3f} '
          .format(mse=mse))

    return mae


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
