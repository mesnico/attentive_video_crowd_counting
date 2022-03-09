import os

import h5py
import numpy as np
import torch
import shutil

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best, model_name):
    filename = 'models/' + model_name + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        try:
            os.mkdir(os.path.dirname('models/'))
        except:
            pass
        shutil.copyfile(filename, 'models/model_best_' + model_name + '.pth.tar')
