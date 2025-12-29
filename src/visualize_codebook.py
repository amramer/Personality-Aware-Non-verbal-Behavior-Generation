import argparse
import json
import logging
import numpy as np
import os
import pickle
import scipy.io as sio

import torch
import pdb
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable

from modules.fact_model import setup_model, calc_logit_loss
from vqgan.vqmodules.gan_models import setup_vq_transformer
from utils.load_utils import *
from scipy.spatial.distance import cdist

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
import numpy as np
import numpy as np


def save_pred(l_vqconfig, output_codebook):
    """ Method to saves predictions and probs to corresponding files """
    ## unstandardize outputs
    B, T, _ = output_codebook.shape
    preprocess = np.load(os.path.join('vqgan/', l_vqconfig['model_path'],
                                      '{}{}_preprocess_core.npz'.format(l_vqconfig['tag'],
                                                                        l_vqconfig['pipeline'])))
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_Y = output_codebook * body_std_Y + body_mean_Y

    frame_num = 0

    for b in range(B):
        for t in range(T):
            save_base = os.path.join('outputs/','codebook/codebook_complex_loss/')
            if not os.path.exists(save_base):
                os.makedirs(save_base)

            save_path = os.path.join(save_base,
                                     '{:08d}.pkl'.format(int(frame_num)))
            frame_num += 1

            data = {
                'exp': torch.from_numpy(test_Y[b, t, :50]).cuda()[None, ...].double(),
                'partbody_pose': torch.from_numpy(test_Y[b, t, 50:152]).cuda()[None, ...].double(),
                'neck_pose': torch.from_numpy(test_Y[b, t, 152:158]).cuda()[None, ...].double(),
                'head_pose': torch.from_numpy(test_Y[b, t, 158:164]).cuda()[None, ...].double(),
                'left_wrist_pose': torch.from_numpy(test_Y[b, t, 164:170]).cuda()[None, ...].double(),
                'right_wrist_pose': torch.from_numpy(test_Y[b, t, 170:176]).cuda()[None, ...].double(),
                'jaw_pose': torch.from_numpy(test_Y[b, t, 176:179]).cuda()[None, ...].double(),
                'left_hand_pose': torch.from_numpy(test_Y[b, t, 179:269]).cuda()[None, ...].double(),
                'right_hand_pose': torch.from_numpy(test_Y[b, t, 269:359]).cuda()[None, ...].double()
            }

            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

    print('done save', test_Y.shape)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)

    with open(args.config) as f:
        config = json.load(f)

    ## setup VQ-VAE model
    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)

    l_model_path = 'vqgan/' + l_vqconfig['model_path'] + \
                   '{}{}_best.pth'.format(l_vqconfig['tag'], l_vqconfig['pipeline'])
    l_vq_model, _, _ = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path,
                                            test=True)
    l_vq_model.eval()

    ## run model and save/eval
    # codebook indices
    codebook_entries = torch.arange(0, 800).view(100, 8)
    codebook_entries = codebook_entries.to(device)

    decoded_entires = None

    quant_size = (codebook_entries.shape[0], 500, 8)
    for t in range(0, codebook_entries.shape[-1], quant_size[-1]):
            curr_decoded = l_vq_model.module.decode_to_img(
                codebook_entries[:, t:t + quant_size[-1]], quant_size)
            decoded_entires = curr_decoded if decoded_entires is None \
                else torch.cat((decoded_entires, curr_decoded), axis=1)
    
    output_codebook = decoded_entires.data.cpu().numpy()

    if args.save:
        save_pred(l_vqconfig, output_codebook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--save', action='store_true')
    
    args = parser.parse_args()
    print(args)
    main(args)