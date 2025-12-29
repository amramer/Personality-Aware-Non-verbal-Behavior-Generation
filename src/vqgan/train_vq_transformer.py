import argparse
import json
import subprocess
import logging
import numpy as np
import os
import pdb
import scipy.io as sio

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.tensorboard import SummaryWriter

from vqmodules.gan_models import setup_vq_transformer, calc_vq_loss
import sys
import shutil

sys.path.append('../')
from utils.load_utils import *
from datetime import datetime


def generator_train_step(config, epoch, generator, g_optimizer, train_X,
                         rng, writer):
    """ Function to do autoencoding training for VQ-VAE

    Parameters
    ----------
    generator:
        VQ-VAE model that takes as input continuous listener and learns to
        outputs discretized listeners
    g_optimizer:
        optimizer that trains the VQ-VAE
    train_X:
        continuous listener motion sequence (acts as the target)
    """

    generator.train()
    batchinds = np.arange(train_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)
    total_epoch_loss = 0.0  # To accumulate loss over the entire epoch
    for bii, bi in enumerate(batchinds):
        idxStart = bi * config['batch_size']
        gtData_np = train_X[idxStart:(idxStart + config['batch_size']), :, :]
        gtData = Variable(torch.from_numpy(gtData_np),
                          requires_grad=False).cuda()
        prediction, quant_loss, perplexity = generator(gtData, None)
        g_loss = calc_vq_loss(prediction, gtData, quant_loss)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step_and_update_lr()
        total_epoch_loss += g_loss.detach().item()
        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}' \
                  .format(epoch, config['num_epochs'], bii, totalSteps,
                          g_loss.detach().item(), perplexity[0].item()))

    avg_epoch_loss = total_epoch_loss / totalSteps
    writer.add_scalar('Loss/train_totalLoss', avg_epoch_loss, epoch)



def generator_val_step(config, epoch, generator, g_optimizer, test_X,
                       currBestLoss, prev_save_epoch, tag, writer):
    """ Function that validates training of VQ-VAE

    see generator_train_step() for parameter definitions
    """

    generator.eval()
    batchinds = np.arange(test_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    total_val_loss = 0.0  # To accumulate loss over the entire validation epoch
    for bii, bi in enumerate(batchinds):
        idxStart = bi * config['batch_size']
        gtData_np = test_X[idxStart:(idxStart + config['batch_size']), :, :]
        gtData = Variable(torch.from_numpy(gtData_np),
                          requires_grad=False).cuda()
        with torch.no_grad():
            prediction, quant_loss, perplexity = generator(gtData, None)
        g_loss = calc_vq_loss(prediction, gtData, quant_loss)
        total_val_loss += g_loss.detach().item()

    avg_val_loss = total_val_loss / totalSteps
    print('val_Epoch [{}/{}], Average Loss: {:.4f}, Perplexity: {:5.4f}' \
          .format(epoch, config['num_epochs'], avg_val_loss, perplexity[0].item()))
    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', avg_val_loss, epoch)

    ## save model if curr loss is less than 180
    if avg_val_loss < 180:
        prev_save_epoch = epoch
        checkpoint = {'config': args.config,
                      'state_dict': generator.state_dict(),
                      'optimizer': {
                          'optimizer': g_optimizer._optimizer.state_dict(),
                          'n_steps': g_optimizer.n_steps,
                      },
                      'epoch': epoch}
        fileName = config['model_path'] + \
                   '{}{}_best.pth'.format(tag, config['pipeline'])
        print('>>>> saving best epoch {}'.format(epoch), avg_val_loss)
        torch.save(checkpoint, fileName)
        # Add graph to tensorboard

    return currBestLoss, prev_save_epoch, avg_val_loss





def main(args):
    """ full pipeline for training the Predictor model """

    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    print('using config', args.config)
    with open(args.config) as f:
        config = json.load(f)
    tag = config['tag']
    pipeline = config['pipeline']
    # create a timestamp string to use in the folder name
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # create a new folder for this experiment
    experiment_dir = os.path.join('experiments/round_3', f'experiment_{timestamp}')
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    # Prompt the user for experiment details
    experiment_type = input("What type of features do you want to quantize (speaking/listening)? ")
    remarks = input("Any remarks or description about the experiment? ")

    # Create a README for the experiment
    with open(os.path.join(experiment_dir, "README.md"), "w") as f:
        f.write(f"# Experiment Details: {timestamp}\n")
        f.write(f"\n## Experiment Type: {experiment_type}\n")
        f.write("\n## Hyperparameters:\n")
        for key, value in config.items():
            f.write(f"- {key}: {value}\n")
        f.write(f"\n## Remarks:\n{remarks}\n")

    # set the models and runs directories to be subdirectories of the experiment directory
    models_dir = os.path.join(experiment_dir, 'models/')
    runs_dir = os.path.join(experiment_dir, 'runs/')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    # save the models_dir path in the configuration file
    # config['model_path'] = models_dir
    with open(args.config, 'w') as f:
        json.dump(config, f, indent=4)
    currBestLoss = 1e3
    ## can modify via configs, these are default for released model
    seq_len = 32
    prev_save_epoch = 0
    with open(args.config) as f:
        config = json.load(f)

    # Save the updated configuration file in the experiment directory
    config_file = os.path.basename(args.config)
    experiment_config_path = os.path.join(experiment_dir, config_file)
    shutil.copyfile(args.config, experiment_config_path)
    # pdb.set_trace()

    writer = SummaryWriter(os.path.join(runs_dir, 'debug_{}{}').format(tag, pipeline))
    ## setting up models and loading last runned checkpoint
    fileName = config['model_path'] + \
               '{}{}_best.pth'.format(tag, config['pipeline'])
    load_path = fileName if os.path.exists(fileName) else None
    generator, g_optimizer, start_epoch = setup_vq_transformer(args, config,
                                                               version=None, load_path=load_path)
    generator.train()

    config['model_path'] = models_dir

    ## training/validation process
    _, _, train_listener, test_listener, _, _ , _ , _ ,= \
        load_data(config, pipeline, tag, rng,
                  segment_tag=config['segment_tag'], smooth=False)
    train_X = np.concatenate((train_listener[:, :seq_len, :],
                              train_listener[:, seq_len:, :]), axis=0)
    test_X = np.concatenate((test_listener[:, :seq_len, :],
                             test_listener[:, seq_len:, :]), axis=0)
    print('loaded listener...', train_X.shape, test_X.shape)
    disc_factor = 0.0
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if epoch == start_epoch + config['num_epochs'] - 1:
            print('early stopping at:', epoch)
            print('best loss:', currBestLoss)
            break
        generator_train_step(config, epoch, generator, g_optimizer, train_X,
                             rng, writer)
        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, generator, g_optimizer, test_X,
                               currBestLoss, prev_save_epoch, tag, writer)
    print('final best loss:', currBestLoss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    main(args)
