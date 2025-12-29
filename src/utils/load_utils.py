import cv2
import numpy as np
import os
import scipy
import pickle
import pandas as pd
import pdb
import random

from torchvision import transforms
import torch
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-10


def bilateral_filter(outputs):
    """ smoothing function

    function that applies bilateral filtering along temporal dim of sequence.
    """
    # print(outputs)
    # Ensure outputs is in the correct data type (float32)
    outputs = outputs.astype(np.float32)
    outputs_smooth = np.zeros(outputs.shape)
    for b in range(outputs.shape[0]):
        for f in range(outputs.shape[2]):
            smoothed = np.reshape(cv2.bilateralFilter(
                outputs[b, :, f], 5, 20, 20), (-1))
            outputs_smooth[b, :, f] = smoothed
    return outputs_smooth.astype(np.float32)


def create_data_vq(l_vq_model, speakerData_np, listenerData_np, audioData_np,
                   personalityData_np, seq_len, startpoint=0, midpoint=None, data_type='on_logit',
                   btc=None, patch_size=8):
    """ data preparation function

    processes the data by truncating full input sequences to remove future info,
    and converts listener raw motion to listener codebook indices
    """

    speakerData = Variable(torch.from_numpy(speakerData_np),
                           requires_grad=False).cuda()
    listenerData = Variable(torch.from_numpy(listenerData_np),
                            requires_grad=False).cuda()
    audioData = Variable(torch.from_numpy(audioData_np),
                         requires_grad=False).cuda()
    
    ## future timesteps for speaker inputs (keep past and current context)
    speaker_full = speakerData[:, :(seq_len + patch_size), :].float()
    audio_full = audioData[:, :(seq_len + patch_size) * 4, :].float()

    personalityData = None
    
    if personalityData_np is not None:
        personalityData = Variable(torch.from_numpy(personalityData_np),
                         requires_grad=False).cuda()
        personalityData = personalityData[:, :].float()

    ## convert listener past inputs to codebook indices
    with torch.no_grad():
        if listenerData.dim() == 3:
            # if listener input is in the raw format, directly convert to indxs
            listener_past, listener_past_index = \
                l_vq_model.module.get_quant(listenerData[:, :seq_len, :].float())
            # print("listener past: ", listener_past.shape)
            # print("listener past index before reshaping: ",listener_past_index.shape)
            btc = listener_past.shape[0], \
                  listener_past.shape[2], \
                  listener_past.shape[1]
            listener_past_index = torch.reshape(listener_past_index,
                                                (listener_past.shape[0], -1))
        else:
            # if listener input is already in index format, fetch the quantized
            # raw listener and then re-encode into a new set of indxs
            tmp_past_index = listenerData[:, :btc[1]]
            tmp_decoded = l_vq_model.module.decode_to_img(tmp_past_index, btc)
            new_past, new_past_index = l_vq_model.module.get_quant(
                tmp_decoded[:, :seq_len, :])
            listener_past_index = torch.reshape(new_past_index,
                                                (new_past.shape[0], -1))

        ## dealing with future listener motion (during training only)
        listener_future = None
        listener_future_index = None
        if listenerData.shape[1] > seq_len:
            listener_future, listener_future_index = \
                l_vq_model.module.get_quant(listenerData[:, seq_len:, :])
            listener_future_index = torch.reshape(listener_future_index,
                                                  (listener_future.shape[0], -1))

    ## build input dictionary, which will be the input to the Predictor
    raw_listener = listenerData[:, seq_len:, :] if listenerData.dim() == 3 \
        else None
    inputs = {"speaker_full": speaker_full,
              "listener_past": listener_past_index,
              "audio_full": audio_full,
              "personality": personalityData}
    # print("--------------------------------------------------------------")
    # print(speaker_full.shape)
    # print(speaker_full)
    # print(audio_full)
    # print(listener_past_index)
    # print(listener_future_index)
    # print(audio_full.shape)
    # print("--------------------------------------------------------------")
    return inputs, listener_future_index, raw_listener, btc


def load_test_data(config, pipeline, tag, out_num=0, vqconfigs=None,
                   smooth=False, speaker=None, eval_mode = '', segment_tag='', num_out=None):
    """ function to load test data from files

    Parameters
    ----------
    pipeline : str
        defines the type of data to be loaded 'er', (e: expression, r: rotation)
    tag: str
        specifies the file with the tag suffix to load from
    out_num: str
        specifies which postion the listener is in the video (left:0, right:1)
        used for definining prefix in file name
    vqconfigs: dict
        specifies the vqconfigs corresponding to the pretrained VQ-VAE
        used to load the std/mean info for listeners
    smooth: bool
        whether to use bilateral filtering to smooth loaded files
    speaker: str
        specifies the speaker name for whom we want to load data
    segment_tag: str
        another one of these prefix tags (not really used for public release)
    num_out: int
        used to specify how many segments to load (for debugging)
    """
    base_dir = config['data']['basedir']
    test_data = pd.read_csv(f'{base_dir}/val/metadata/parts_val_unmasked.csv')
    # load all speaker information from files
    test_dir = os.path.join(base_dir, "val", "smoothed_final_features_id")
    all_test_speakers = next(os.walk(test_dir))[1]
    all_test_speakers.sort(key=lambda x: int(x))
    all_test_speakers = [str(speaker) for speaker in all_test_speakers]
    test_ids = test_data['ID'].astype(str).tolist()
    all_test_speakers = [speaker for speaker in all_test_speakers if speaker in test_ids]
    all_test_speakers = all_test_speakers if speaker == 'all' else [speaker]
    # Check the user's response and take action accordingly
    if eval_mode == 'c':
        # Code to handle the conditioned evaluation
        print("You have selected conditioned evaluation using personality data.")
        # Prompt for user input
        extraversion_input = input("To generate a listener's motion sequence, please specify their personality type:\n"
                                "- Press 'i' for an introverted listener.\n"
                                "- Press 'e' for an extroverted listener.\n"
                                "- Press 'a' for an actual listener values.\n"
                                "- Press 'r' for an random listener values.\n"
                                "- Press 'Enter' directly for a neutral personality.\n"
                                "Choose an option: ").strip().lower()

        # Determine train_extraversion_z_value based on input
        if extraversion_input == 'i':
            test_extraversion_z_value = -8.0  # Introvert case
            test_extraversion_z_value = np.full(len(test_data), test_extraversion_z_value).tolist()
        elif extraversion_input == 'e':
            test_extraversion_z_value = 8.0   # Extrovert case
            test_extraversion_z_value = np.full(len(test_data), test_extraversion_z_value).tolist()
        elif extraversion_input == 'a':  # List of actual extraversion values 
            test_extraversion_z_value = test_data['EXTRAVERSION_Z'].tolist()
        elif extraversion_input == 'r':   # List of random extraversion values sampled from gaussian dist.
            # Generate a random list by randomly shuffling listener extraversion scores
            # test_extraversion_z_value = random.shuffle(test_data['EXTRAVERSION_Z'].tolist())
            # Convert dataframe column to a list
            test_extraversion_z_value = test_data['EXTRAVERSION_Z'].tolist()

            # Shuffle the list in place
            random.shuffle(test_extraversion_z_value)

        else:
            test_extraversion_z_value = 0.0   # Neutral case (either 'neutral' or no input)
            test_extraversion_z_value = np.full(len(test_data), test_extraversion_z_value).tolist()

        test_X = np.empty((0, 64, 359))
        test_Y = np.empty((0, 64, 359))
        test_audio = np.empty((0, 256, 128))
        listeners_test_extraversion_z = np.empty((0, 1))
        for i, speaker in enumerate(all_test_speakers):
            speaker_index = test_ids.index(speaker)
            p1_fp = '{}/val/smoothed_final_features_id/{}/p{}_speak_all_body_pix{}.npy' \
                .format(base_dir, speaker, 1, segment_tag)
            p0_fp = '{}/val/smoothed_final_features_id/{}/p{}_list_all_body_pix{}.npy' \
                .format(base_dir, speaker, 0, segment_tag)
            audio_fp = '{}/val/smoothed_final_features_id/{}/p{}_speak_audio_mfcc.npy' \
                .format(base_dir, speaker, 1, segment_tag)
            # tmp_filepaths = np.load(fp)
            p1 = np.load(p1_fp)
            test_tmp_X = p1.astype(np.float32)[:, :, :]
            test_tmp_Y = np.load(p0_fp).astype(np.float32)[:, :, :]
            test_tmp_audio = np.load(audio_fp).astype(np.float32)
            test_extraversion_z = np.full((test_tmp_Y.shape[0], 1), test_extraversion_z_value[speaker_index])

            # filepaths = np.concatenate((filepaths, tmp_filepaths), axis=0)
            test_X = np.concatenate((test_X, test_tmp_X), axis=0)
            test_Y = np.concatenate((test_Y, test_tmp_Y), axis=0)
            test_audio = np.concatenate((test_audio, test_tmp_audio), axis=0)
            listeners_test_extraversion_z = np.concatenate((listeners_test_extraversion_z, test_extraversion_z), axis=0)
        
        # optional post processing steps on data
        if num_out is not None:
            # filepaths = filepaths[:num_out, :, :]
            test_X = test_X[:num_out, :, :]
            test_Y = test_Y[:num_out, :, :]
            test_audio = test_audio[:num_out, :, :]
            listeners_test_extraversion_z = listeners_test_extraversion_z[:num_out, :]
        if smooth:
            test_X = bilateral_filter(test_X)
            test_Y = bilateral_filter(test_Y)

    elif eval_mode == 'u':
        # Code to handle the unconditioned evaluation
        print("You have selected unconditioned evaluation without using personality data.")
        test_X = np.empty((0, 64, 359))
        test_Y = np.empty((0, 64, 359))
        test_audio = np.empty((0, 256, 128))
        listeners_test_extraversion_z = None
        for i, speaker in enumerate(all_test_speakers):
            speaker_index = test_ids.index(speaker)
            p0_fp = '{}/val/smoothed_final_features_id/{}/p{}_speak_all_body_pix{}.npy' \
                .format(base_dir, speaker, 1, segment_tag)
            p1_fp = '{}/val/smoothed_final_features_id/{}/p{}_list_all_body_pix{}.npy' \
                .format(base_dir, speaker, 0, segment_tag)
            audio_fp = '{}/val/smoothed_final_features_id/{}/p{}_speak_audio_mfcc.npy' \
                .format(base_dir, speaker, 1, segment_tag)
            # tmp_filepaths = np.load(fp)
            p1 = np.load(p1_fp)
            test_tmp_X = p1.astype(np.float32)[:, :, :]
            test_tmp_Y = np.load(p1_fp).astype(np.float32)[:, :, :]
            test_tmp_audio = np.load(audio_fp).astype(np.float32)
            # filepaths = np.concatenate((filepaths, tmp_filepaths), axis=0)
            test_X = np.concatenate((test_X, test_tmp_X), axis=0)
            test_Y = np.concatenate((test_Y, test_tmp_Y), axis=0)
            test_audio = np.concatenate((test_audio, test_tmp_audio), axis=0)
        
        # optional post processing steps on data
        if num_out is not None:
            # filepaths = filepaths[:num_out, :, :]
            test_X = test_X[:num_out, :, :]
            test_Y = test_Y[:num_out, :, :]
            test_audio = test_audio[:num_out, :, :]
        if smooth:
            test_X = bilateral_filter(test_X)
            test_Y = bilateral_filter(test_Y)

        
    else:
        # Code to handle invalid input or provide a default action
        print("Invalid input. Please select either 'c' for conditioned or 'u' for unconditioned.")
    

    # standardize dataset
    preprocess = np.load(os.path.join(config['model_path'],
                                      '{}{}_preprocess_core.npz'.format(tag, pipeline)))
    body_mean_X = preprocess['body_mean_X']
    body_std_X = preprocess['body_std_X']
    body_mean_audio = preprocess['body_mean_audio']
    body_std_audio = preprocess['body_std_audio']
    # take the std/mean from the listener vqgan training
    y_preprocess = np.load(os.path.join('vqgan/',
                                        vqconfigs['l_vqconfig']['model_path'], '{}{}_preprocess_core.npz' \
                                        .format(vqconfigs['l_vqconfig']['tag'], pipeline)))
    body_mean_Y = y_preprocess['body_mean_Y']
    body_std_Y = y_preprocess['body_std_Y']
    std_info = {'body_mean_X': body_mean_X,
                'body_std_X': body_std_X,
                'body_mean_Y': body_mean_Y,
                'body_std_Y': body_std_Y}
    test_X = (test_X - body_mean_X) / body_std_X
    test_Y = (test_Y - body_mean_Y) / body_std_Y
    test_audio = (test_audio - body_mean_audio) / body_std_audio
    return test_X, test_Y, test_audio, listeners_test_extraversion_z ,std_info


def load_data(config, pipeline, tag, rng, vqconfigs=None, segment_tag='',
              smooth=False):
    """ function to load train data from files

    see load_test_data() for associated parameters
    """

    base_dir = config['data']['basedir']
    train_dir = os.path.join(base_dir, "train", "smoothed_final_features_id")
    val_dir = os.path.join(base_dir, "val", "smoothed_final_features_id")
    out_num = 0
    train_gt_windows = np.empty((0, 64, 359))
    train_quant_windows = np.empty((0, 64, 359))
    train_audio_windows = np.empty((0, 256, 128))
    train_pers_features = np.empty((0, 1))

    val_gt_windows = np.empty((0, 64, 359))
    val_quant_windows = np.empty((0, 64, 359))
    val_audio_windows = np.empty((0, 256, 128))
    val_pers_features = np.empty((0, 1))

    if config['data']['speaker'] == 'all':
        # load associated files for all speakers
        all_train_speakers = next(os.walk(train_dir))[1]
        all_val_speakers = next(os.walk(val_dir))[1]
        all_train_speakers.sort(key=lambda x: int(x))
        all_val_speakers.sort(key=lambda x: int(x))
        # all_train_speakers = ['2','3']
        # curr_paths = train_gt_windows = train_quant_windows = train_audio_windows = None
        print("loading training data....")
        for speaker in all_train_speakers:
            train_tmp_gt, train_tmp_quant, train_tmp_audio, train_extraversion_z = get_local_train_files(base_dir, speaker, out_num, segment_tag)
            train_gt_windows = np.concatenate((train_gt_windows, train_tmp_gt), axis=0)
            train_quant_windows = np.concatenate((train_quant_windows, train_tmp_quant),
                                           axis=0)
            train_audio_windows = np.concatenate((train_audio_windows, train_tmp_audio),
                                           axis=0)
            train_pers_features = np.concatenate((train_pers_features, train_extraversion_z),
                                           axis=0)
            print('curr:', train_tmp_gt.shape, train_tmp_quant.shape, train_tmp_audio.shape, train_extraversion_z.shape)
        
        print("loading validation data....")
        for speaker in all_val_speakers:
            val_tmp_gt, val_tmp_quant, val_tmp_audio, val_extraversion_z = get_local_val_files(base_dir, speaker, out_num, segment_tag)
            val_gt_windows = np.concatenate((val_gt_windows, val_tmp_gt), axis=0)
            val_quant_windows = np.concatenate((val_quant_windows, val_tmp_quant),
                                           axis=0)
            val_audio_windows = np.concatenate((val_audio_windows, val_tmp_audio),
                                           axis=0)
            val_pers_features = np.concatenate((val_pers_features, val_extraversion_z),
                                           axis=0)
            print('curr:', val_tmp_gt.shape, val_tmp_quant.shape, val_tmp_audio.shape, val_extraversion_z.shape)
    else:
        # load specific training files for specified training speaker
        train_gt_windows, train_quant_windows, train_audio_windows, train_pers_features = get_local_train_files(base_dir, config['data']['speaker'], out_num,
                                                                   segment_tag)

        # load specific validation files for specified validation speaker
        val_gt_windows, val_quant_windows, val_audio_windows, val_pers_features = get_local_val_files(base_dir, config['data']['speaker'], out_num,
                                                                   segment_tag)
    print('===> training in/out',
          train_gt_windows.shape, train_quant_windows.shape, train_audio_windows.shape, train_pers_features.shape)
    print('===> validation in/out',
          val_gt_windows.shape, val_quant_windows.shape, val_audio_windows.shape, val_pers_features.shape)
    # Pre-processing of loaded data
    if smooth:
        train_gt_windows = bilateral_filter(train_gt_windows)
        train_quant_windows = bilateral_filter(train_quant_windows)
        val_gt_windows = bilateral_filter(val_gt_windows)
        val_quant_windows = bilateral_filter(val_quant_windows)
    # randomize train/test splits
    # N = train_gt_windows.shape[0]
    train_N = train_audio_windows.shape[0]
    val_N = val_audio_windows.shape[0]
    train_idx = np.random.permutation(train_N)
    val_idx = np.random.permutation(val_N)

    train_X, val_X = train_gt_windows[train_idx, :, :].astype(np.float32), \
                      val_gt_windows[val_idx, :, :].astype(np.float32)
    train_Y, val_Y = train_quant_windows[train_idx, :, :].astype(np.float32), \
                      val_quant_windows[val_idx, :, :].astype(np.float32)
    train_audio, val_audio = train_audio_windows[train_idx, :, :].astype(np.float32), \
                              val_audio_windows[val_idx, :, :].astype(np.float32)
    train_meta, val_meta = train_pers_features[train_idx, :].astype(np.float32), \
                            val_pers_features[val_idx, :].astype(np.float32)
    
    # check to see how to load/calculate std/dev
    body_mean_X, body_std_X, body_mean_Y, body_std_Y, \
    body_mean_audio, body_std_audio = calc_stats(config, vqconfigs, tag,
                                                 pipeline, train_X, train_Y,
                                                 train_audio)
    train_X = (train_X - body_mean_X) / body_std_X
    val_X = (val_X - body_mean_X) / body_std_X
    train_Y = (train_Y - body_mean_Y) / body_std_Y
    val_Y = (val_Y - body_mean_Y) / body_std_Y
    train_audio = (train_audio - body_mean_audio) / body_std_audio
    val_audio = (val_audio - body_mean_audio) / body_std_audio

    print("=====> standardization done, personality traits are un-normalized")
    print("--------------------------------------------------------------------------------------------------------------------")
    print("====> speaker motion train/val", train_X.shape, val_X.shape)
    print("====> speaker audio train/val", train_audio.shape, val_audio.shape)
    print("====> listener motion train/val", train_Y.shape, val_Y.shape)
    print("====> personality train/val", train_meta.shape, val_meta.shape)
    print("--------------------------------------------------------------------------------------------------------------------")
    print("loading data done, now start training ------>")
    return train_X, val_X, train_Y, val_Y, train_audio, val_audio, train_meta, val_meta


def get_extraversion_z(personality_data, listener_id):
    if listener_id in personality_data['ID'].values:
        return personality_data[personality_data['ID'] == listener_id]['EXTRAVERSION_Z'].iloc[0]
    else:
        return "ID not found in the dataset"

def get_local_train_files(base_dir, speaker, out_num, segment_tag):
    """ helper function for loading associated files """

    p1_fp = '{}/train/smoothed_final_features_id/{}/p{}_speak_all_body_pix{}.npy' \
        .format(base_dir, speaker, 1, segment_tag)
    p0_fp = '{}/train/smoothed_final_features_id/{}/p{}_list_all_body_pix{}.npy' \
        .format(base_dir, speaker, 0, segment_tag)
    audio_fp = '{}/train/smoothed_final_features_id/{}/p{}_speak_audio_mfcc.npy' \
        .format(base_dir, speaker, 1, segment_tag)
    
    personality_data = pd.read_csv(f'{base_dir}/train/metadata/parts_train.csv')
    # Standardizing the 'EXTRAVERSION_Z' column
    scaler = StandardScaler()
    personality_data['EXTRAVERSION_Z'] = scaler.fit_transform(personality_data[['EXTRAVERSION_Z']])
    extraversion_z_value = get_extraversion_z(personality_data, int(speaker))
    
    train_gt_windows = np.load(p1_fp)
    train_quant_windows = np.load(p0_fp)
    train_audio_windows = np.load(audio_fp)
    train_extraversion_z = np.full((train_quant_windows.shape[0], 1), extraversion_z_value)

    print('loaded participant ID...', speaker)
    print(f"personality EXTRAVERSION_Z for ID {speaker}: {extraversion_z_value}")

    return train_gt_windows, train_quant_windows, train_audio_windows, train_extraversion_z

def get_local_val_files(base_dir, speaker, out_num, segment_tag):
    """ helper function for loading associated files """

    p1_fp = '{}/val/smoothed_final_features_id/{}/p{}_speak_all_body_pix{}.npy' \
        .format(base_dir, speaker, 1, segment_tag)
    p0_fp = '{}/val/smoothed_final_features_id/{}/p{}_list_all_body_pix{}.npy' \
        .format(base_dir, speaker, 0, segment_tag)
    audio_fp = '{}/val/smoothed_final_features_id/{}/p{}_speak_audio_mfcc.npy' \
        .format(base_dir, speaker, 1, segment_tag)
    
    personality_data = pd.read_csv(f'{base_dir}/val/metadata/parts_val_unmasked.csv')
    # Standardizing the 'EXTRAVERSION_Z' column
    scaler = StandardScaler()
    personality_data['EXTRAVERSION_Z'] = scaler.fit_transform(personality_data[['EXTRAVERSION_Z']])
    extraversion_z_value = get_extraversion_z(personality_data, int(speaker))
    
    val_gt_windows = np.load(p1_fp)
    val_quant_windows = np.load(p0_fp)
    val_audio_windows = np.load(audio_fp)
    val_extraversion_z = np.full((val_quant_windows.shape[0], 1), extraversion_z_value)

    print('loaded participant ID...', speaker)
    print(f"personality EXTRAVERSION_Z for ID {speaker}: {extraversion_z_value}")

    return val_gt_windows, val_quant_windows, val_audio_windows, val_extraversion_z

def calc_stats(config, vqconfigs, tag, pipeline, train_X, train_Y, train_audio):
    """ helper function to calculate std/mean for different cases """
    if vqconfigs is not None:
        # if vqconfig is defined, use std/mean from VQ-VAE for listener
        y_preprocess = np.load(os.path.join('vqgan/',
                                            vqconfigs['l_vqconfig']['model_path'], '{}{}_preprocess_core.npz' \
                                            .format(vqconfigs['l_vqconfig']['tag'], pipeline)))
        body_mean_Y = y_preprocess['body_mean_Y']
        body_std_Y = y_preprocess['body_std_Y']
        # then calculate std/mean for speaker motion + audio
        body_mean_X, body_std_X = mean_std_swap(train_X)
        body_mean_audio, body_std_audio = mean_std_swap(train_audio)
        np.savez_compressed(config['model_path'] + \
                            '{}{}_preprocess_core.npz'.format(tag, pipeline),
                            body_mean_X=body_mean_X, body_std_X=body_std_X,
                            body_mean_audio=body_mean_audio, body_std_audio=body_std_audio)
    else:
        # if vqconfig not defined, no prior mean/std info exists
        body_mean_X, body_std_X = mean_std_swap(train_X)
        body_mean_Y, body_std_Y = mean_std_swap(train_Y)
        body_mean_audio, body_std_audio = mean_std_swap(train_audio)
        assert body_mean_X.shape[0] == 1 and body_mean_X.shape[1] == 1
        np.savez_compressed(config['model_path'] + \
                            '{}{}_preprocess_core.npz'.format(tag, pipeline),
                            body_mean_X=body_mean_X, body_std_X=body_std_X,
                            body_mean_Y=body_mean_Y, body_std_Y=body_std_Y,
                            body_mean_audio=body_mean_audio, body_std_audio=body_std_audio)
    return body_mean_X, body_std_X, body_mean_Y, body_std_Y, \
           body_mean_audio, body_std_audio


def mean_std_swap(data):
    """ helper function to calc std and mean """
    B, T, F = data.shape
    mean = data.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]
    std = data.std(axis=1).std(axis=0)[np.newaxis, np.newaxis, :]
    std += EPSILON
    return mean, std
