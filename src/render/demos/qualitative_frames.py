import os, sys
import pdb
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import csv
import pickle
import imageio
import random
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from scipy.interpolate import Akima1DInterpolator
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg


def process_pkl_files(pkl_files, input_codedict):
    predictions = []
    for feature_file in tqdm(pkl_files):
        with open(feature_file, 'rb') as f:
            codedict = pickle.load(f)
        # Transfer parameters
        for param in ['shape', 'tex', 'body_cam', 'light', 'global_pose']:
            codedict[param] = input_codedict[param]
        # Convert torch tensors values to float
        for key in codedict.keys():
            codedict[key] = torch.tensor(codedict[key]).to("cuda").float()

        predictions.append(codedict)
    return predictions


def main(args):
    # cuda device 'cuda:0' by default 
    device = args.device

    # check env
    if not torch.cuda.is_available():
        print('CUDA is not available! use CPU instead')
    else:
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    #-- run PIXIE
    pixie_cfg.model.use_tex = args.useTex
    pixie = PIXIE(config = pixie_cfg, device=device)
    visualizer = Visualizer(render_size=args.render_size, config = pixie_cfg, device=device, rasterizer_type=args.rasterizer_type)

    # Directories for saving individual frames
    extrovert_frames_dir = "/home/pipeline/outputs/extrovert"
    introvert_frames_dir = "/home/pipeline/outputs/introvert"
    speaker_frames_dir = "/home/pipeline/outputs/speaker"

    # Ensure these directories exist
    os.makedirs(extrovert_frames_dir, exist_ok=True)
    os.makedirs(introvert_frames_dir, exist_ok=True)
    os.makedirs(speaker_frames_dir, exist_ok=True)    
    
    # speaker_list = ['1','38','39','52','57','80','81','85','119','124','141','145','146','147','180','181']
    speaker_list = ['181']
    # ids = ['1','38','39']
    for id in speaker_list:
        # Format each directory with the current ID
        input_list_dir = "/home/UDIVAv0.5/extracted_features/val/frames_mirror_id/{}/p0_list_all_body_pixie/".format(id)
        input_speak_dir = "/home/UDIVAv0.5/extracted_features/val/frames_mirror_id/{}/p1_speak_all_body_pixie/".format(id)
        sessions_dir = "/home/UDIVAv0.5/val/metadata/sessions_val.csv"
        parts_dir = "/home/UDIVAv0.5/val/metadata/parts_val_unmasked.csv"
        first_pose = "/home/UDIVAv0.5/extracted_features/val/smoothed_encode_pixie_id/{}/p0_list_all_body_pixi/".format(id)
        extrovert_dir = "/home/pipeline/outputs/all_udiva/conditioned/{}/_extrovert/results/delta_v6_predicted/".format(id)
        introvert_dir = "/home/pipeline/outputs/all_udiva/conditioned/{}/_introvert/results/delta_v6_predicted/".format(id)

        # load test images 
        testdata = TestData(input_list_dir, iscrop=False, body_detector='rcnn', device=device)

        ## 1. fit smplx model of first image in sequence
        batch = testdata[0]
        util.move_dict_to_device(batch, device)
        batch['image'] = batch['image'].unsqueeze(0)
        batch['image_hd'] = batch['image_hd'].unsqueeze(0)
        name = batch['name']
        input_image = batch['image']
        data = {
            'body': batch
        }
        param_dict = pixie.encode(data)
        input_codedict = param_dict['body'] 

        # vis smplx results
        input_optdict = pixie.decode(input_codedict, param_type='body')
        input_optdict['albedo'] = visualizer.tex_flame2smplx(input_optdict['albedo'])
        visdict_extrovert = visualizer.render_results(input_optdict, data['body']['image_hd'], overlay=False)
        input_image = batch['image_hd'].clone()
        input_shape = visdict_extrovert['shape_images'].clone()    
            
        
        ## 2. transfer/fix some parameters to all predictions
        predictions = []
            
        # load smpl-x predictions of the model output as .pkl files
        extrovert_list = os.listdir(extrovert_dir)
        introvert_list = os.listdir(introvert_dir)

        extrovert_pkl_files = [os.path.join(extrovert_dir, f) for f in extrovert_list if f.endswith('.pkl')]
        # neutral_pkl_files = [os.path.join(neutral_dir, f) for f in neutral_list if f.endswith('.pkl')]
        introvert_pkl_files = [os.path.join(introvert_dir, f) for f in introvert_list if f.endswith('.pkl')]


        extrovert_pkl_files = sorted(extrovert_pkl_files)
        # neutral_pkl_files = sorted(neutral_pkl_files)
        introvert_pkl_files = sorted(introvert_pkl_files)

        pose_list = os.listdir(first_pose)
        pose_files = [os.path.join(first_pose, f) for f in pose_list if f.endswith('.pkl')]
        pose_files = sorted(pose_files)

        with open(pose_files[0], 'rb') as f:
            input_codedict = pickle.load(f)

        # Set shape parameters to zero to get neutral gender
        input_codedict['shape'][:, :] = 0
        
  
        # Process each personality type
        extrovert_predictions = process_pkl_files(extrovert_pkl_files[50:250], input_codedict)
        # neutral_predictions = process_pkl_files(neutral_pkl_files[:720], input_codedict)
        introvert_predictions = process_pkl_files(introvert_pkl_files[50:250], input_codedict)
                    

        # List of speaker frames
        speaker_frames_list = sorted([os.path.join(input_speak_dir, f) for f in os.listdir(input_speak_dir) if f.endswith('.png') or f.endswith('.jpg')])
        
        speaker_frames_list = speaker_frames_list[50:250]

        # Loop through both the smoothed predictions and the original frames
        for i, (predict_exrovert, predict_introvert, speaker_frame_path) in enumerate(zip(tqdm(extrovert_predictions), introvert_predictions, speaker_frames_list)):
            if i >= len(speaker_frames_list):
                break  # Stop if there are no more predictions
            
            # Decode the current predictions of each personality type to get the rendered image
            optdict_extrovert = pixie.decode(predict_exrovert, param_type='body')
            # optdict_neutral = pixie.decode(predict_neutral, param_type='body')
            optdict_introvert = pixie.decode(predict_introvert, param_type='body')

            optdict_extrovert['albedo'] = input_optdict['albedo']
            # optdict_neutral['albedo'] = input_optdict['albedo']
            optdict_introvert['albedo'] = input_optdict['albedo']

            visdict_extrovert = visualizer.render_results(optdict_extrovert, input_image)
            # visdict_neutral = visualizer.render_results(optdict_neutral, input_image)
            visdict_introvert = visualizer.render_results(optdict_introvert, input_image)

            transfered_shape_extrovert = visdict_extrovert['shape_images'].clone()
            # transfered_shape_neutral = visdict_neutral['shape_images'].clone()
            transfered_shape_introvert = visdict_introvert['shape_images'].clone()

            # Read the corresponding original frame
            # original_frame = imageio.imread(listener_frame_path)
            speaker_frame = imageio.imread(speaker_frame_path)

            # Resize the original & speaker frames to match the prediction's size
            # original_frame =  cv2.resize(original_frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            speaker_frame =  cv2.resize(speaker_frame, (512, 512), interpolation=cv2.INTER_LINEAR)

            # Visualize and save the transfer image alone
            visdict_extrovert_transfer = {'transfer': transfered_shape_extrovert}
            # visdict_neutral_transfer = {'transfer': transfered_shape_neutral}
            visdict_introvert_transfer = {'transfer': transfered_shape_introvert}

            grid_image_extrovert = visualizer.visualize_grid(visdict_extrovert_transfer, size=512)
            # grid_image_neutral = visualizer.visualize_grid(visdict_neutral_transfer, size=512)
            grid_image_introvert = visualizer.visualize_grid(visdict_introvert_transfer, size=512)
            # writer_pred.append_data(grid_image_extrovert[:,:,[2,1,0]])

            grid_image_extrovert= np.ascontiguousarray(grid_image_extrovert[:,:,[2,1,0]]).copy()
            # grid_image_neutral= np.ascontiguousarray(grid_image_neutral[:,:,[2,1,0]]).copy()
            grid_image_introvert= np.ascontiguousarray(grid_image_introvert[:,:,[2,1,0]]).copy()
  
            cv2.imwrite(os.path.join(extrovert_frames_dir, f"extrovert_frame_{i:04d}.png"), grid_image_extrovert)
            cv2.imwrite(os.path.join(introvert_frames_dir, f"introvert_frame_{i:04d}.png"), grid_image_introvert)
            cv2.imwrite(os.path.join(speaker_frames_dir, f"speaker_frame_{i:04d}.png"), speaker_frame)


                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PIXIE')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    # rendering option
    parser.add_argument('--render_size', default=1024, type=int,
                        help='image size of renderings' )
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--reproject_mesh', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to reproject the mesh and render it in original image size, \
                            currently only available if rasterizer_type is standard, because pytorch3d does not support non-squared image...\
                            default is False, means use the cropped image and its corresponding results')
    # save
    parser.add_argument('--deca_path', default=None, type=str,
                        help='absolute path of DECA folder, if exists, will return facial details by running DECA\
                        details of DECA: https://github.com/YadiraF/DECA' )
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--uvtex_type', default='SMPLX', type=str,
                        help='texture type to save, can be SMPLX or FLAME')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveGif', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to visualize other views of the output, save as gif' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveParam', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save parameters as pkl file' )
    parser.add_argument('--savePred', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save smplx prediction as pkl file' )
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())
