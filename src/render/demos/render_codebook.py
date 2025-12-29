import os, sys
import pdb
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import imageio
import pickle
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg
from scipy.spatial.distance import pdist, squareform

def calculate_diversity(predictions, feature_keys):
    # Extract the specified features and flatten them
    feature_matrix = torch.stack([torch.cat([predict[key].flatten() for key in feature_keys]) for predict in predictions])

    # Calculate pairwise Euclidean distances
    distance_matrix = squareform(pdist(feature_matrix.cpu(), 'euclidean'))

    # Sum distances for each feature to get a diversity score
    diversity_scores = np.sum(distance_matrix, axis=0)

    # Return the indices of the top 50 most diverse entries
    return np.argsort(diversity_scores)[-100:]

# Function to save a video from given indices
def save_video(predictions, indices, video_name, all_output_dir, pixie, input_opdict, input_image, visualizer):
    writer = imageio.get_writer(os.path.join(all_output_dir, video_name), fps=1)
    for idx in tqdm(indices):
        predict = predictions[idx]
        opdict = pixie.decode(predict, param_type='body')
        opdict['albedo'] = input_opdict['albedo']
        visdict = visualizer.render_results(opdict, input_image)
        transfered_shape = visdict['shape_images'].clone()
        visdict_transfer = {'transfer': transfered_shape}
        grid_image_transfer = visualizer.visualize_grid(visdict_transfer, size=512)
        writer.append_data(grid_image_transfer[:, :, [2, 1, 0]])
    writer.close()



def main(args):
    # cuda device 'cuda:0' by default 
    device = args.device
    # defining Input,predictions,and rendering video directories
    input_dir = "/home/UDIVAv0.5/extracted_features/val/frames_mirror_id/145/p0_list_all_body_pixie/"
    first_pose = "/home/UDIVAv0.5/extracted_features/val/smoothed_encode_pixie_id/52/p0_list_all_body_pixi/"
    transfer_output_dir = "/home/pipeline/outputs/codebook/codebook_complex_loss/"
    all_output_dir = "/home/pipeline/outputs/codebook/"
    
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


    # load test images 
    testdata = TestData(input_dir, iscrop=False, body_detector='rcnn', device=device)

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
    input_opdict = pixie.decode(input_codedict, param_type='body')
    input_opdict['albedo'] = visualizer.tex_flame2smplx(input_opdict['albedo'])
    visdict = visualizer.render_results(input_opdict, data['body']['image_hd'], overlay=False)
    input_image = batch['image_hd'].clone()
    input_shape = visdict['shape_images'].clone()    
        
    
    ## 2. transfer/fix some parameters to all predictions
    predictions_face = []
    predictions_body = []
        
    # load smpl-x predictions of the model output as .pkl files
    file_list = os.listdir(transfer_output_dir)
    pkl_files = [os.path.join(transfer_output_dir, f) for f in file_list if f.endswith('.pkl')]
    pkl_files = sorted(pkl_files)

    pose_list = os.listdir(first_pose)
    pose_files = [os.path.join(first_pose, f) for f in pose_list if f.endswith('.pkl')]
    pose_files = sorted(pose_files)

    with open(pose_files[0], 'rb') as f:
        input_codedict = pickle.load(f)

    # Set shape parameters to zero to get neutral gender
    input_codedict['shape'][:, :] = 0
    
    # Define feature subsets for each video
    features_face = ['exp', 'jaw_pose','shape', 'tex', 'body_cam', 'light','global_pose'] # features related to face expression
    features_body = ['neck_pose', 'head_pose','partbody_pose', 'left_wrist_pose', 'right_wrist_pose', 'left_hand_pose', 'right_hand_pose','shape', 'tex', 'body_cam', 'light','global_pose'] # features related to body poses
    
    for feature_file in tqdm(pkl_files):
        with open(feature_file, 'rb') as f:
            codedict = pickle.load(f)
        # Create separate copies for face and body
        codedict_face = codedict.copy()
        codedict_body = codedict.copy()
        # transfer parameters
        for param in features_body:
            codedict_face[param] = input_codedict[param]
        for param in features_face:
            codedict_body[param] = input_codedict[param]
        #Convert torch tensors values to float
        for key in codedict_body.keys():
            codedict_body[key] = torch.tensor(codedict_body[key]).to("cuda").float()
            codedict_face[key] = torch.tensor(codedict_face[key]).to("cuda").float()                
      
        predictions_face.append(codedict_face)
        predictions_body.append(codedict_body)

    # Calculate top 50 for each video
    top_50_video1 = calculate_diversity(predictions_face, features_face)
    top_50_video2 = calculate_diversity(predictions_body, features_body)

    # Save videos
    save_video(predictions_face, top_50_video1, "diversity_expressions_complex_loss.mp4", all_output_dir, pixie, input_opdict, input_image, visualizer)
    save_video(predictions_body, top_50_video2, "diversity_body_poses_complex_loss.mp4", all_output_dir, pixie, input_opdict, input_image, visualizer)


    print(f'-- please check the results in {all_output_dir}')
        


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
