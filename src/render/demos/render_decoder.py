import os, sys
import pdb
import numpy as np
import torch.backends.cudnn as cudnn
import torch
from tqdm import tqdm
import argparse
import cv2
import csv
import imageio
import pickle
import random
import pdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from pixielib.pixie import PIXIE
from pixielib.visualizer import Visualizer
from scipy.interpolate import Akima1DInterpolator
from pixielib.datasets.body_datasets import TestData
from moviepy.editor import ImageSequenceClip, AudioFileClip
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg


def apply_median_filter_at_intervals(predictions, interval=32, window=11, device='cuda:0'):
    if len(predictions) < window:
        return predictions

    smoothed_predictions = predictions.copy()
    half_window = window // 2
    dtype = predictions[0][list(predictions[0].keys())[0]].dtype  # Get the data type from the first element

    # Iterate over the predictions with a step size of 'interval'
    for i in range(0, len(predictions), interval):
        for param in predictions[0].keys():
            # Define the window range
            start_index = max(i - half_window, 0)
            end_index = min(i + half_window + 1, len(predictions))
            
            # Extract the window values for the current parameter
            window_values = [predictions[j][param] for j in range(start_index, end_index)]
            
            # Calculate the median and apply it to the central frame of the window
            if window_values:
                # Ensure all tensors are on CPU before converting to NumPy array
                array_values = np.array([value.cpu().numpy() if isinstance(value, torch.Tensor) else value for value in window_values])
                median_value = np.median(array_values, axis=0)
                
                # Convert median_value back to tensor if necessary and move to the original device
                if isinstance(predictions[0][param], torch.Tensor):
                    median_value = torch.tensor(median_value, dtype=dtype, device=device)
                
                central_index = min(max(i, half_window), len(predictions) - half_window - 1)
                smoothed_predictions[central_index][param] = median_value

    return smoothed_predictions

def load_speaker_audio(input_audio_dir,sessions_dir,parts_dir,id):
    participants_dict = {}
    with open(parts_dir, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            participant_id = row['ID']
            sessions = []
            for i in range(1, 2):
                session_id = row[f'SESSION{i}']
                if session_id:
                    sessions.append(session_id)
            participants_dict[participant_id] = sessions
    # print(len(participants_dict))

    sessions_dict = {}

    with open(sessions_dir, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            session_id = row['ID']
            participant_ids = [row['PART.1'], row['PART.2']]
            sessions_dict[session_id] = participant_ids

    for participant_id in participants_dict:
        if participant_id == id: 
            sessions = participants_dict[participant_id]
            for session_id in sessions:
                participant_ids_list = sessions_dict[str(session_id).zfill(6)]
                if participant_id in participant_ids_list:
                    index = participant_ids_list.index(participant_id)
                    if index == 0:
                        p0 = "part1"
                        p1 = "part2"
                    elif index == 1:
                        p0 = "part2"
                        p1 = "part1"
                    else:
                        continue  # invalid index, skip to next iteration
                    session_dir = os.path.join(input_audio_dir, str(session_id).zfill(6))
                    audio_file_path = os.path.join(session_dir,f"{p1}_{'_'.join(['speak_all_body_pixie.mp3'])}")
                    return audio_file_path
    

def interpolate_sequences(predictions, frame_step=32, window_size=12, device='cuda:0'):
    interpolated_predictions = []
    n = len(predictions)
    dtype = predictions[0][list(predictions[0].keys())[0]].dtype  # Get the data type from the first element

    for i in range(0, n, frame_step):
        # Add the current set of frames to the interpolated predictions
        interpolated_predictions.extend(predictions[i:min(i + frame_step, n)])

        # Check if there is a next set of frames to interpolate with
        if i + frame_step < n:
            last_frame_index = i + frame_step - 1
            next_frame_index = i + frame_step

            # Define interpolation points
            interpolation_points = [
                last_frame_index + (next_frame_index - last_frame_index) // 3,
                last_frame_index + 2 * (next_frame_index - last_frame_index) // 3
            ]

            for interpolated_index in interpolation_points:
                # Initialize an empty dictionary for each interpolated frame
                interpolated_seq = {}

                for param in predictions[i].keys():
                    # Prepare data for Akima interpolation
                    x = []
                    y = []

                    # Define the window range for interpolation
                    window_start = max(0, interpolated_index - window_size)
                    window_end = min(n, interpolated_index + window_size + 1)

                    for frame_index in range(window_start, window_end):
                        x.append(frame_index)
                        y.append(predictions[frame_index][param].cpu().numpy())

                    # Perform Akima interpolation
                    if len(x) > 1:  # Akima requires at least two data points
                        akima_interp = Akima1DInterpolator(x, y)
                        interpolated_value = akima_interp(interpolated_index)
                    else:
                        # Fallback to the original value if not enough points for interpolation
                        interpolated_value = predictions[interpolated_index][param].cpu().numpy()

                    # Convert the interpolated value back to tensor and move to device, while preserving the original data type
                    interpolated_seq[param] = torch.tensor(interpolated_value, dtype=dtype, device=device)

                interpolated_predictions.append(interpolated_seq)

    return interpolated_predictions


# Function to draws a filled rectangle behind frame's title
def draw_label(img, text, position, font, font_scale, text_color, bg_color, thickness):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    x, y = position
    # Adjust the x-coordinate for center alignment
    x -= text_width // 2
    # Create the rectangle background
    cv2.rectangle(img, (x, y - text_height - 7), (x + text_width, y + 7), bg_color, cv2.FILLED)
    # Put the text on top of the rectangle
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def smooth_predictions(predictions, window = 7):
    smoothed = []
    for i in range(len(predictions)):
        if window <= i < len(predictions) - window:  # for middle frames with sufficient surrounding frames
            smoothed_codedict = {}
            # Identify the first parameter (key)
            first_param = next(iter(predictions[i]))
            for param in predictions[i].keys():
                if param == first_param:
                    # Copy the first parameter as is, without smoothing
                    smoothed_codedict[param] = predictions[i][param]
                else:
                    # Smooth other parameters
                    sum_frames = sum(predictions[j][param] for j in range(i - window, i + window + 1))
                    smoothed_codedict[param] = sum_frames / (2 * window + 1)
            smoothed.append(smoothed_codedict)
        else:  # for first and last frames where the window is not full
            smoothed.append(predictions[i])
    return smoothed


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
    
    # predicting listener participants ids
    # ids = [38,39,52,81,145,180,183]

    info_output_dir = "/home/pipeline/outputs/all_udiva/unconditioned/"

    # Initialize a list to keep track of the positions
    position_tracker = []  
    
    # ids = ['1','38','39','52','182','183','186','190']
    # speaker_list = ['1','38','39','52','57','80','81','85','119','124','141','145','146','147','180','181']
    speaker_list = ['181']
    # ids = ['1','38','39']
    for id in speaker_list:
        # Format each directory with the current ID
        input_list_dir = "/home/UDIVAv0.5/extracted_features/val/frames_mirror_id/{}/p0_list_all_body_pixie/".format(id)
        input_speak_dir = "/home/UDIVAv0.5/extracted_features/val/frames_mirror_id/{}/p1_speak_all_body_pixie/".format(id)
        input_audio_dir = "/home/UDIVAv0.5/extracted_features/val/video_crop/"
        sessions_dir = "/home/UDIVAv0.5/val/metadata/sessions_val.csv"
        parts_dir = "/home/UDIVAv0.5/val/metadata/parts_val_unmasked.csv"
        first_pose = "/home/UDIVAv0.5/extracted_features/val/smoothed_encode_pixie_id/{}/p0_list_all_body_pixi/".format(id)
        # extrovert_dir = "/home/pipeline/outputs/all_udiva/conditioned/{}/_extrovert/results/delta_v6_predicted/".format(id)
        # neutral_dir = "/home/pipeline/outputs/conditioned_perso/{}/_neutral/results/delta_v6_predicted/".format(id)
        #introvert_dir = "/home/pipeline/outputs/all_udiva/conditioned/{}/_introvert/results/delta_v6_predicted/".format(id)
        actual_cond_dir = "/home/pipeline/outputs/all_udiva/conditioned/{}/_actual/results/delta_v6_predicted/".format(id)
        uncond_dir = "/home/pipeline/outputs/all_udiva/unconditioned/{}/results/delta_v6_predicted/".format(id)
        all_output_dir = "/home/pipeline/outputs/all_udiva/unconditioned/{}/".format(id)

        # load speaker
        speaker_audio_file = load_speaker_audio(input_audio_dir,sessions_dir,parts_dir,id)
        # speaker_audio = AudioSegment.from_file(speaker_audio_file)
        audio_clip = AudioFileClip(speaker_audio_file)
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
        extrovert_list = os.listdir(actual_cond_dir)
        # neutral_list = os.listdir(neutral_dir)
        introvert_list = os.listdir(uncond_dir)

        extrovert_pkl_files = [os.path.join(actual_cond_dir, f) for f in extrovert_list if f.endswith('.pkl')]
        # neutral_pkl_files = [os.path.join(neutral_dir, f) for f in neutral_list if f.endswith('.pkl')]
        introvert_pkl_files = [os.path.join(uncond_dir, f) for f in introvert_list if f.endswith('.pkl')]


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
        extrovert_predictions = process_pkl_files(extrovert_pkl_files[:500], input_codedict)
        # neutral_predictions = process_pkl_files(neutral_pkl_files[:720], input_codedict)
        introvert_predictions = process_pkl_files(introvert_pkl_files[:500], input_codedict)
                    
        ## 3. Apply smoothing to predictions
        extrovert_predictions = smooth_predictions(extrovert_predictions)
        # neutral_predictions = smooth_predictions(neutral_predictions)
        introvert_predictions = smooth_predictions(introvert_predictions)

        extrovert_predictions = apply_median_filter_at_intervals(extrovert_predictions)
        introvert_predictions = apply_median_filter_at_intervals(introvert_predictions)

        ## 5. Apply interpolation between smoothed predictions
        extrovert_predictions = interpolate_sequences(extrovert_predictions)
        # neutral_predictions = interpolate_sequences(neutral_predictions)
        introvert_predictions = interpolate_sequences(introvert_predictions)

        # Randomly decide the position for the entire video
        conditioned_on_left = random.choice([True, False])
        position_info = {
            'video_name': f"listener_{id}_vs_speaker_models_comp.mp4",
            'left': 'Conditioned' if conditioned_on_left else 'Unconditioned',
            'right': 'Unconditioned' if conditioned_on_left else 'Conditioned'
        }

        # Add the position info to the tracker
        position_tracker.append(position_info)


        # List of speaker frames
        speaker_frames_list = sorted([os.path.join(input_speak_dir, f) for f in os.listdir(input_speak_dir) if f.endswith('.png') or f.endswith('.jpg')])
        
        # Skipping the first 32 frames
        # original_frames_list = original_frames_list[32:]
        # speaker_frames_list = speaker_frames_list[:500]

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        thickness = 1
        
        frames = []
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

            # Add titles to each frame
            # draw_label(original_frame, 'Listener - original', (256, 30), font, font_scale, text_color, bg_color, thickness)
            draw_label(speaker_frame, 'Speaker', (256, 30), font, font_scale, text_color, bg_color, thickness)

            grid_image_extrovert= np.ascontiguousarray(grid_image_extrovert[:,:,[2,1,0]]).copy()
            # grid_image_neutral= np.ascontiguousarray(grid_image_neutral[:,:,[2,1,0]]).copy()
            grid_image_introvert= np.ascontiguousarray(grid_image_introvert[:,:,[2,1,0]]).copy()
  
            
            
            # Use the decided positions for this video
            if conditioned_on_left:
                cv2.putText(grid_image_introvert, 'Listener Avatar "1"', (200, 30), font, font_scale, text_color, thickness, cv2.LINE_AA)
                cv2.putText(grid_image_extrovert, 'Listener Avatar "2"', (200, 30), font, font_scale, text_color, thickness, cv2.LINE_AA)
                conversation_image = np.concatenate((grid_image_introvert, grid_image_extrovert, speaker_frame), axis=1)
            else:
                cv2.putText(grid_image_extrovert, 'Listener Avatar "1"', (200, 30), font, font_scale, text_color, thickness, cv2.LINE_AA)
                cv2.putText(grid_image_introvert, 'Listener Avatar "2"', (200, 30), font, font_scale, text_color, thickness, cv2.LINE_AA)
                conversation_image = np.concatenate((grid_image_extrovert, grid_image_introvert, speaker_frame), axis=1)
            
            
            # Add the numpy array of the frame to the list
            frames.append(conversation_image)
            # Visualize and save the concatenated image
            # writer_speaker_pred.append_data(conversation_image)
            # writer_speaker_pred.append_data(conversation_image, {'fps': 25, 'audio': speaker_audio})

        
        # Create a video clip from the frames (numpy arrays)
        video_clip = ImageSequenceClip(frames, fps=25)
        # Set the audio of the video clip as your audio clip
        video_clip = video_clip.set_audio(audio_clip)
        
        # Cut the video from the start (0 seconds) to 20 seconds
        video_clip = video_clip.subclip(0, 20)
        
        # Write the result to a file
        output_video_path = os.path.join(all_output_dir, f"listener_{id}_vs_speaker_models_comp.mp4")
        video_clip.write_videofile(output_video_path, codec="libx264", audio_codec='aac')  
        
        print(f'-- please check the results in {all_output_dir}')    
    
    # Save the position information to a CSV file
    positions_csv_path = os.path.join(info_output_dir, "positions_cond_uncond_info.csv")
    with open(positions_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['video_name', 'left', 'right']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for position_info in position_tracker:
            writer.writerow(position_info)
            
    # Close the video writer
    # writer_comparison.close()
    # writer_speaker_pred.close()
    # writer_pred.close()
    
        


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
