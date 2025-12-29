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
from scipy.interpolate import Akima1DInterpolator
from pixielib.datasets.body_datasets import TestData
from pixielib.utils import util
from pixielib.utils.config import cfg as pixie_cfg

def calculate_optical_flow(prev_frame, current_frame):
    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            motion = np.sqrt((good_new[:,0] - good_old[:,0])**2 + (good_new[:,1] - good_old[:,1])**2)
        else:
            motion = np.array([0])
    except Exception as e:
        print(f"Error in calculate_optical_flow: {e}")
        motion = np.array([0])

    return motion

def detect_scene_changes(frame_paths, threshold = 6.0):
    """ Detects scene changes in a list of frames using optical flow. """
    prev_frame = cv2.imread(frame_paths[0])
    scene_changes = []

    for i, frame_path in enumerate(frame_paths[1:], 1):
        current_frame = cv2.imread(frame_path)
        motion = calculate_optical_flow(prev_frame, current_frame)
        # print(f"frame {i}:  {np.mean(motion)}")
        # Check if the average motion is above the threshold
        if np.mean(motion) > threshold:
            scene_changes.append(i)  # Store the index of the frame

        prev_frame = current_frame

    return scene_changes

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

def calculate_abs_differences(predictions):
    """
    Calculates the absolute differences in parameter values between consecutive frames.
    """
    jumps = []
    # skipped_params = ['body_cam', 'global_pose', 'shape', 'exp', 'jaw_pose', 'tex', 'light']
    skipped_params = ['body_cam', 'global_pose', 'shape','tex', 'light']
    for i in range(1, len(predictions)):
        total_diff_sum = 0.0
        for param in predictions[i].keys():
            if param not in skipped_params:
                # Calculate the sum of absolute differences for the parameter sequence
                total_diff = sum(abs(a - b) for a, b in zip(predictions[i][param], predictions[i-1][param]))
                # Sum up all elements in the tensor to get a single value
                total_diff_sum += total_diff.sum() if isinstance(total_diff, torch.Tensor) else total_diff
        # Convert the tensor to a standard Python number and print
        total_diff_value = total_diff_sum.cpu().item() if isinstance(total_diff_sum, torch.Tensor) else total_diff_sum
        # print(f"Frame {i} ~ {i+1}: {total_diff_value}")
        if total_diff_value > 6.8:
            jumps.append(i-1)
            # print(f"Frame {i} ~ {i+1}: {total_diff_value}")
    return jumps


def apply_median_filter_at_jumps(predictions, scene_changes, window=7):
    if not scene_changes:
        return predictions

    smoothed_predictions = predictions.copy()
    half_window = window // 2

    for i in scene_changes:
        for param in predictions[0].keys():
            start_index = max(i - half_window, 0)
            end_index = min(i + half_window + 1, len(predictions))
            window_values = [predictions[j][param] for j in range(start_index, end_index)]
            if window_values:
                array_values = np.array(window_values)
                median_value = np.median(array_values, axis=0)
                smoothed_predictions[i][param] = median_value

    return smoothed_predictions
    
def smooth_predictions(predictions, window = 6):
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

    ids = [1]
    for id in ids:
        # Format each directory with the current ID
        input_list_dir = "/home/UDIVAv0.5/extracted_features/val/frames_mirror_id/{}/p0_list_all_body_pixie/".format(id)
        data_dir = "/home/UDIVAv0.5/extracted_features/val/smoothed_encode_pixie_id/{}/p0_list_all_body_pixi/".format(id)
        all_output_dir = "/home/pipeline/outputs/{}/".format(id)

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
        visdict = visualizer.render_results(input_optdict, data['body']['image_hd'], overlay=False)
        input_image = batch['image_hd'].clone()
        input_shape = visdict['shape_images'].clone()    
            
        
        ## 2. transfer/fix some parameters to all predictions
        predictions = []
            
        # load smpl-x predictions of the model output as .pkl files
        data_list = os.listdir(data_dir)

        data_pkl_files = [os.path.join(data_dir, f) for f in data_list if f.endswith('.pkl')]
        
        data_pkl_files = sorted(data_pkl_files)

        pose_list = os.listdir(data_dir)
        pose_files = [os.path.join(data_dir, f) for f in pose_list if f.endswith('.pkl')]
        pose_files = sorted(pose_files)

        with open(pose_files[0], 'rb') as f:
            input_codedict = pickle.load(f)

        # Set shape parameters to zero to get neutral gender
        input_codedict['shape'][:, :] = 0
        
  
        # Process each personality type
        predictions = process_pkl_files(data_pkl_files[:720], input_codedict)
                    
        ## 3. Apply smoothing to predictions
        # predictions = smooth_predictions(predictions)

        # List of original frames
        original_frames_list = sorted([os.path.join(input_list_dir, f) for f in os.listdir(input_list_dir) if f.endswith('.png') or f.endswith('.jpg')])
        ## 4. Detect video cuts that corresponds to abrupt jumps
        # scene_changes = detect_scene_changes(original_frames_list[:1440])
        
        # pdb.set_trace()
        
        ## 5. Apply median filter at scene change indices
        # predictions = apply_median_filter_at_jumps(predictions, scene_changes)

        # predictions vs. listener
        writer_listener_pred = imageio.get_writer(os.path.join(all_output_dir, f"listener_{id}_vs_pred.mp4"), fps=12)
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        thickness = 1
        
        # Loop through both the smoothed predictions and the original frames
        for i, (predict, listener_frame_path) in enumerate(zip(tqdm(predictions), original_frames_list)):
            if i >= len(original_frames_list):
                break  # Stop if there are no more predictions
            
            # Decode the current predictions of each personality type to get the rendered image
            optdict = pixie.decode(predict, param_type='body')           
            optdict['albedo'] = input_optdict['albedo']

            visdict = visualizer.render_results(optdict, input_image)
            transfered_shape = visdict['shape_images'].clone()
        
            # Read the corresponding original frame
            original_frame = imageio.imread(listener_frame_path)

            # Resize the original & speaker frames to match the prediction's size
            original_frame =  cv2.resize(original_frame, (512, 512), interpolation=cv2.INTER_LINEAR)

            # Visualize and save the transfer image alone
            visdict_transfer = {'transfer': transfered_shape}

            grid_image = visualizer.visualize_grid(visdict_transfer, size=512)
            # writer_pred.append_data(grid_image[:,:,[2,1,0]])

            # Add titles to each frame
            draw_label(original_frame, 'Listener - original', (256, 30), font, font_scale, text_color, bg_color, thickness)
            
            grid_image= np.ascontiguousarray(grid_image[:,:,[2,1,0]]).copy()

            cv2.putText(grid_image, 'reference', (200, 30), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Check if the current frame is a jump frame and add a visual indicator
            # if i in scene_changes:
            #     # Add a red border to the frame
            #     border_color = (255, 0, 0)  # Red color in RGB
            #     border_thickness = 10
            #     cv2.rectangle(grid_image, (0, 0), (grid_image.shape[1], grid_image.shape[0]), border_color, border_thickness)
            #     cv2.rectangle(original_frame, (0, 0), (original_frame.shape[1], original_frame.shape[0]), border_color, border_thickness)

            # Concatenate the speaker frame with the predicted frame (side by side)
            conversation_image = np.concatenate((grid_image, original_frame), axis=1)        
            # Visualize and save the concatenated image
            writer_listener_pred.append_data(conversation_image)

        print(f'-- please check the results in {all_output_dir}')
            
    # Close the video writer
    writer_listener_pred.close()


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
