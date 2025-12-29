import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

frame_array = []
frame_dir = '/home/amr/Clusters@DFKI/Thesis/UDIVAv0.5/extracted_features/train/test/*.jpg'
frame_dir = sorted(filter(os.path.isfile, glob.glob(frame_dir)))

# Set the dimensions of the video
frame_width = 1024
frame_height = 1024

# Set the codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/home/amr/Clusters@DFKI/Thesis/UDIVAv0.5/extracted_features/train/test.mp4', fourcc, 12.0, (frame_width, frame_height))

# Iterate through the list of images
for filename in tqdm(frame_dir):
    frame = cv2.imread(filename)
    out.write(frame)

out.release()
