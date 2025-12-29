import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

frame_array = []

frame_dir =   '/media/amr/Backup/myserver/Thesis/udiva/train/recordings_samples/cropped/188189/FC2_T/*.jpg'
frame_dir = sorted(filter(os.path.isfile, glob.glob(frame_dir)))

new_width = new_height = 1024

for filename in tqdm(frame_dir):
    #print(filename)
    frame = cv2.imread(filename)
    height, width, layers = frame.shape
    size = (width, height)
    if size != (new_width, new_height):
        new_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imwrite(filename, new_frame)
