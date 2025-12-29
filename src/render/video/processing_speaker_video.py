import os
import sys
import argparse
import pdb
from tqdm import tqdm
import logging
import traceback
import numpy as np
import glob
import csv
import time
import cv2
import subprocess

import subprocess
import os
import cv2

def merge_video_and_audio(input_video_path, input_audio_path, output_dir, id):
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, f"{id}_processed.mp4")
    temp_video_path = os.path.join(temp_dir, f"{id}_temp_2.mp4")

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Write the original frame (not mirrored) to the temporary video file
        out.write(frame)

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Merge the video with the audio
    subprocess.run([
        'ffmpeg', '-y','-i', temp_video_path, '-i', input_audio_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', 
        '-c:a', 'aac', '-b:a', '128k',
        '-async', '1',
        '-vsync', '1',
        output_video_path
    ], check=True)

    # Remove the temporary video file
    os.remove(temp_video_path)

    return output_video_path



def mirror_video_and_merge_audio(input_video_path, input_audio_path, output_dir, id):
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, f"{id}_mirrored.mp4")
    temp_video_path = os.path.join(temp_dir, f"{id}_temp.mp4")

    # Mirroring the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        mirrored_frame = cv2.flip(frame, 1)
        out.write(mirrored_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Re-encoding the mirrored video with audio
    if os.path.exists(temp_video_path):
        subprocess.run([
            'ffmpeg','-y', '-i', temp_video_path, '-i', input_audio_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', 
            '-c:a', 'aac', '-b:a', '128k',
            '-async', '1',
            '-vsync', '1',
            output_video_path
        ], check=True)
        os.remove(temp_video_path)
    else:
        print(f"Temporary video file not found: {temp_video_path}")
        return None

    return output_video_path


def concatenate_videos(video_paths, output_dir, id):
    # Temporary file for the concatenated video
    temp_concat_path = os.path.join(output_dir, f"{id}_temp_concat.mp4")
    final_output_path = os.path.join(output_dir, f"speaker_vs_{id}.mp4")

    # Create a file listing all videos to concatenate
    with open('input_files.txt', 'w') as file:
        for path in video_paths:
            file.write(f"file '{path}'\n")

    # Concatenate videos using the concat demuxer
    subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'input_files.txt', '-c', 'copy', temp_concat_path], check=True)

    # Calculate the time to skip first 32 frames (for 25 fps)
    skip_time = 1.28  # 32 frames / 25 fps

    # Skip first 32 frames and re-encode the concatenated video to ensure consistent frame timing
    subprocess.run([
        'ffmpeg','-y','-ss', str(skip_time), '-i', temp_concat_path, 
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-r', '24',  # Set output frame rate to 24 fps 
        '-c:a', 'aac', '-b:a', '128k',
        '-async', '1',
        '-vsync', '1',
        final_output_path
    ], check=True)

    # Clean up temporary files
    os.remove('input_files.txt')
    os.remove(temp_concat_path)

    return final_output_path





if __name__ == "__main__":

    input_dir = "/home/UDIVAv0.5/"
    output_dir = "/home/pipeline/outputs/speaker_videos"

    video_crop_dir = os.path.join(input_dir, "extracted_features/test/video_crop")
    parts_val_file = os.path.join(input_dir, "test/metadata", "parts_val.csv")
    sessions_val_file = os.path.join(input_dir, "test/metadata", "sessions_val.csv")

    participants_dict = {}

    with open(parts_val_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            participant_id = row['ID']
            sessions = []
            for i in range(1, 6):
                session_id = row[f'SESSION{i}']
                if session_id:
                    sessions.append(session_id)
            participants_dict[participant_id] = sessions
    # print(participants_dict)

    sessions_dict = {}

    with open(sessions_val_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            session_id = row['ID']
            participant_ids = [row['PART.1'], row['PART.2']]
            sessions_dict[session_id] = participant_ids

    p1_speak_dict = {}

    for participant_id in participants_dict:
        sessions = participants_dict[participant_id]
        for session_id in sessions:
            participant_ids_list = sessions_dict[session_id]
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
                # print(f"{participant_id}: {p0} {p1}")
                session_dir = os.path.join(video_crop_dir, session_id)
                p1_speak = f"{p1}_{'_'.join(['speak_all'])}"
                # print(f"{participant_id}: {p1_speak}")
                p1_speak_files = [filename for filename in os.listdir(session_dir) if filename.startswith(p1_speak)]
                # print(f"{participant_id}: {p1_speak_files}")

                for file_name in p1_speak_files:
                    # Initialize paths
                    video_file_path = None
                    audio_file_path = None

                    if file_name.endswith('.mp4'):
                        video_file_path = os.path.join(session_dir, file_name)

                        # Assuming the audio file has the same name but with .mp3 extension
                        audio_file_candidate = file_name.replace('.mp4', '.mp3')
                        if audio_file_candidate in p1_speak_files:
                            audio_file_path = os.path.join(session_dir, audio_file_candidate)

                        # Add the pair to the dictionary if both video and audio files are found
                        if video_file_path and audio_file_path:
                            if participant_id not in p1_speak_dict:
                                p1_speak_dict[participant_id] = []
                            p1_speak_dict[participant_id].append([video_file_path, audio_file_path, p1])


    for id in tqdm(participants_dict):
        if id in p1_speak_dict:
            processed_videos = []
            for input_video_path, audio_path, part in p1_speak_dict[id]:
                if part == "part1":
                    processed_path = mirror_video_and_merge_audio(input_video_path, audio_path, output_dir, id)
                    processed_videos.append(processed_path)
                else:
                    processed_path = merge_video_and_audio(input_video_path, audio_path, output_dir, id)
                    processed_videos.append(processed_path)

            if len(processed_videos) > 1:
                # final_output_path = os.path.join(output_dir, f"speaker_vs_{id}.mp4")
                concatenate_videos(processed_videos, output_dir, id)
            elif len(processed_videos) == 1:
                final_output_path = os.path.join(output_dir, f"speaker_vs_{id}.mp4")
                for video_path in processed_videos:
                    # Calculate the time to skip first 32 frames (for 25 fps)
                    skip_time = 1.28  # 32 frames / 25 fps
                    # Skip first 32 frames and re-encode the concatenated video to ensure consistent frame timing
                    subprocess.run([
                        'ffmpeg','-y','-ss', str(skip_time), '-i', video_path, 
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                        '-r', '24',  # Set output frame rate to 24 fps 
                        '-c:a', 'aac', '-b:a', '128k',
                        '-async', '1',
                        '-vsync', '1',
                        final_output_path
                    ], check=True)

                

