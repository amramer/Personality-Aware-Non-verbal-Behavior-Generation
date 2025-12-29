import ffmpeg

# Input video file
input_video = "/media/amr/Backup/myserver/Thesis/udiva/train/recordings_samples/cropped/188189/FC2_T.mp4"

# Output video file
output_video = "/media/amr/Backup/myserver/Thesis/udiva/train/recordings_samples/cropped/188189/FC2_T/FC2_T.mp4"

# New width and height
width = height = 1024

# Run ffmpeg command to resize video
(
    ffmpeg
    .input(input_video)
    .filter('scale', width, height)
    .output(output_video)
    .run()
)
