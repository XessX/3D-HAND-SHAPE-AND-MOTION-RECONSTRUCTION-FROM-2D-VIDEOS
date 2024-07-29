import cv2
import os
import sys
from tqdm import tqdm
from util.HandKeypointProcessor import HandKeypointProcessor
from acr.main import ACR
from acr.config import args, parse_args, ConfigContext

# Assuming ACR and other required imports are correctly set up as per your second code snippet

def process_video_and_save_3d_mesh(video_path, output_video_path):
    """
    Processes a video file to generate and save a video of 3D hand meshes.
    """
    # Initialize HandKeypointProcessor for extracting 2D keypoints
    hand_processor = HandKeypointProcessor(video_path, None)  # Output path not needed for this use case

    # Initialize ACR model for generating 3D hand meshes
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading configurations from {}'.format(args_set.configs_yml))
        acr_model = ACR(args_set=args_set)

    # Prepare for video processing
    frames = hand_processor.load_video()
    results = []
    output_frames = []  # To store rendered 3D mesh images

    # Process each frame
    for frame in (frames):
        # Use ACR model to process each frame and get the 3D model rendering as an image
        rendered_image, res = acr_model(frame, 'frame')  # Assuming acr_model() returns the rendered 3D model image
        if rendered_image is not None:
            output_frames.append(rendered_image)
            results.append(res)

    # Save output_frames as a video
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for img in output_frames:
        video_writer.write(img)

    video_writer.release()
    print(f"Saved rendered 3D mesh video to {output_video_path}")


# Usage
video_path = '1_f.mp4'
output_video_path = '1_out.mp4'
print('Start..')
process_video_and_save_3d_mesh(video_path, output_video_path)