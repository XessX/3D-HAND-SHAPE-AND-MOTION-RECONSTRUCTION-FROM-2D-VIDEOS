import open3d as o3d
from HandKeypointProcessor import HandKeypointProcessor
from Hand3DReconstructor import Hand3DReconstructor
import numpy as np

class VideoTo3DPointCloudConverter:
    """
    Class to convert front and side view videos into a 3D hand model.
    """
    def __init__(self, front_video_path, side_video_path, output_file_path):
        """
        Initializes the converter with video paths and output file path.
        """
        self.front_video_path = front_video_path
        self.side_video_path = side_video_path
        self.output_file_path = output_file_path

    def convert_and_save_3d_model(self):
        """
        Converts the videos to a 3D model and saves the point cloud.
        """
        # Process front and side videos to extract keypoints
        front_keypoints = self.process_video(self.front_video_path)
        side_keypoints = self.process_video(self.side_video_path)

        # Check if keypoints extraction is successful
        if not front_keypoints or not side_keypoints:
            print("Could not extract keypoints from one or both videos.")
            return

        # Reconstruct 3D model from keypoints
        reconstructor = Hand3DReconstructor(front_keypoints, side_keypoints)
        hand_model_3d = reconstructor.reconstruct_3d_model()

        # Print the points
        print("Points:")
        print_val = np.asarray(hand_model_3d.points)
        for i in range(len(print_val)):
            print(print_val[i])

        # Save the 3D model as a point cloud file
        o3d.io.write_point_cloud(self.output_file_path, hand_model_3d)
        print(f"3D model saved to {self.output_file_path}")

    def process_video(self, video_path):
        """
        Process a video to extract hand keypoints using MediaPipe.
        """
        processor = HandKeypointProcessor(video_path, None)
        frames = processor.load_video()
        keypoints = []

        # Initialize MediaPipe Hands
        hands_detector = processor.mp_hands.Hands(static_image_mode=True,
                                                  max_num_hands=2,
                                                  min_detection_confidence=0.5)

        for frame in frames:
            preprocessed_frame = processor.preprocess_frame(frame)
            _, hand_landmarks = processor.detect_keypoints(preprocessed_frame, hands_detector)

            if hand_landmarks:
                # Extract keypoints from the first detected hand
                keypoint = self.extract_keypoints(hand_landmarks[0])
                if keypoint:
                    keypoints.append(keypoint)

        hands_detector.close()
        return keypoints

    def extract_keypoints(self, hand_landmarks):
        """
        Extracts keypoints from a single hand's landmarks.
        """
        # Convert landmarks to a list of (x, y) tuples
        keypoints = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
        # return keypoints if len(keypoints) == 2 else None  # Ensure keypoints are in pairs
        return keypoints

# Example Usage
front_video_path = '../1_f.mp4'  # Path to the front view video
side_video_path = '1_s.mp4'    # Path to the side view video
output_file_path = '../hand_model_3d.ply'  # Output file path for the 3D model

converter = VideoTo3DPointCloudConverter(front_video_path, side_video_path, output_file_path)
converter.convert_and_save_3d_model()