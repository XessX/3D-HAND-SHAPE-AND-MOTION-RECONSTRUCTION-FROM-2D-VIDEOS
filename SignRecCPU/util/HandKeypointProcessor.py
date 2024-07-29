import cv2
import mediapipe as mp
import numpy as np


class HandKeypointProcessor:
    """
    A class to process a video, detect hand keypoints using MediaPipe, and save the output video.
    """

    def __init__(self, input_path, output_path):
        """
        Initializes the HandKeypointProcessor with input and output video paths.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def load_video(self):
        """
        Load a video from the given path and extract frames.
        """
        cap = cv2.VideoCapture(self.input_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.preprocess_frame(frame)
            if frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def preprocess_frame(self, frame, target_size=(320, 320)):
        """
        Preprocess a single frame (crop hand region and resize).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.Hands(static_image_mode=True).process(frame_rgb)

        if results.multi_hand_landmarks:
            # # Assuming only one hand is present, get the first hand's landmarks
            # hand_landmarks = results.multi_hand_landmarks[0]
            #
            # # Get bounding box coordinates for the hand
            # x_coords = [lm.x for lm in hand_landmarks.landmark]
            # y_coords = [lm.y for lm in hand_landmarks.landmark]
            # x_min, x_max = min(x_coords), max(x_coords)
            # y_min, y_max = min(y_coords), max(y_coords)
            #
            # # Convert normalized coordinates to pixel values
            # frame_height, frame_width = frame.shape[:2]
            # const = 0.01
            # margin_x = int(const * frame_width)
            # margin_y = int(const * frame_height)
            # x_min, x_max = int(x_min * frame_width) - margin_x, int(x_max * frame_width) + margin_x
            # y_min, y_max = int(y_min * frame_height) - margin_y, int(y_max * frame_height) + margin_y
            # x_min = x_min if x_min >= 0 else 0
            # x_max = x_max if x_max < frame_width else frame_width - 1
            # y_min = y_min if y_min >= 0 else 0
            # y_max = y_max if y_max < frame_height else frame_height - 1
            #
            # # Crop and resize
            # cropped_hand = frame[y_min:y_max, x_min:x_max]
            # resized_hand = cv2.resize(cropped_hand, target_size)
            # return resized_hand
            return frame

        return None  # Resize the whole frame if no hand is detected

    def detect_keypoints(self, frame, hands_detector):
        """
        Use MediaPipe to detect hand keypoints and draw them on the frame.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return frame, results.multi_hand_landmarks

    def process_and_save_video(self):
        """
        Process the input video to detect hand keypoints, draw them on frames,
        and save the output video.
        """
        frames = self.load_video()
        output_frames = []

        # Initialize MediaPipe Hands
        hands_detector = self.mp_hands.Hands(static_image_mode=True,
                                             max_num_hands=2,
                                             min_detection_confidence=0.5)

        for frame in frames:
            # preprocessed_frame = self.preprocess_frame(frame)
            if frame is not None:
                output_frame, hand_landmarks = self.detect_keypoints(frame, hands_detector)

                if hand_landmarks:
                    output_frames.append(output_frame)

        # Release MediaPipe resources
        hands_detector.close()

        # Save the output video
        self.save_video(output_frames)

    def process_video(self):
        """
        Process a video to extract hand keypoints using MediaPipe.
        """
        frames = self.load_video()
        keypoints = []

        # Initialize MediaPipe Hands
        hands_detector = self.mp_hands.Hands(static_image_mode=True,
                                                  max_num_hands=2,
                                                  min_detection_confidence=0.5)

        for frame in frames:
            # preprocessed_frame = self.preprocess_frame(frame)
            if frame is not None:
                _, hand_landmarks = self.detect_keypoints(frame, hands_detector)

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

    def save_video(self, frames):
        """
        Save the processed frames as a video.
        """
        height, width, layers = frames[0].shape
        size = (width, height)
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

        for frame in frames:
            out.write(frame)

        out.release()


# # Usage
# input_video_path = '1_f.mp4'  # Replace with your input video path
# output_video_path = '1_out.mp4'  # Replace with your desired output video path
#
# processor = HandKeypointProcessor(input_video_path, output_video_path)
# processor.process_and_save_video()