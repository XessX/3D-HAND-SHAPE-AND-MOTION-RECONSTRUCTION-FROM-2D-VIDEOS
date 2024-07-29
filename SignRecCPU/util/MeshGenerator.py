import cv2
import numpy as np
import open3d as o3d
import mediapipe as mp
from HandKeypointProcessor import HandKeypointProcessor

# Existing HandKeypointProcessor class remains the same

class HandMeshGenerator:
    def __init__(self, keypoints):
        self.keypoints = keypoints

    def create_mesh(self):
        # Convert keypoints to point cloud
        points = np.array(self.keypoints)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # Estimate normals for the point cloud
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Orient the normals (optional, can help in some cases)
        point_cloud.orient_normals_consistent_tangent_plane(30)

        # Using Ball Pivoting algorithm to generate a basic mesh
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud, o3d.utility.DoubleVector(radii))
        return mesh


class VideoMeshRenderer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def estimate_depth(self, keypoints_2d):
        """
        Estimate the depth (z-coordinate) for each 2D keypoint.
        This function is heuristic and should be adapted based on your application's requirements.
        """
        # Example: Assign a fixed depth based on the order of keypoints
        # This is a placeholder - you might want to refine this based on the hand's anatomy
        depths = np.linspace(0.1, 0.5, len(keypoints_2d))
        keypoints_3d = [(*kp, z) for kp, z in zip(keypoints_2d, depths)]
        return keypoints_3d

    def render_mesh_to_video(self):
        processor = HandKeypointProcessor(self.input_path, self.output_path)
        frames = processor.load_video()

        # Initialize video writer
        height, width, layers = frames[0].shape
        size = (width, height)
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

        # Initialize MediaPipe Hands
        hands_detector = self.mp_hands.Hands(static_image_mode=True,
                                                  max_num_hands=2,
                                                  min_detection_confidence=0.5)

        # Process each frame
        for frame in frames:
            # Assuming the method to extract and convert keypoints to 3D is implemented
            _, hand_landmarks = processor.detect_keypoints(frame, hands_detector)

            if hand_landmarks:
                # Extract keypoints from the first detected hand
                keypoints_2d = processor.extract_keypoints(hand_landmarks[0])
                if keypoints_2d:
                    # Convert 2D keypoints to 3D
                    keypoints = self.estimate_depth(keypoints_2d)

                    mesh_generator = HandMeshGenerator(keypoints)
                    mesh = mesh_generator.create_mesh()

                    # Render the mesh - this is a basic placeholder rendering using Open3D
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(visible=False)  # Set visible to False for offscreen rendering
                    vis.add_geometry(mesh)
                    vis.update_geometry(mesh)
                    vis.poll_events()
                    vis.update_renderer()
                    image = vis.capture_screen_float_buffer(False)
                    vis.destroy_window()

                    # Convert Open3D image to a format suitable for video writing
                    image = np.asarray(image)
                    image = (image * 255).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Write frame to video
                    out.write(image)

        hands_detector.close()
        out.release()


# Usage example
input_video_path = '../1_f.mp4'
output_video_path = '../1_out.mp4'

renderer = VideoMeshRenderer(input_video_path, output_video_path)
renderer.render_mesh_to_video()