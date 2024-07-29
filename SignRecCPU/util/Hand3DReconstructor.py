import cv2
import numpy as np
import open3d as o3d

class Hand3DReconstructor:
    def __init__(self, front_keypoints, side_keypoints):
        """
        Initializes the 3D reconstructor with keypoints.
        """
        self.front_keypoints = np.array(front_keypoints, dtype=float)
        self.side_keypoints = np.array(side_keypoints, dtype=float)

    def reconstruct_3d_model(self):
        """
        Reconstructs a 3D hand model from front and side view keypoints.
        """
        points_3d = self.triangulate_points(self.front_keypoints, self.side_keypoints)
        hand_model_3d = self.create_point_cloud(points_3d)
        return hand_model_3d

    def reshape_keypoints_for_triangulation(self, keypoints):
        """
        Reshapes keypoints into the correct format [2 x N] for cv2.triangulatePoints.
        """
        kp_array = np.array(keypoints, dtype=float).reshape(2, -1)
        if kp_array.shape[0] == 2 and kp_array.shape[1] != 2:
            # If keypoints are already in [2 x N] format
            return kp_array
        elif kp_array.shape[0] != 2 and kp_array.shape[1] == 2:
            # If keypoints are in [N x 2] format, transpose them
            return kp_array.T
        else:
            raise ValueError("Key points format is incorrect. Expected [N x 2] or [2 x N].")

    def match_keypoints(self, front_kp, side_kp):
        """
        Matches keypoints between front and side views.
        This is a simplified version. In practice, you might need a more robust matching algorithm.
        """
        # Assuming the keypoints are sorted in the same order in both views
        min_length = min(len(front_kp), len(side_kp))
        return front_kp[:min_length], side_kp[:min_length]

    def triangulate_points(self, front_kp, side_kp):
        """
        Triangulates 2D keypoints into 3D points using default camera parameters.
        """
        # Match keypoints between front and side views
        front_kp_matched, side_kp_matched = self.match_keypoints(front_kp, side_kp)

        # Ensure keypoints are in the correct shape [2 x N]
        front_kp_reshaped = self.reshape_keypoints_for_triangulation(front_kp_matched)
        side_kp_reshaped = self.reshape_keypoints_for_triangulation(side_kp_matched)

        # Intrinsic parameters
        img_width, img_height = 640, 640
        cx = img_width / 2
        cy = img_height / 2
        focal_length = 35  # Example focal length
        principal_point = (cx, cy)
        K = np.array([[focal_length, 0, principal_point[0]],
                      [0, focal_length, principal_point[1]],
                      [0, 0, 1]])

        # Extrinsic parameters for front camera (Identity rotation and zero translation)
        R_front = np.eye(3)
        t_front = np.zeros((3, 1))

        # Corrected extrinsic parameters for side camera (90 degrees rotation around Y-axis and translation along X-axis)
        theta = np.radians(90)  # 90 degrees in radians
        R_side = np.array([[np.cos(theta), 0, np.sin(theta)],
                           [0, 1, 0],
                           [-np.sin(theta), 0, np.cos(theta)]])  # Correct 90 degrees rotation around Y-axis
        t_side = np.array([[50], [0], [0]])  # Translation along X-axis

        # Projection matrices
        P_front = np.dot(K, np.hstack((R_front, t_front)))
        P_side = np.dot(K, np.hstack((R_side, t_side)))

        # Triangulate points
        points_3d_homog = cv2.triangulatePoints(P_front, P_side, front_kp_reshaped, side_kp_reshaped)
        points_3d = points_3d_homog[:3] / points_3d_homog[3]

        return points_3d.T  # Transpose to get 3D points in shape [N x 3]

    def create_point_cloud(self, points):
        """
        Creates a 3D point cloud from the given 3D points.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

# # Example Usage
# front_keypoints = [[x1, y1], [x2, y2], ...]  # Replace with actual 2D keypoints from the front view
# side_keypoints = [[x1, y1], [x2, y2], ...]  # Replace with actual 2D keypoints from the side view
#
# reconstructor = Hand3DReconstructor(front_keypoints, side_keypoints)
# hand_model_3d = reconstructor.reconstruct_3d_model()
#
# # Optionally visualize the 3D point cloud
# o3d.visualization.draw_geometries([hand_model_3d])