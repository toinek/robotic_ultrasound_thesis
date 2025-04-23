#!/usr/bin/env python
import argparse
import open3d as o3d
import cv2
import rospy
import time
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp

from realsense_phantom import RealSenseCamera
from realsense.data_analysis.computation_logger import timing_logger

class RealSenseCameraMediaPipe(RealSenseCamera):
    def __init__(self, width=1280, height=720, fps=15, clip=3, debug=False, use_pcd=False):
        super().__init__(width=width, height=height, fps=fps, clip=clip, debug=debug)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=1,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        self.use_pcd = use_pcd

    def extract_body_landmarks(self, color_image):
        """Extracts all body landmarks using MediaPipe Pose but only displays left arm landmarks."""
        results = self.pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None, color_image  # No landmarks detected, return original image

        # Extract all keypoints
        keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

        # Indices for left arm landmarks
        arm_indices = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_THUMB.value
        ]
        annotated_image = np.copy(color_image)
        # Show annotated image with only left arm keypoints
        for i in arm_indices:
            lm = results.pose_landmarks.landmark[i]
            cv2.circle(annotated_image, (int(lm.x * color_image.shape[1]), int(lm.y * color_image.shape[0])), 5,
                       (0, 255, 0), -1)

        self.last_arm_points = [(int(lm.x * color_image.shape[1]), int(lm.y * color_image.shape[0])) for i in arm_indices]

        return keypoints, annotated_image

    def generate_keypoint_pcd(self, keypoints, depth_frame, color_image):
        """Extracts only the left arm keypoints and generates a PCD with color."""
        arm_indices = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
            self.mp_pose.PoseLandmark.LEFT_WRIST.value,
            self.mp_pose.PoseLandmark.LEFT_THUMB.value
        ]
        arm_keypoints_3d = []
        arm_colors = []
        for i in arm_indices:
            u, v = int(keypoints[i][0] * self.width), int(keypoints[i][1] * self.height)
            if 0 <= u < self.width and 0 <= v < self.height:
                depth = depth_frame.get_distance(u, v)
                if 0 < depth < self.clip_distance:
                    point_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u, v], depth)
                    arm_keypoints_3d.append(point_3d)
                    color = color_image[v, u] / 255.0  # Normalize color values to [0, 1]
                    arm_colors.append(color)

        if len(arm_keypoints_3d) != 4:
            rospy.logwarn("Insufficient left arm keypoints detected. Skipping frame.")
            return None

        # Convert to numpy array and Open3D PCD
        arm_keypoints_3d = np.array(arm_keypoints_3d, dtype=np.float64)
        arm_colors = np.array(arm_colors, dtype=np.float64)
        arm_pcd = o3d.geometry.PointCloud()
        arm_pcd.points = o3d.utility.Vector3dVector(arm_keypoints_3d)
        arm_pcd.colors = o3d.utility.Vector3dVector(arm_colors)
        return arm_pcd

    def generate_scene_pcd(self, depth_frame, color_image):
        """
        Generates a full dense point cloud of the scene from the depth and color images.
        """
        points_3d = []
        colors = []

        depth_intrinsics = self.depth_intrin
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        for v in range(self.height):
            for u in range(self.width):
                depth = depth_frame.get_distance(u, v)
                if 0 < depth < self.clip_distance:  # Ensure valid depth
                    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                    points_3d.append(point_3d)
                    colors.append(color_image_rgb[v, u] / 255.0)  # Normalize color values

        if not points_3d:
            rospy.logwarn("No valid points found in the scene.")
            return None

        # Convert to Open3D point cloud
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(np.array(points_3d, dtype=np.float64))
        scene_pcd.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
        return scene_pcd

    def extract_human_point_cloud(self, depth_frame, dense_pcd, keypoints):
        """
        Extracts the human body point cloud from a dense scene point cloud using key landmarks from MediaPipe Pose.

        Args:
            depth_frame (pyrealsense2.depth_frame): The depth frame from the RealSense camera.
            dense_pcd (o3d.geometry.PointCloud): The dense point cloud of the scene.
            keypoints (list): List of 2D keypoints detected by MediaPipe Pose.

        Returns:
            o3d.geometry.PointCloud: The filtered point cloud containing only the human body, with color.
        """
        # Step 1: Detect landmarks and project them to 3D
        if keypoints is None:
            rospy.logwarn("No landmarks detected. Skipping human extraction.")
            return None, None

        # Project the 2D keypoints to 3D using the depth frame
        keypoints_3d = []
        depth_intrinsics = self.depth_intrin
        for kp in keypoints:
            u, v = int(kp[0] * self.width), int(kp[1] * self.height)
            # Ensure the pixel is within the valid image dimensions
            if 0 <= u < self.width and 0 <= v < self.height:
                depth = depth_frame.get_distance(u, v)
                if 0 < depth < self.clip_distance:  # Ensure valid depth
                    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                    keypoints_3d.append(point)
            else:
                rospy.logwarn(f"Keypoint ({u}, {v}) is out of range and will be skipped.")

        if not keypoints_3d:
            rospy.logwarn("No valid 3D keypoints could be extracted.")
            return None, None

        keypoints_3d = np.array(keypoints_3d)

        # Step 2: Filter the dense point cloud based on keypoints
        filtered_points = []
        filtered_colors = []  # To store colors corresponding to filtered points
        radius = 0.2  # Adjust radius around each landmark for inclusion
        dense_points = np.asarray(dense_pcd.points)
        dense_colors = np.asarray(dense_pcd.colors)

        for kp_3d in keypoints_3d:
            distances = np.linalg.norm(dense_points - kp_3d, axis=1)
            mask = distances < radius  # Filter points within the radius
            filtered_points.extend(dense_points[mask])
            filtered_colors.extend(dense_colors[mask])  # Map corresponding colors

        # Step 3: Create a new point cloud with the filtered points and their colors
        filtered_pcd = o3d.geometry.PointCloud()
        if filtered_points:
            filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
            filtered_pcd.colors = o3d.utility.Vector3dVector(np.array(filtered_colors))
            rospy.loginfo(f"Extracted human point cloud with {len(filtered_points)} points and color.")
        else:
            rospy.logwarn("No points were found near the landmarks.")

        return filtered_pcd


    def remove_plane(self, pcd, x_range=None, y_range=None, z_range=None):
        """Filters points outside the specified x, y, z ranges in the point cloud."""
        points = np.asarray(pcd.points)
        mask = np.ones(points.shape[0], dtype=bool)

        if x_range is not None:
            mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        if y_range is not None:
            mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        if z_range is not None:
            mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])

        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])

        if pcd.has_colors():
            filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

        return filtered_pcd

    def _run(self, show_img=True):
        """Extracts keypoints, generates PCD, aligns, filters, and publishes based on mode."""

        # Step 1: Extract frames from the RealSense camera
        start_time = time.perf_counter()
        self.color_frame, self.depth_frame = self.get_frames()

        color_image = np.asanyarray(self.color_frame.get_data())
        self.color_image_copy = color_image.copy()
        end_time = time.perf_counter()
        timing_logger.log_timing("RS_mediapipe/Frame Extraction", end_time - start_time)

        # Step 2: Detect ArUco marker
        if not show_img:
            self.detect_aruco(color_image, draw_aruco=False)
        else:
            self.detect_aruco(color_image)

        if self.num_frames < 5:
            # 3. Calculate the transformation from the camera frame to the robot frame
            self.calculate_transformation(
                aruco_camera_pos=self.camera_pos,
                aruco_camera_orient=self.camera_orient,
                aruco_robot_pos=[0.0, 0.425, -0.018]
            )

        # Step 3: Extract body landmarks
        start_time = time.perf_counter()
        keypoints, annotated_image = self.extract_body_landmarks(color_image)
        if keypoints is None:
            rospy.logwarn("No keypoints detected. Skipping frame.")
            return
        end_time = time.perf_counter()
        timing_logger.log_timing("RS_mediapipe/Keypoint Extraction", end_time - start_time)

        # Step 3.5: Overlay trajectory and show extracted body landmarks
        self.image_with_trajectory = self.overlay_trajectory(annotated_image)
        if show_img:
            cv2.imshow("MediaPipe Body Landmarks", self.image_with_trajectory)
            cv2.waitKey(1)

        # Step 4: Generate PCD
        start_time = time.perf_counter()
        if self.use_pcd:
            scene_pcd = self.generate_scene_pcd(self.depth_frame, color_image)
            pcd = self.extract_human_point_cloud(self.depth_frame, scene_pcd, keypoints)
        else:
            pcd = self.generate_keypoint_pcd(keypoints, self.depth_frame, color_image)
        end_time = time.perf_counter()
        timing_logger.log_timing("RS_mediapipe/Point Cloud Generation", end_time - start_time)

        if pcd is None:
            rospy.logwarn("No point cloud generated. Skipping frame.")
            return

        # Step 5: Align PCD to robot frame
        start_time = time.perf_counter()
        transformed_pcd = self.align_pcd(pcd, publish=False)
        end_time = time.perf_counter()
        timing_logger.log_timing("RS_mediapipe/Point Cloud Transformation/Alignment", end_time - start_time)

        # Step 6: If dense PCD, remove plane
        if self.use_pcd:
            start_time = time.perf_counter()
            transformed_pcd = self.remove_plane(
                transformed_pcd, x_range=(0.0, 1.0), y_range=(-0.6, 0.5), z_range=(0.0, 0.8)
            )
            end_time = time.perf_counter()
            timing_logger.log_timing("RS_mediapipe/Plane Removal", end_time - start_time)
        # o3d.io.write_point_cloud(
        #     "../gaussian_process_transportation/policy_transportation/apply_laplacian_editing/1800feb_sparse_arm_test.pcd", transformed_pcd)

        # Step 7: Publish using `convert_pcd_to_pc2msg()`
        if transformed_pcd is not None:
            start_time = time.perf_counter()
            self.pcd_processor.convert_pcd_to_pc2msg(transformed_pcd, publisher=self.pcd_publisher)
            end_time = time.perf_counter()
            timing_logger.log_timing("RS_mediapipe/PCD Publishing", end_time - start_time)

        # Debugging: Visualize if enabled
        if self.debug:
            o3d.visualization.draw_geometries([transformed_pcd], window_name="Transformed Point Cloud")

if __name__ == '__main__':
    rospy.init_node('object_detection')

    parser = argparse.ArgumentParser(description="Choose between dense point cloud or keypoint-based extraction")
    parser.add_argument('--pcd', type=str, default="false", help="Set to 'true' for dense PCD, 'false' for keypoints")
    args, unknown = parser.parse_known_args()

    detector = RealSenseCameraMediaPipe(debug=False, use_pcd=args.pcd.lower() == "true")

    while not rospy.is_shutdown():
        detector._run()
