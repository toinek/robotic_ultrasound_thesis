#!/usr/bin/env python
import time
import cv2
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import pandas as pd
from cv_bridge import CvBridge
from std_msgs.msg import UInt64
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Quaternion

from pcd_helpers import PointCloudConverter
from realsense.data_analysis.computation_logger import timing_logger

class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=15, clip=3, debug=False):
        """Initialize the RealSense camera node.

        Args:
            width (int): Width of the color and depth images.
            height (int): Height of the color and depth images.
            fps (int): Frame rate of the camera.
            clip (float): Clipping distance for the point cloud.
            real (bool): Flag to use the RealSense camera or not.
            debug (bool): Flag to enable debugging mode.
        """

        self.width = width
        self.height = height
        self.fps = fps
        self.clip_distance = clip
        self.debug = debug

        self.bridge = CvBridge()
        self.pcd_processor = PointCloudConverter()
        # self.visualizer = TrajectoryVisualizer()

        self.pcd_publisher = rospy.Publisher('transport/calibrated_pcd', pc2.PointCloud2, queue_size=1)

        # Subscriber for the Kuka trajectory
        self.trajectory_sub = rospy.Subscriber('/iiwa14/kuka_trajectory_old', PoseArray, self.trajectory_cb)
        self.trajectory_sub_new = rospy.Subscriber('/iiwa14/kuka_trajectory_new', PoseArray, self.trajectory_cb)
        self.spline_index_sub = rospy.Subscriber('/CartesianImpedanceController/current_spline_index', UInt64,
                                                 self.spline_index_cb)

        self.camera_pos = None
        self.camera_orient = None
        self.rotation_matrix = None
        self.translation_vector = None
        self.num_frames = 0
        self.kuka_trajectory = None  # Store received trajectory points
        self.current_spline_index = 0

        self.setup_pipeline()


    def setup_pipeline(self):
        """Set up the RealSense pipeline and configuration."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.profile = self.pipeline.start(self.config)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # Intrinsics and extrinsics
        self.depth_intrin = None
        self.color_intrin = None
        self.color_to_depth_extrin = None
        self.depth_to_color_extrin = None

        self.setup_intrinsics_and_extrinsics()

    def setup_intrinsics_and_extrinsics(self):
        """Get the camera intrinsics and extrinsics for the RealSense camera."""
        # Get the depth and color intrinsics and extrinsics
        depth_stream = self.profile.get_stream(rs.stream.depth)
        color_stream = self.profile.get_stream(rs.stream.color)
        self.depth_intrin = depth_stream.as_video_stream_profile().intrinsics
        self.color_intrin = color_stream.as_video_stream_profile().intrinsics

        # Intrinsic matrix
        self.camera_intrinsics = np.array([[self.color_intrin.fx, 0, self.color_intrin.ppx],
                                           [0, self.color_intrin.fy, self.color_intrin.ppy],
                                           [0, 0, 1]])

        # Distortion coefficients
        self.distortion_coeffs = np.array(self.color_intrin.coeffs)

        self.depth_to_color_extrin = depth_stream.get_extrinsics_to(color_stream)
        self.color_to_depth_extrin = color_stream.get_extrinsics_to(depth_stream)

    def trajectory_cb(self, msg):
        """Callback to receive and store trajectory points from PoseArray in the Kuka frame."""
        self.kuka_trajectory = []
        for pose in msg.poses:
            self.kuka_trajectory.append([pose.position.x, pose.position.y, pose.position.z])
        self.kuka_trajectory = np.array(self.kuka_trajectory)

        rospy.loginfo(f"Received {len(self.kuka_trajectory)} trajectory points.")

    def spline_index_cb(self, spline_index_msg):
        self.current_spline_index = spline_index_msg.data

    def get_frames(self):
        """Get aligned color and depth frames from the RealSense camera."""
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        return color_frame, depth_frame

    def project_3d_to_2d(self, points_3d):
        """Projects 3D points in camera frame to 2D pixel coordinates using intrinsics."""
        points_2d = []
        for point in points_3d:
            X, Y, Z = point
            if Z > 0:  # Prevent division by zero
                u = int((self.camera_intrinsics[0, 0] * X / Z) + self.camera_intrinsics[0, 2])
                v = int((self.camera_intrinsics[1, 1] * Y / Z) + self.camera_intrinsics[1, 2])

                if 0 <= u < self.width and 0 <= v < self.height:
                    points_2d.append((u, v))
                else:
                    pass
                    # rospy.logwarn(f"Projected point ({u}, {v}) is out of bounds.")
        # rospy.loginfo(f"Projected {len(points_2d)} 2D points onto the image.")
        return points_2d

    def overlay_trajectory(self, color_image):
        """Transform the trajectory from the robot frame to the camera frame and overlay it onto the image."""
        if self.kuka_trajectory is None or len(self.kuka_trajectory) == 0:
            rospy.logwarn("No trajectory data available. Skipping overlay.")
            return color_image

        if self.rotation_matrix is None or self.translation_vector is None:
            rospy.logwarn("Skipping trajectory overlay: Transformation matrices are not available.")
            return color_image

        # Compute inverse transformation (robot → camera)
        inv_rotation_matrix = self.rotation_matrix.T  # Transpose of rotation matrix is its inverse
        inv_translation_vector = -np.dot(inv_rotation_matrix, self.translation_vector)

        # Transform trajectory from robot frame to camera frame
        trajectory_camera = np.dot(self.kuka_trajectory[self.current_spline_index:, :], inv_rotation_matrix.T) + inv_translation_vector

        # Project to 2D image space
        trajectory_2d = self.project_3d_to_2d(trajectory_camera)

        # Draw the trajectory on the image
        for pt in trajectory_2d:
            if 0 <= pt[0] < self.width and 0 <= pt[1] < self.height:  # Ensure valid pixel coordinates
                cv2.circle(color_image, pt, 3, (0, 255, 0), -1)  # Green dots for trajectory

        return color_image


    def detect_light_brown(self, color_image):
        bounding_box = None

        # Convert the BGR image to HSV color space
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Values to distinguish light brown color
        lower_brown = np.array([5, 50, 100])  # Lower bound of light brown in HSV
        upper_brown = np.array([25, 255, 255])  # Upper bound of light brown in HSV

        # 5, 100,100 and 20,255,255 for female

        # Create a binary mask where light brown colors are in range
        mask = cv2.inRange(hsv_image, lower_brown, upper_brown)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to store the outermost bounds
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')

        # Draw bounding boxes around detected brown regions
        for contour in contours:
            if cv2.contourArea(contour) > 10000:  # Filter small contours based on area
                x, y, w, h = cv2.boundingRect(contour)

                # Update the outer bounds to encompass all bounding boxes
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

        # Check if any valid bounding boxes were found
        if x_min < x_max and y_min < y_max:
            bounding_box = [x_min, y_min, x_max, y_max]
            # Draw the final bounding box on the original color image
            cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        else:
            rospy.logerr("No valid light brown regions detected.")

        return color_image, bounding_box

    def detect_aruco(self, color_image, aruco_dict=cv2.aruco.DICT_6X6_100, marker_length=0.096,
                     camera_matrix=None, dist_coeffs=None, draw_aruco=True):
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, _ = detector.detectMarkers(color_image)
        aruco_poses = {}

        if ids is not None:
            if draw_aruco:
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            camera_matrix = self.camera_intrinsics
            dist_coeffs = self.distortion_coeffs

            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                used_marker_length = marker_length if marker_id == 0 else 0.05

                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=self.get_marker_object_points(used_marker_length),
                    imagePoints=corner[0],
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    quaternion = tf_trans.quaternion_from_matrix(
                        np.vstack((np.hstack((rotation_matrix, np.zeros((3, 1)))), np.array([0, 0, 0, 1])))
                    )

                    if marker_id == 0:
                        self.camera_pos = np.array(tvec).reshape((3,))
                        self.camera_orient = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
                    else:
                        aruco_poses[marker_id] = {"translation": tvec.flatten(), "rotation_matrix": rotation_matrix, "quaternion": quaternion}

            return color_image, aruco_poses
        else:
            rospy.logwarn("No ArUco markers detected in the image.")
        return color_image, None

    def compute_aruco_line(self, color_image):
        _, aruco_poses = self.detect_aruco(color_image)
        if aruco_poses is None:
            rospy.logwarn("No ArUco markers detected.")
            return

        markers = {idx: pose for idx, pose in aruco_poses.items() if idx != 0}
        sorted_markers = sorted(markers.items())

        transformed_poses = {}
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = "robot_base"

        for idx, pose in sorted_markers:
            tvec = np.array(pose["translation"]).reshape(3, 1)
            rotation_matrix = pose["rotation_matrix"]
            quaternion = pose["quaternion"]  # Extract quaternion from the dictionary

            homogeneous_point = np.vstack((tvec, [[1]]))

            if self.rotation_matrix is not None and self.translation_vector is not None:
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = self.rotation_matrix
                transformation_matrix[:3, 3] = self.translation_vector

                # Transform position
                transformed_point = np.dot(transformation_matrix, homogeneous_point)[:3].flatten()

                # Transform orientation
                transformed_rotation = np.dot(self.rotation_matrix, rotation_matrix)
                transformed_quaternion = tf_trans.quaternion_from_matrix(
                    np.vstack((np.hstack((transformed_rotation, np.zeros((3, 1)))), np.array([0, 0, 0, 1])))
                )

                transformed_poses[idx] = {
                    "translation": transformed_point,
                    "rotation_matrix": transformed_rotation,
                    "quaternion": transformed_quaternion
                }

                # Create Pose message with both position and orientation
                pose_msg = Pose()
                pose_msg.position.x, pose_msg.position.y, pose_msg.position.z = transformed_point
                pose_msg.orientation = Quaternion(*quaternion)  # Assign quaternion directly

                pose_array_msg.poses.append(pose_msg)
            else:
                rospy.logwarn("Transformation matrices are not available. Cannot transform points.")
                return

        # Publish the list of ArUco poses (position + orientation)
        self.aruco_line_pub.publish(pose_array_msg)

        return transformed_poses

    def get_marker_object_points(self, marker_length):
        """
        Returns the 3D coordinates of the marker corners in the marker's coordinate system.
        The marker is assumed to be in the XY plane (Z=0) and centered at the origin.
        """
        half_length = marker_length / 2.0
        return np.array([
            [-half_length, half_length, 0],
            [half_length, half_length, 0],
            [half_length, -half_length, 0],
            [-half_length, -half_length, 0]
        ], dtype=np.float32)

    def filter_point_cloud_by_bounding_box(self, depth_frame, color_frame, bounding_box):
        # Get the depth and color image data as numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the color image from BGR to RGB
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Get the intrinsics for the depth stream
        depth_intrinsics = self.depth_intrin

        # Ensure bounding box coordinates are within the image dimensions
        height, width = depth_image.shape

        # Extract bounding box coordinates and validate them
        x_min, y_min, x_max, y_max = bounding_box
        # x_min, y_min, x_max, y_max = 0, 0, width, height

        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
            rospy.logwarn("Bounding box coordinates are out of image bounds. Adjusting to fit within image dimensions.")
            x_min = max(0, x_min) #- 50
            y_min = max(0, y_min) #+ 50
            x_max = min(width, x_max) #- 50
            y_max = min(height, y_max) #+ 50

        # Check if the bounding box is valid
        if x_min >= x_max or y_min >= y_max:
            rospy.logerr("Invalid bounding box dimensions. Returning an empty point cloud.")
            return o3d.geometry.PointCloud()

        # Create lists to store filtered points and their corresponding colors
        filtered_points = []
        filtered_colors = []

        # Iterate over the pixels inside the bounding box to extract 3D points
        for v in range(y_min, y_max):
            for u in range(x_min, x_max):
                depth = depth_image[v, u] * depth_frame.get_units()
                if depth > 0 and depth < self.clip_distance:  # Filter out points beyond clipping distance
                    try:
                        # Deproject the 2D pixel (u, v) into a 3D point using the depth value
                        point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
                        if point[2] < 1.5:
                            filtered_points.append(point)

                            # Get the color corresponding to the pixel (u, v) in the color image
                            color = color_image_rgb[v, u] / 255.0  # Normalize RGB values to [0, 1]
                            filtered_colors.append(color)
                    except Exception as e:
                        rospy.logwarn(f"Error deprojecting pixel ({u}, {v}): {e}")
                        continue

        # Convert the list of points and colors to numpy arrays
        filtered_points = np.array(filtered_points, dtype=np.float64)
        filtered_colors = np.array(filtered_colors, dtype=np.float64)

        # Handle the case where no points were found within the bounding box or valid range
        if filtered_points.size == 0:
            rospy.logerr("No valid points were found within the bounding box. Returning an empty point cloud.")
            return o3d.geometry.PointCloud()


        # Create an Open3D point cloud object and set its points and colors
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Pass the filtered point cloud to the alignment
        rospy.loginfo("Filtered point cloud processed")

        return filtered_pcd

    def calculate_transformation(self, aruco_camera_pos, aruco_camera_orient, aruco_robot_pos):
        """
        Calculates and stores the transformation matrices from the camera frame to the robot frame.

        Args:
            aruco_camera_pos (np.array): Position of the ArUco marker in the camera frame (x, y, z).
            aruco_camera_orient (np.array): Quaternion orientation of the ArUco marker in the camera frame (x, y, z, w).
            aruco_robot_pos (np.array): Position of the ArUco marker in the robot frame (x, y, z).
        """
        if aruco_camera_pos is None or aruco_camera_orient is None:
            rospy.loginfo("ArUco marker position or orientation not available.")
            return None

        self.num_frames += 1

        # Compute the rotation matrix from the camera to the ArUco marker
        camera_to_aruco_rotation = tf_trans.quaternion_matrix(aruco_camera_orient)[:3, :3]

        # Since robot and ArUco marker axes are aligned, use identity for aruco to robot rotation
        aruco_to_robot_rotation = np.eye(3)  # Identity matrix

        # Compute the combined rotation matrix from the camera frame to the robot frame
        self.rotation_matrix = np.dot(aruco_to_robot_rotation, camera_to_aruco_rotation.T)

        # Compute the translation vector from the camera frame to the robot frame
        self.translation_vector = aruco_robot_pos - np.dot(self.rotation_matrix, aruco_camera_pos)

        # Log the transformation details for debugging
        rospy.loginfo(f"Calculated Rotation Matrix: \n{self.rotation_matrix}")
        rospy.loginfo(f"Calculated Translation Vector: {self.translation_vector}")

    def align_pcd(self, pcd, publish=True):
        """
        Aligns the point cloud from the camera frame to the robot frame using stored transformation matrices.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to align.

        Returns:
            o3d.geometry.PointCloud: Transformed point cloud in the robot frame.
        """
        if self.rotation_matrix is None or self.translation_vector is None:
            rospy.logwarn("Transformation matrices not available. Skipping alignment.")
            return None

        # Apply the transformation to the point cloud
        if isinstance(pcd, o3d.geometry.PointCloud):
            to_transform = np.asarray(pcd.points)
        else:
            to_transform = np.asarray(pcd)

        try:
            # Do the transform on the (pcd) points
            transformed_points = np.dot(to_transform, self.rotation_matrix.T) + self.translation_vector
        except Exception as e:
            rospy.logerr(f"Error transforming point cloud: {e}")
            return None

        # Create a new point cloud with transformed points
        if isinstance(pcd, o3d.geometry.PointCloud):
            transformed_pcd = o3d.geometry.PointCloud()
            transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
            transformed_pcd.colors = pcd.colors
        else:
            transformed_pcd = transformed_points

        if publish:
            self.pcd_processor.convert_pcd_to_pc2msg(transformed_pcd, publisher=self.pcd_publisher)
        return transformed_pcd

    def _run(self):
        """Main loop to run the object detection and point cloud transformation pipeline."""

        # 1. Extract frames from the RealSense camera
        start_time = time.perf_counter()
        color_frame, depth_frame = self.get_frames()
        color_image = np.asanyarray(color_frame.get_data())
        end_time = time.perf_counter()
        timing_logger.log_timing("RS/Frame Extraction", end_time - start_time)

        # 2. Detect ArUco markers in the color image
        self.detect_aruco(color_image)

        if self.num_frames < 5:
            # 3. Calculate the transformation from the camera frame to the robot frame
            self.calculate_transformation(
                aruco_camera_pos=self.camera_pos,
                aruco_camera_orient=self.camera_orient,
                aruco_robot_pos=[0.0, 0.425, -0.018]
            )

        # 4. Detect light brown regions in the color image
        start_time = time.perf_counter()
        image_brown, bounding_box = self.detect_light_brown(color_image)
        end_time = time.perf_counter()
        timing_logger.log_timing("RS/2D Object Detection", end_time - start_time)

        # Overlay trajectory onto the 2D image
        image_with_trajectory = self.overlay_trajectory(image_brown)

        if self.debug:
            cv2.imshow('RS/Trajectory Overlay', image_with_trajectory)
            cv2.waitKey(1)

        if bounding_box:
            # 5. Filter the point cloud based on the detected bounding box
            start_time = time.perf_counter()
            filtered_pcd = self.filter_point_cloud_by_bounding_box(depth_frame, color_frame, bounding_box)
            end_time = time.perf_counter()
            timing_logger.log_timing("RS/Bounding Box-based Point Cloud Filtering", end_time - start_time)

            if filtered_pcd:
                # 6. Align the point cloud to the robot frame
                start_time = time.perf_counter()
                transformed_pcd = self.align_pcd(filtered_pcd)
                # o3d.io.write_point_cloud(
                #     "../gaussian_process_transportation/policy_transportation/apply_laplacian_editing/1feb_male_phantom.pcd", transformed_pcd)
                end_time = time.perf_counter()
                timing_logger.log_timing("RS/Point Cloud Transformation/Alignment", end_time - start_time)



if __name__ == '__main__':
    rospy.init_node('object_detection')
    detector = RealSenseCamera(debug=True)

    while not rospy.is_shutdown():
        detector._run()
