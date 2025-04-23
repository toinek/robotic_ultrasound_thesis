#!/usr/bin/env python
import rosbag
import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import time
import copy

from std_msgs.msg import Header

from realsense.data_analysis.computation_logger import timing_logger

# RosbagReader: Handles reading messages from a ROS bag file
class RosbagReader:
    def __init__(self, bag_path, topic):
        self.bag_path = bag_path
        self.topic = topic
        self.bag = rosbag.Bag(bag_path)
        self.generator = self.bag.read_messages(topics=topic)

    def get_next_message(self):
        try:
            topic, msg, t = next(self.generator)
            return msg
        except StopIteration:
            return None

    def close(self):
        self.bag.close()


# PointCloudConverter: Converts ROS PointCloud2 messages into Open3D point clouds with filtering
class PointCloudConverter:
    def __init__(self, x_min=None, x_max=None, y_min=None, y_max=None, z_min=None, z_max=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    @staticmethod
    def unpack_rgb_uint32(rgb_uint32):
        r = (rgb_uint32 & 0x00FF0000) >> 16
        g = (rgb_uint32 & 0x0000FF00) >> 8
        b = (rgb_uint32 & 0x000000FF)
        return [r / 255.0, g / 255.0, b / 255.0]

    @staticmethod
    def pack_rgb_uint32(pcd):
        points_with_rgb = []
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]

            # Convert the color components to integer [0, 255] range
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)

            # Pack as a 32-bit unsigned integer (uint32)
            rgb_uint32 = (r << 16) | (g << 8) | b

            # Append as [x, y, z, rgb_uint32]
            points_with_rgb.append([x, y, z, rgb_uint32])

        return points_with_rgb

    def convert_pc2msg_to_pcd(self, msg):
        points = []
        colors = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z = point[0], point[1], point[2]
            rgb = int(point[3])  # Cast directly to int for consistent decoding
            if self.z_min is not None:
                if self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max and self.z_min <= z <= self.z_max:
                    points.append([x, y, z])
                    colors.append(self.unpack_rgb_uint32(rgb))
            else:
                points.append([x, y, z])
                colors.append(self.unpack_rgb_uint32(rgb))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        return pcd

    def convert_pcd_to_pc2msg(self, pcd, publisher=None):
        # Ensure that the point cloud has colors
        if not pcd.has_colors():
            rospy.logwarn("Point cloud does not have color information. Publishing without RGB data.")
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'camera_link'
            pcd_msg = pc2.create_cloud_xyz32(header, np.asarray(pcd.points))
            if publisher:
                publisher.publish(pcd_msg)
            return pcd_msg

        # Pack the points with RGB values
        points_with_rgb = self.pack_rgb_uint32(pcd)

        # Define the fields for the PointCloud2 message (x, y, z, rgb)
        fields = [
            pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
            pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
            pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
            pc2.PointField('rgb', 12, pc2.PointField.UINT32, 1),
        ]

        # Create the PointCloud2 message with the header, fields, and data
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_link'
        pcd_msg = pc2.create_cloud(header, fields, points_with_rgb)

        if publisher:
            # Publish the PointCloud2 message
            publisher.publish(pcd_msg)

        return pcd_msg

# PointCloudProcessor: Processes point clouds with various algorithms
class PointCloudProcessor:
    def __init__(self, pcd_converter, matcher, pcdutil):
        self.pcd_converter = pcd_converter
        self.matcher = matcher
        self.pcdutil = pcdutil

    def process_point_clouds(self, source_surface, target_surface, debug=False):
        voxel_size = 0.01

        # Step 1: Downsampling and Chamfer Distance Calculation
        start_time = time.perf_counter()
        source_down = self.pcdutil.downsample_and_denoise(source_surface, voxel_size=voxel_size)
        target_down = self.pcdutil.downsample_and_denoise(target_surface, voxel_size=voxel_size)

        # source_down.points = o3d.utility.Vector3dVector(np.asarray(source_down.points) + [0, 0, 0.12])
        # target_down.points = o3d.utility.Vector3dVector(np.asarray(target_down.points) + [0, 0, 0.12])

        source_copy = copy.deepcopy(source_down)
        target_copy = copy.deepcopy(target_down)

        chamfer_dist = self.pcdutil.chamfer_distance(source_down.points, target_down.points)

        end_time = time.perf_counter()
        timing_logger.log_timing("PCD_rigid/Downsampling & Chamfer Distance", end_time - start_time)

        rospy.loginfo(f"Chamfer distance: {chamfer_dist}")
        if chamfer_dist < 0.025:
            rospy.loginfo(f"Source and target are close enough. Skipping...")
            return None, None
        if debug:
            o3d.visualization.draw_geometries([source_down, target_down], window_name="Downsampled point clouds")

        # Step 2: Coarse Alignment
        start_time = time.perf_counter()
        coarse_transformation = self.pcdutil.coarse_alignment(source_down, target_down, voxel_size)
        source_copy.transform(coarse_transformation)
        end_time = time.perf_counter()

        # Log computation time using the centralized logger
        timing_logger.log_timing("PCD_rigid/Coarse Alignment", end_time - start_time)

        # Step 3: ICP Fine Alignment
        start_time = time.perf_counter()
        transformation_matrix = self.matcher.run_icp(source_copy, target_down).transformation
        source_copy.transform(transformation_matrix)
        end_time = time.perf_counter()
        timing_logger.log_timing("PCD_rigid/ICP Fine Alignment", end_time - start_time)

        if debug:
            o3d.visualization.draw_geometries([source_copy, target_down], window_name="ICP aligned point clouds")

        # Step 4: Reordering Point Clouds Based on Correspondence
        start_time = time.perf_counter()
        reordered_source, reordered_target = self.matcher.reorder_point_clouds_based_on_correspondences(source_down,
                                                                                                        target_copy)
        end_time = time.perf_counter()
        timing_logger.log_timing("PCD_rigid/Reordering Point Clouds", end_time - start_time)

        if debug:
            o3d.visualization.draw_geometries([reordered_source, reordered_target],
                                              window_name="Reordered point clouds")

        return reordered_source, reordered_target

    def process_keypoints(self, target_surface, pcd_msg, last_update_time, distance_threshold=0.1, time_threshold=0.5):
        """
        Processes point clouds using sparse keypoints instead of dense matching. Skips transformation
        if keypoints are too similar or updated too frequently.

        Args:
            source_surface (open3d.geometry.PointCloud): The source surface point cloud.
            target_surface (open3d.geometry.PointCloud): The target surface point cloud.
            pcd_msg (sensor_msgs.PointCloud2): The ROS point cloud message.
            last_update_time (float): Timestamp of the last processed point cloud.
            distance_threshold (float, optional): Minimum distance for triggering transportation. Defaults to 0.1.
            time_threshold (float, optional): Minimum time gap for updating. Defaults to 0.5 seconds.

        Returns:
            tuple: (Updated source surface, updated target surface) or (None, None) if skipped.
        """
        target_pcd = self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg)

        if len(target_surface.points) != 4 or len(target_pcd.points) != 4:
            rospy.logwarn("Not enough keypoints detected. Skipping transportation.")
            return None, None

        distance = np.sum(np.linalg.norm(np.array(target_surface.points) - np.array(target_pcd.points), axis=1))
        current_time = time.time()

        if distance < distance_threshold or current_time - last_update_time < time_threshold:
            return None, None

        return target_surface, target_pcd

# ICPMatcher: Performs ICP alignment between two point clouds
class ICPMatcher:
    def __init__(self):
        self.registration_result = None

    def run_icp(self, source_pcd, target_pcd, threshold=0.04):
        transformation = np.eye(4)
        self.registration_result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        print(f'information on the result: {self.registration_result}')
        return self.registration_result

    def reorder_point_clouds_based_on_correspondences(self, source_pcd, target_pcd):
        correspondences = np.asarray(self.registration_result.correspondence_set)

        source_points = np.asarray(source_pcd.points)
        source_colors = np.asarray(source_pcd.colors)

        target_points = np.asarray(target_pcd.points)
        target_colors = np.asarray(target_pcd.colors)


        reordered_source_points = []
        reordered_source_colors = []
        reordered_target_points = []
        reordered_target_colors = []

        for source_idx, target_idx in correspondences:
            reordered_source_points.append(source_points[source_idx])
            reordered_source_colors.append(source_colors[source_idx])
            reordered_target_points.append(target_points[target_idx])
            reordered_target_colors.append(target_colors[target_idx])

        reordered_source = o3d.geometry.PointCloud()
        reordered_source.points = o3d.utility.Vector3dVector(reordered_source_points)
        reordered_source.colors = o3d.utility.Vector3dVector(reordered_source_colors)

        reordered_target = o3d.geometry.PointCloud()
        reordered_target.points = o3d.utility.Vector3dVector(reordered_target_points)
        reordered_target.colors = o3d.utility.Vector3dVector(reordered_target_colors)

        return reordered_source, reordered_target
