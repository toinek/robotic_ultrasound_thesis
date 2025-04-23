#!/usr/bin/env python
import numpy as np
import rospy
import threading
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import time

from std_msgs.msg import UInt64, Float32
from geometry_msgs.msg import PoseArray, Pose
from policy_transportation.transportation.laplacian_editing_transportation import \
    LaplacianEditingTransportation as Transport

from realsense.realsense_vision.pcd_helpers import PointCloudConverter, PointCloudProcessor, ICPMatcher
from realsense.realsense_vision.registration_utils import PointCloudUtility, TrajectoryUtility, VisualizeUtility
from realsense.data_analysis.computation_logger import timing_logger

class ApplyTransport:
    """
    Class for applying trajectory adaptation using Laplacian Editing Transportation (LTE).
    Subscribes to a calibrated point cloud and an initial trajectory, processes the point clouds,
    and publishes an adapted trajectory in response to changes in the environment.
    """
    def __init__(self, technique, first_pcd_path, debug=False):
        """
        Initializes the ApplyTransport node, sets up ROS subscribers and publishers, and loads the first point cloud.

        Args:
            technique: An instance of the trajectory transportation technique (LTE).
            first_pcd_path (str): Path to the initial point cloud file.
            debug (bool, optional): If True, enables debug mode. Defaults to False.
        """
        self.transport = technique
        self.debug = debug
        
        self.first_pcd = o3d.io.read_point_cloud(first_pcd_path)

        self.pcd_converter = PointCloudConverter(x_min=-5, x_max=5, y_min=-5, y_max=5, z_min=0.0, z_max=1.5)
        self.matcher = ICPMatcher()
        self.pcdutil = PointCloudUtility()
        self.trajutil = TrajectoryUtility()
        self.pcd_processor = PointCloudProcessor(self.pcd_converter, self.matcher, self.pcdutil)

        self.source_surface = None
        self.target_surface = None
        self.old_policy = None
        self.current_spline_index = 0

        self.ns = '/iiwa14'
        self.pcd_sub = rospy.Subscriber('transport/calibrated_pcd', pc2.PointCloud2, self.pcd_cb, queue_size=1)
        self.traj_sub = rospy.Subscriber(self.ns + '/kuka_trajectory_old', PoseArray, self.traj_cb)
        self.spline_index_sub = rospy.Subscriber('/CartesianImpedanceController/current_spline_index', UInt64,
                                                 self.spline_index_cb)

        self.new_traj_pub = rospy.Publisher(self.ns + '/kuka_trajectory_new', PoseArray, queue_size=10)
        self.force_control_pub = rospy.Publisher('transport/phantom_height', Float32, queue_size=10)  # New publisher

        self.traj_lock = threading.Lock()

    def spline_index_cb(self, spline_index_msg):
        """
        Callback function for receiving the current spline index from the Cartesian Impedance Controller.

        Args:
            spline_index_msg (std_msgs.UInt64): The received spline index message.
        """
        self.current_spline_index = spline_index_msg.data

    def pcd_cb(self, pcd_msg):
        """
        Callback function for handling incoming point cloud messages and applying trajectory adaptation.

        Args:
            pcd_msg (sensor_msgs.PointCloud2): The incoming point cloud message.
        """
        rospy.loginfo("Received a new point cloud.")
        start_time = time.perf_counter()
        if self.target_surface is None:
            # self.target_surface = self.first_pcd
            # o3d.visualization.draw_geometries([self.target_surface], window_name="First point cloud")
            self.target_surface = self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg)
            self.target_surface.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            return

        if self.old_policy is None:
            rospy.logerr("Registration callback waiting for the old policy...")
            return

        self.source_surface = self.target_surface
        self.target_surface = self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg)

        self.target_surface.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        end_time = time.perf_counter()
        timing_logger.log_timing("PCD_rigid/PCD Message Handling & Preprocessing", end_time - start_time)
        reordered_source, reordered_target = self.pcd_processor.process_point_clouds(self.source_surface, self.target_surface, self.debug)

        if reordered_source is not None:
            # Publish the top z-coordinate of the point cloud
            # self.publish_top_z_coordinate(reordered_target)
            # Apply transportation
            self.apply_transport(reordered_source, reordered_target)

    def publish_top_z_coordinate(self, point_cloud):
        """
        Calculates and publishes the top z-coordinate of the given point cloud.

        Args:
            point_cloud (open3d.geometry.PointCloud): The processed target point cloud.
        """
        points = np.asarray(point_cloud.points)
        if points.size == 0:
            rospy.logwarn("Point cloud is empty. Cannot calculate top z-coordinate.")
            return

        top_z = np.max(points[:, 2])  # Extract maximum z-coordinate
        rospy.loginfo(f"Top z-coordinate: {top_z}")
        self.force_control_pub.publish(top_z)

    def traj_cb(self, traj_msg):
        """
        Callback function for receiving the original trajectory.

        Args:
            traj_msg (geometry_msgs.PoseArray): The received trajectory message.
        """
        with self.traj_lock:
            self.old_policy = np.array([[pose.position.x, pose.position.y, pose.position.z, pose.orientation.x,
                                         pose.orientation.y, pose.orientation.z, pose.orientation.w] for pose in
                                        traj_msg.poses])
            rospy.loginfo("Received a new trajectory.")

    def apply_transport(self, reordered_source, reordered_target, keypoint=False):
        """
        Applies Laplacian Editing (LTE) to adapt the trajectory based on the changes in the point cloud.

        Args:
            reordered_source (open3d.geometry.PointCloud): The source point cloud of the trajectory.
            reordered_target (open3d.geometry.PointCloud): The target point cloud to transport the trajectory to.
            keypoint (bool, optional): Whether to use keypoint-based trajectory adaptation. Defaults to False.
        """

        with self.traj_lock:
            if self.old_policy is None or self.source_surface is None or self.target_surface is None:
                rospy.logerr("Waiting for the old policy and source surface...")
                return

            policy_to_use = self.old_policy.copy()
            current_spline = self.current_spline_index

        # Set a distance threshold for identifying the masks in the source and trajectory
        distance_threshold = 0.05

        # Apply LTE to the initial trajectory to obtain new waypoints
        LE_traj = TrajectoryUtility.apply_transport(reordered_source, reordered_target, policy_to_use, self.transport,
                                                    distance_threshold=distance_threshold,
                                                    current_spline_index=current_spline, keypoint=keypoint)

        # Apply surface normal orientation adjusting to the new trajectory
        # new_trajectory = self.trajutil.surface_normals(LE_traj, self.target_surface, distance_threshold=0)
        new_trajectory = LE_traj

        if self.debug:
            # Visualize the transportation
            # VisualizeUtility.visualize_simple(self.old_policy, new_trajectory, reordered_source, reordered_target, self.transport)
            correspondence_lines = PointCloudUtility.create_correspondence_lines(reordered_source, reordered_target)

            VisualizeUtility.visualize_all(reordered_source, reordered_target, self.target_surface, self.old_policy, new_trajectory, self.transport, correspondence_lines)
        rospy.loginfo("Transport sending new trajectory")
        self.publish_new_trajectory(new_trajectory)

    def publish_new_trajectory(self, new_trajectory):
        """
        Publishes the adapted trajectory after applying LTE.

        Args:
            new_trajectory (numpy.ndarray): The adapted trajectory as a numpy array.
        """
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = "base_link"

        for pose in new_trajectory:
            ros_pose = Pose()
            ros_pose.position.x, ros_pose.position.y, ros_pose.position.z = pose[:3]
            ros_pose.orientation.x, ros_pose.orientation.y, ros_pose.orientation.z, ros_pose.orientation.w = pose[3:]
            pose_array_msg.poses.append(ros_pose)

        self.new_traj_pub.publish(pose_array_msg)
        self.old_policy = new_trajectory

if __name__ == '__main__':
    rospy.init_node('policy_transportation', anonymous=True)
    technique = Transport()
    transport = ApplyTransport(technique, "20jan_static_pointcloud.pcd", debug=True)
    rospy.spin()