#!/usr/bin/env python

import rospy
import rosbag
import pandas as pd
import numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose
from realsense.realsense_vision.registration_utils import PointCloudUtility
from realsense.realsense_vision.pcd_helpers import PointCloudConverter

class TrajectoryRelay:
    def __init__(self, bag_path, csv_file_path):
        self.csv_file_path = csv_file_path
        self.bag_path = bag_path
        self.use_new_trajectory = False  # Flag to switch to the new trajectory
        self.new_trajectory = None  # Variable to store the new trajectory

        # Initialize ROS node
        rospy.init_node('trajectory_relay', anonymous=True)

        # Create publishers
        self.trajectory_pub = rospy.Publisher('/iiwa14/kuka_trajectory_old', PoseArray, queue_size=10)
        self.calibrated_pcd_pub = rospy.Publisher('/transport/calibrated_pcd', PointCloud2, queue_size=10)

        # Subscribe to the kuka_trajectory_new topic
        self.trajectory_new_sub = rospy.Subscriber(
            '/iiwa14/kuka_trajectory_new', PoseArray, self.trajectory_new_callback
        )

        # Load the trajectory from CSV
        self.original_trajectory = self.load_trajectory_from_csv()

        # Initialize variables for rosbag reading
        self.bag = None
        self.pcd_msgs = None
        self.last_pcd_msg = None  # Store last known PointCloud2 message

        # Open rosbag once at the start
        self.open_rosbag()

        # Create timers for publishing data
        # self.trajectory_timer = rospy.Timer(rospy.Duration(0.1), self.publish_trajectory_callback)
        self.pcd_timer = rospy.Timer(rospy.Duration(0.1), self.publish_pcd_callback)

        self.converter = PointCloudConverter(x_min=-5, x_max=5, y_min=-5, y_max=5, z_min=-0.0, z_max = 1.5)
        self.index = 0

    def load_trajectory_from_csv(self):
        """Load trajectory data from a CSV file and convert it to a PoseArray message."""
        df = pd.read_csv(self.csv_file_path)

        pose_array = PoseArray()
        pose_array.header.frame_id = "world"
        pose_array.header.stamp = rospy.Time.now()

        for i, row in df.iterrows():
            if i%10 == 0:
                pose = Pose()
                pose.position.x = row['position_x']
                pose.position.y = row['position_y'] #+ 0.2
                pose.position.z = row['position_z'] #+ 0.20  # Adjust z-coordinate
                pose.orientation.x = row['orientation_x']
                pose.orientation.y = row['orientation_y']
                pose.orientation.z = row['orientation_z']
                pose.orientation.w = row['orientation_w']
                pose_array.poses.append(pose)

        return pose_array

    def trajectory_new_callback(self, msg):
        """Callback to handle new trajectory messages."""
        rospy.loginfo("Received a new trajectory. Switching to it.")
        self.new_trajectory = msg
        self.use_new_trajectory = True

    def publish_trajectory_callback(self, event):
        """Periodic callback to publish the appropriate trajectory."""
        if self.use_new_trajectory and self.new_trajectory:
            # Publish the new trajectory
            self.trajectory_pub.publish(self.new_trajectory)
        else:
            # Publish the original trajectory from the CSV
            self.trajectory_pub.publish(self.original_trajectory)

    def open_rosbag(self):
        """Open the rosbag and prepare for reading messages only once."""
        try:
            self.bag = rosbag.Bag(self.bag_path, 'r')  # Open the rosbag once
            self.pcd_msgs = self.bag.read_messages(topics=['/transport/calibrated_pcd'])  # Create an iterator
            self.last_pcd_msg = None
            rospy.loginfo("Rosbag opened successfully.")
        except Exception as e:
            rospy.logerr(f"Error opening rosbag: {e}")
            self.bag = None
            self.pcd_msgs = None

    def publish_pcd_callback(self, event):
        """Periodic callback to publish PointCloud2 data from the rosbag or the last known message."""
        if self.bag is None or self.pcd_msgs is None:
            rospy.logwarn("Rosbag is not open. Attempting to open...")
            self.open_rosbag()
            return  # Wait until the next cycle

        try:
            # Get the next message from the rosbag iterator
            topic, msg, t = next(self.pcd_msgs)
            if self.index % 2 != 0:
                rospy.loginfo(f"Applying breathing on {self.index}")
                pcd = self.converter.convert_pc2msg_to_pcd(msg)
                amp = np.random.uniform(0.0, 0.12)
                breathing_pcd = PointCloudUtility.apply_breathing_effect(pcd, amplitude=amp)
                msg = self.converter.convert_pcd_to_pc2msg(breathing_pcd)

            self.calibrated_pcd_pub.publish(msg)
            self.last_pcd_msg = msg  # Store last message
            self.index += 1

        except StopIteration:
            # When all messages are read, just keep publishing the last known message
            if self.last_pcd_msg:
                rospy.logwarn("Rosbag is exhausted. Re-publishing the last known PointCloud2 message.")
                self.calibrated_pcd_pub.publish(self.last_pcd_msg)

        except Exception as e:
            rospy.logerr(f"Error reading from rosbag: {e}")


if __name__ == '__main__':
    try:
        # Define the paths to your rosbag and CSV file
        bag_path = '/calibrated_pcd30dec.bag'
        csv_file_path = '/home/toine/catkin_ws/src/traj_csvs/20jan_static_demo.csv'

        relay = TrajectoryRelay(bag_path, csv_file_path)
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass
