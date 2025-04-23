#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.signal import butter, filtfilt
import signal
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class TrajectoryAnalyzer:
    def __init__(self):
        rospy.init_node('trajectory_analyzer', anonymous=True)

        self.trajectory_sub = rospy.Subscriber('/CartesianImpedanceController/trajectory', PoseStamped, self.trajectory_callback)
        self.pose_sub = rospy.Subscriber('/CartesianImpedanceController/cartesian_pose', PoseStamped, self.pose_callback)

        self.trajectory = None  # Holds trajectory reference (PoseStamped)
        self.current_pose = None  # Holds the current pose (PoseStamped)

        self.time_stamps = []
        self.raw_positions = []
        self.position_deltas = []  # Tracking error
        self.jerk_magnitudes = []

        self.last_velocity = None
        self.last_acceleration = None

        rospy.loginfo("Trajectory Analyzer initialized. Waiting for messages...")

    def low_pass_filter(self, data, cutoff=5, fs=200, order=2):
        """Apply a Butterworth low-pass filter to smooth the data."""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)

    def trajectory_callback(self, msg):
        """Callback for trajectory topic. Stores the reference trajectory over time."""
        trajectory_point = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        if not hasattr(self, "trajectory_positions"):
            self.trajectory_positions = []  # Initialize storage
            self.trajectory_time_stamps = []

        self.trajectory_positions.append(trajectory_point)
        self.trajectory_time_stamps.append(rospy.Time.now().to_sec())

    def pose_callback(self, msg):
        """Callback for cartesian_pose topic. Stores actual positions over time."""
        current_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        current_time = rospy.Time.now().to_sec()

        if not hasattr(self, "raw_positions"):
            self.raw_positions = []
            self.time_stamps = []

        self.raw_positions.append(current_pose)
        self.time_stamps.append(current_time)


    def plot_results(self):
        """Plot the reference and actual positions over time as separate plots for LaTeX."""
        if len(self.raw_positions) == 0 or len(self.trajectory_positions) == 0:
            rospy.logwarn("No tracking data collected. Ensure topics are publishing correctly.")
            return

        # Convert lists to NumPy arrays
        actual_times = np.array(self.time_stamps)
        actual_positions = np.array(self.raw_positions)

        reference_times = np.array(self.trajectory_time_stamps)
        reference_positions = np.array(self.trajectory_positions)

        # Ensure timestamps are sorted (just in case)
        sort_idx = np.argsort(reference_times)
        reference_times = reference_times[sort_idx]
        reference_positions = reference_positions[sort_idx]

        # Interpolate reference positions to match actual timestamps
        interp_x = interp1d(reference_times, reference_positions[:, 0], kind='linear', fill_value="extrapolate")
        interp_y = interp1d(reference_times, reference_positions[:, 1], kind='linear', fill_value="extrapolate")
        interp_z = interp1d(reference_times, reference_positions[:, 2], kind='linear', fill_value="extrapolate")

        interpolated_reference_positions = np.vstack(
            (interp_x(actual_times), interp_y(actual_times), interp_z(actual_times))).T

        # Set larger font sizes for LaTeX
        plt.rcParams.update({'font.size': 14})  # Adjusts all text in the plots

        # Plot X Position
        plt.figure(figsize=(10, 5))
        plt.plot(actual_times - actual_times[0], actual_positions[:, 0], label="Actual X", color="blue")
        plt.plot(actual_times - actual_times[0], interpolated_reference_positions[:, 0],
                 label="Reference X", color="red", linestyle="dashed")
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("X Position (m)", fontsize=16)
        plt.legend(fontsize=14)
        plt.grid()
        plt.title("X Position Over Time", fontsize=18)
        # plt.savefig("x_position_plot.pdf", bbox_inches='tight')  # Save as PDF for LaTeX
        plt.show()

        # Plot Y Position
        plt.figure(figsize=(10, 5))
        plt.plot(actual_times - actual_times[0], actual_positions[:, 1], label="Actual Y", color="blue")
        plt.plot(actual_times - actual_times[0], interpolated_reference_positions[:, 1],
                 label="Reference Y", color="red", linestyle="dashed")
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Y Position (m)", fontsize=16)
        plt.legend(fontsize=14)
        plt.grid()
        plt.title("Y Position Over Time", fontsize=18)
        # plt.savefig("y_position_plot.pdf", bbox_inches='tight')  # Save as PDF for LaTeX
        plt.show()

        # Plot Z Position
        plt.figure(figsize=(10, 5))
        plt.plot(actual_times - actual_times[0], actual_positions[:, 2], label="Actual Z", color="blue")
        plt.plot(actual_times - actual_times[0], interpolated_reference_positions[:, 2],
                 label="Reference Z", color="red", linestyle="dashed")
        plt.xlabel("Time (s)", fontsize=16)
        plt.ylabel("Z Position (m)", fontsize=16)
        plt.legend(fontsize=14)
        plt.grid()
        plt.title("Z Position Over Time", fontsize=18)
        # plt.savefig("z_position_plot.pdf", bbox_inches='tight')  # Save as PDF for LaTeX
        plt.show()

    def shutdown_handler(self, signum, frame):
        rospy.loginfo("Shutting down...")
        self.plot_results()
        rospy.signal_shutdown("User interrupted.")
        sys.exit(0)

if __name__ == '__main__':
    analyzer = TrajectoryAnalyzer()

    # Handle SIGINT (Ctrl+C) gracefully
    signal.signal(signal.SIGINT, analyzer.shutdown_handler)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
