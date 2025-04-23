import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D


class DataAnalyzer:
    def __init__(self, demo_file, replay_files):
        self.demo_file = demo_file
        self.replay_files = replay_files
        self.demo_df = self.load_trajectory(demo_file)
        self.replay_dfs = [self.load_trajectory(f) for f in replay_files]
        self.aligned_replays = self.align_trajectories()

    def load_trajectory(self, csv_file):
        data = pd.read_csv(csv_file)
        if {'position_x', 'position_y', 'position_z'}.issubset(data.columns):
            return data[['timestamp', 'position_x', 'position_y', 'position_z']]
        else:
            raise ValueError("CSV file must contain 'position_x', 'position_y', and 'position_z' columns.")

    def align_trajectories(self):
        demo_start = self.demo_df.iloc[0, 1:].values
        demo_end = self.demo_df.iloc[-1, 1:].values
        aligned_replays = []

        for replay_df in self.replay_dfs:
            start_distances = np.linalg.norm(replay_df.iloc[:, 1:].values - demo_start, axis=1)
            start_index = np.argmin(start_distances)
            end_index = replay_df.shape[0] - 1  # Keep full trajectory after alignment
            replay_df = replay_df.iloc[start_index:end_index + 1].reset_index(drop=True)
            aligned_replays.append(replay_df)

        return aligned_replays

    def plot_aligned_trajectories_3d(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(self.demo_df['position_x'], self.demo_df['position_y'], self.demo_df['position_z'],
                label='Demo', linestyle='--', color='red', linewidth=2)

        ax.scatter(self.demo_df['position_x'].iloc[0], self.demo_df['position_y'].iloc[0],
                   self.demo_df['position_z'].iloc[0],
                   color='green', s=100, label='Demo Start', marker='o')
        ax.scatter(self.demo_df['position_x'].iloc[-1], self.demo_df['position_y'].iloc[-1],
                   self.demo_df['position_z'].iloc[-1],
                   color='blue', s=100, label='Demo End', marker='o')

        for idx, replay_df in enumerate(self.aligned_replays):
            ax.scatter(replay_df['position_x'].iloc[0], replay_df['position_y'].iloc[0],
                       replay_df['position_z'].iloc[0],
                       color='green', s=50, marker='^', label=f'Adaptation Start' if idx == 0 else None)
            ax.scatter(replay_df['position_x'].iloc[-1], replay_df['position_y'].iloc[-1],
                       replay_df['position_z'].iloc[-1],
                       color='blue', s=50, marker='^', label=f'Adaptation End' if idx == 0 else None)
            ax.plot(replay_df['position_x'], replay_df['position_y'], replay_df['position_z'],
                    label=f'Adaptation {idx + 1}')

        ax.set_title("T20 Rigid Adaptated Trajectories", fontsize=15)
        ax.set_xlabel("Position X (m)", fontsize=15)
        ax.set_ylabel("Position Y (m)", fontsize=15)
        ax.set_zlabel("Position Z (m)", fontsize=15)
        ax.legend(fontsize=12)
        plt.show()

    def calculate_jerk(self, data, visualize=False):
        # Extract columns
        timestamps = data["timestamp"].values
        positions = data[["position_x", "position_y", "position_z"]].values

        # Check time intervals
        dt = np.diff(timestamps)
        if not np.allclose(dt, dt.mean(), atol=0.001):
            print(
                f"Inconsistent time intervals detected. Mean dt: {dt.mean():.6f}, Min dt: {dt.min():.6f}, Max dt: {dt.max():.6f}")

        # Calculate speed (magnitude of velocity between consecutive points)
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt

        # Threshold for movement detection (e.g., speed below a threshold indicates no significant movement)
        movement_threshold = 0.04  # Adjust as needed
        moving_indices = np.where(speeds > movement_threshold)[0]

        # Retain only the active movement portion
        if len(moving_indices) > 0:
            start_idx = moving_indices[0]
            end_idx = moving_indices[-1] + 1  # Include the last moving point
            timestamps = timestamps[start_idx:end_idx]
            positions = positions[start_idx:end_idx]
        else:
            print("No significant movement detected.")
            exit()

        # Resample timestamps to uniform intervals (e.g., 0.005 seconds for 200 Hz)
        uniform_timestamps = np.arange(timestamps[0], timestamps[-1], 0.005)
        interp_func = interp1d(timestamps, positions, axis=0, kind="linear", fill_value="extrapolate")
        uniform_positions = interp_func(uniform_timestamps)

        # Replace timestamps and positions with the resampled data
        timestamps = uniform_timestamps
        positions = uniform_positions

        # Calculate the new dt (should now be consistent)
        dt = np.diff(timestamps)

        # Apply Gaussian smoothing to positions to reduce noise
        smoothed_positions = gaussian_filter1d(positions, sigma=3, axis=0)

        # Calculate velocity
        velocities = np.diff(smoothed_positions, axis=0) / dt[:, None]

        # Calculate acceleration
        accelerations = np.diff(velocities, axis=0) / dt[1:, None]

        # Calculate jerk
        jerks = np.diff(accelerations, axis=0) / dt[2:, None]

        # Resultant jerk (magnitude)
        resultant_jerk = np.linalg.norm(jerks, axis=1)

        # Clip extreme jerk values (optional, for stability)
        resultant_jerk = np.clip(resultant_jerk, 0, np.percentile(resultant_jerk, 99))

        # Compute mean and max jerk
        mean_jerk = np.mean(resultant_jerk)
        max_jerk = np.max(resultant_jerk)

        # Print results
        print(f"Mean Jerk: {mean_jerk:.6f} m/s³")
        print(f"Max Jerk: {max_jerk:.6f} m/s³")
        if visualize:
            # Visualize smoothed positions
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps, smoothed_positions[:, 0], label="Position X")
            plt.plot(timestamps, smoothed_positions[:, 1], label="Position Y")
            plt.plot(timestamps, smoothed_positions[:, 2], label="Position Z")
            plt.legend()
            plt.title("Smoothed Positions")
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.show()

            # Visualize velocity
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps[1:], velocities[:, 0], label="Velocity X")
            plt.plot(timestamps[1:], velocities[:, 1], label="Velocity Y")
            plt.plot(timestamps[1:], velocities[:, 2], label="Velocity Z")
            plt.legend()
            plt.title("Velocity")
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.show()

            # Visualize acceleration
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps[2:], accelerations[:, 0], label="Acceleration X")
            plt.plot(timestamps[2:], accelerations[:, 1], label="Acceleration Y")
            plt.plot(timestamps[2:], accelerations[:, 2], label="Acceleration Z")
            plt.legend(fontsize=15)  # Increase legend font size
            plt.title("Acceleration", fontsize=15)  # Increase title font size
            plt.xlabel("Time (s)", fontsize=15)  # Increase x-axis label font size
            plt.ylabel("Acceleration (m/s²)", fontsize=15)  # Increase y-axis label
            plt.xticks(fontsize=12)  # Increase x-axis tick labels font size
            plt.yticks(fontsize=12)  # Increase y-axis tick labels font size
            plt.show()

            # Visualize jerk
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps[3:], resultant_jerk, label="Resultant Jerk")
            plt.title("Jerk")
            plt.xlabel("Time (s)")
            plt.ylabel("Jerk (m/s³)")
            plt.legend()
            plt.show()

    def analyze(self):
        print("Analyzing demo trajectory...")
        self.calculate_jerk(self.demo_df, visualize=True)

        for idx, replay_df in enumerate(self.aligned_replays):
            print(f"Analyzing replay {idx + 1}...")
            self.calculate_jerk(replay_df, visualize=True)

        self.plot_aligned_trajectories_3d()


# Example usage:
# demo_file = "/home/toine/catkin_ws/src/realsense/4feb_dense_demo.csv"
# replay_files = [
#     "/home/toine/catkin_ws/src/traj_csvs/5feb_arm_nonrigid_2.csv",
#     "/home/toine/catkin_ws/src/traj_csvs/5feb_arm_nonrigid_3.csv",
#     "/home/toine/catkin_ws/src/traj_csvs/5feb_arm_nonrigid_4.csv",
#     "/home/toine/catkin_ws/src/traj_csvs/5feb_arm_nonrigid_5.csv"
# ]

# demo_file = '/home/toine/catkin_ws/src/realsense/4feb_male_demo.csv'
#
# replay_files = [
#     '/home/toine/catkin_ws/src/traj_csvs/14feb_generalize_3.csv',
#     '/home/toine/catkin_ws/src/traj_csvs/14feb_generalize_4.csv',
#     '/home/toine/catkin_ws/src/traj_csvs/14feb_generalize_5.csv',
#     '/home/toine/catkin_ws/src/traj_csvs/14feb_generalize_6.csv',
#     '/home/toine/catkin_ws/src/traj_csvs/14feb_generalize_7.csv'
# ]

demo_file = "/home/toine/catkin_ws/src/traj_csvs/20jan_static_demo.csv"
replay_files = [
    "/home/toine/catkin_ws/src/traj_csvs/20jan_rigid_T20_1.csv",
    "/home/toine/catkin_ws/src/traj_csvs/20jan_rigid_T20_2.csv",
    "/home/toine/catkin_ws/src/traj_csvs/20jan_rigid_T20_3.csv",
    "/home/toine/catkin_ws/src/traj_csvs/20jan_rigid_T20_4.csv",
    "/home/toine/catkin_ws/src/traj_csvs/20jan_rigid_T20_5.csv"
]

# demo_file = "/home/toine/catkin_ws/src/realsense/18feb_demo.csv"
#
# replay_files =[
#     "/home/toine/catkin_ws/src/traj_csvs/25feb_nonrigid_1.csv",
#     "/home/toine/catkin_ws/src/traj_csvs/25feb_nonrigid_2.csv"
# ]
analyzer = DataAnalyzer(demo_file, replay_files)
analyzer.analyze()