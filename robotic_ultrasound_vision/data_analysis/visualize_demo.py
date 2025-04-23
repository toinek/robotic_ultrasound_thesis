import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_trajectory(csv_file):
    """
    Load trajectory points from a CSV file.
    The CSV is expected to have columns: timestamp, position_x, position_y, position_z, orientation_x, orientation_y, orientation_z, orientation_w.
    """
    data = pd.read_csv(csv_file)
    if {'position_x', 'position_y', 'position_z', 'orientation_x', 'orientation_y', 'orientation_z',
        'orientation_w'}.issubset(data.columns):
        trajectory_points = data[['position_x', 'position_y', 'position_z']].to_numpy()[::10]
        orientations = data[['orientation_x', 'orientation_y', 'orientation_z', 'orientation_w']].to_numpy()[::10]
    else:
        raise ValueError(
            "CSV file must contain 'position_x', 'position_y', 'position_z', 'orientation_x', 'orientation_y', 'orientation_z', and 'orientation_w' columns.")

    return trajectory_points, orientations

def create_arrow_with_orientation(position, quaternion, scale=0.1):
    """
    Create an arrow mesh with orientation defined by a quaternion.

    :param position: Position of the arrow (3D point)
    :param quaternion: Orientation of the arrow (quaternion)
    :param scale: Scaling factor for the arrow size
    :return: An Open3D TriangleMesh arrow
    """
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.005 * scale,
        cone_radius=0.01 * scale,
        cylinder_height=0.1 * scale,
        cone_height=0.02 * scale
    )
    arrow.paint_uniform_color([0, 1, 0])  # Green color for orientation arrows
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    arrow.rotate(rotation_matrix, center=(0, 0, 0))
    arrow.translate(position)
    return arrow

def align_trajectories(demo_df, replay_dfs):
    demo_start = demo_df[['position_x', 'position_y', 'position_z']].iloc[0].values  # Starting point of the demo
    demo_end = demo_df[['position_x', 'position_y', 'position_z']].iloc[-1].values  # Ending point of the demo
    aligned_replays = []

    for replay_df in replay_dfs:
        # Find the closest point in the replay to the demo start
        start_distances = np.sqrt(
            (replay_df['position_x'] - demo_start[0]) ** 2 +
            (replay_df['position_y'] - demo_start[1]) ** 2 +
            (replay_df['position_z'] - demo_start[2]) ** 2
        )
        start_index = start_distances.idxmin()

        # Find the closest point in the replay to the demo end
        end_distances = np.sqrt(
            (replay_df['position_x'] - demo_end[0]) ** 2 +
            (replay_df['position_y'] - demo_end[1]) ** 2 +
            (replay_df['position_z'] - demo_end[2]) ** 2
        )
        end_index = end_distances.idxmin()

        # Slice the replay to start and end at the matching points
        replay_df = replay_df.iloc[start_index:end_index + 1].reset_index(drop=True)
        aligned_replays.append(replay_df)

    return aligned_replays

def visualize_trajectories_and_pcd_with_alignment(demo_file, replay_files, pcd_file):
    """
    Visualize a demo trajectory, aligned replays, and the point cloud.

    :param demo_file: CSV file path containing the demo trajectory data
    :param replay_files: List of CSV file paths containing replay trajectory data
    :param pcd_file: Path to the .pcd file
    """
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(pcd_file)

    if point_cloud.is_empty():
        raise ValueError("The point cloud file is empty or invalid.")

    geometries = [point_cloud]

    # Load the demo trajectory
    demo_df = pd.read_csv(demo_file)

    if not {'position_x', 'position_y', 'position_z', 'orientation_x', 'orientation_y', 'orientation_z',
            'orientation_w'}.issubset(demo_df.columns):
        raise ValueError("Demo CSV file must contain the required columns.")

    # Load replay trajectories
    replay_dfs = [pd.read_csv(replay_file) for replay_file in replay_files]

    # Align the replays to the demo
    aligned_replays = align_trajectories(demo_df, replay_dfs)

    # Visualize the demo trajectory
    demo_points, demo_orientations = load_trajectory(demo_file)
    demo_pcd = o3d.geometry.PointCloud()
    demo_pcd.points = o3d.utility.Vector3dVector(demo_points)
    demo_pcd.paint_uniform_color([1, 0, 0])  # Red for demo trajectory
    geometries.append(demo_pcd)

    # Add arrows for demo orientation
    demo_arrows = [create_arrow_with_orientation(pos, quat) for pos, quat in zip(demo_points, demo_orientations)]
    geometries.extend(demo_arrows)

    # Visualize aligned replays
    for idx, replay_df in enumerate(aligned_replays):
        replay_points = replay_df[['position_x', 'position_y', 'position_z']].to_numpy()
        replay_pcd = o3d.geometry.PointCloud()
        replay_pcd.points = o3d.utility.Vector3dVector(replay_points)
        color = [0, 1, 0] if idx == 0 else [0, 0, 1] if idx == 1 else [1, 1, 0]  # Different colors for each replay
        replay_pcd.paint_uniform_color(color)
        geometries.append(replay_pcd)

    # Visualize all geometries
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Aligned Trajectories and Point Cloud",
        zoom=0.8,
        front=[0, -1, 0],
        lookat=[0, 0, 0],
        up=[0, 0, 1]
    )


def interpolate_trajectory(reference_timestamps, replay_df):
    """
    Interpolate the replay trajectory to match the reference timestamps.

    :param reference_timestamps: Timestamps from the demo trajectory
    :param replay_df: DataFrame containing the replay trajectory
    :return: Interpolated positions (numpy array)
    """
    replay_timestamps = replay_df['timestamp'].to_numpy()
    positions = replay_df[['position_x', 'position_y', 'position_z']].to_numpy()

    # Interpolate positions for each coordinate
    interpolated_positions = np.array([
        np.interp(reference_timestamps, replay_timestamps, positions[:, i])
        for i in range(3)
    ]).T  # Transpose to match shape (N, 3)

    return interpolated_positions


def calculate_tracking_error_over_time(demo_file, replay_files):
    """
    Calculate the tracking error between the demo trajectory and replay trajectories over time.

    :param demo_file: Path to the demo CSV file
    :param replay_files: List of paths to the replay CSV files
    :return: Dictionary containing errors over time for each replay
    """
    # Load the demo trajectory
    demo_df = pd.read_csv(demo_file)
    demo_timestamps = demo_df['timestamp'].to_numpy()
    demo_positions = demo_df[['position_x', 'position_y', 'position_z']].to_numpy()

    # Normalize the demo timestamps to start at 0
    demo_timestamps -= demo_timestamps[0]

    errors_over_time = {}

    for replay_file in replay_files:
        # Load the replay trajectory
        replay_df = pd.read_csv(replay_file)

        # Interpolate replay trajectory to match demo timestamps
        interpolated_positions = interpolate_trajectory(demo_timestamps, replay_df)

        # Calculate the Euclidean distance (tracking error) for each point
        errors = np.linalg.norm(interpolated_positions - demo_positions, axis=1)
        errors_over_time[replay_file] = {
            'timestamps': demo_timestamps,  # Use normalized timestamps
            'errors': errors
        }

    return errors_over_time


def plot_tracking_errors(errors_over_time):
    """
    Plot the tracking errors over time for all replay files.

    :param errors_over_time: Dictionary containing errors over time for each replay
    """
    plt.figure(figsize=(10, 6))

    for replay_file, data in errors_over_time.items():
        plt.plot(data['timestamps'], data['errors'], label=f"{replay_file.split('/')[-1]}")

    plt.title("Tracking Errors Over Time (Normalized Time)")
    plt.xlabel("Time (s)")
    plt.ylabel("Tracking Error (m)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def normalize_time(timestamps):
    """
    Normalize time to start at 0.

    :param timestamps: Original timestamps
    :return: Normalized timestamps
    """
    return timestamps - timestamps[0]

def plot_trajectories_over_time(demo_file, replay_files):
    """
    Plot the demo and aligned replay trajectories over time with matching timescales.

    :param demo_file: Path to the demo CSV file
    :param replay_files: List of paths to the replay CSV files
    """
    # Load demo trajectory
    demo_df = pd.read_csv(demo_file)
    demo_timestamps = demo_df['timestamp'].to_numpy()
    demo_timestamps = normalize_time(demo_timestamps)  # Normalize time to start at 0
    demo_positions = demo_df[['position_x', 'position_y', 'position_z']].to_numpy()

    # Align the replay trajectories with bounds on alignment distance
    def align_with_bound(demo_point, replay_points, bound):
        distances = np.linalg.norm(replay_points - demo_point, axis=1)
        within_bound_indices = np.where(distances <= bound)[0]
        if len(within_bound_indices) > 0:
            return within_bound_indices[0]
        else:
            return None

    def align_trajectories_with_bound(demo_df, replay_dfs, bound):
        demo_start = demo_df[['position_x', 'position_y', 'position_z']].iloc[0].values
        demo_end = demo_df[['position_x', 'position_y', 'position_z']].iloc[-1].values
        aligned_replays = []

        for replay_df in replay_dfs:
            replay_points = replay_df[['position_x', 'position_y', 'position_z']].to_numpy()

            # Find the first point within the bound for the start
            start_index = align_with_bound(demo_start, replay_points, bound)

            # Find the first point within the bound for the end
            end_index = align_with_bound(demo_end, replay_points, bound)

            if start_index is not None and end_index is not None and start_index < end_index:
                aligned_replays.append(replay_df.iloc[start_index:end_index + 1].reset_index(drop=True))

        return aligned_replays

    # Define alignment bound (e.g., 0.05 meters)
    alignment_bound = 0.03
    replay_dfs = [pd.read_csv(replay_file) for replay_file in replay_files]
    aligned_replays = align_trajectories_with_bound(demo_df, replay_dfs, alignment_bound)

    plt.figure(figsize=(12, 8))

    # Plot the demo trajectory
    for i, label in enumerate(['X', 'Y', 'Z']):
        plt.plot(demo_timestamps, demo_positions[:, i], label=f"Demo {label}", linestyle='--', linewidth=2)

    # Plot the aligned replay trajectories with scaled timescales
    demo_duration = demo_timestamps[-1] - demo_timestamps[0]

    for idx, replay_df in enumerate(aligned_replays):
        replay_timestamps = normalize_time(replay_df['timestamp'].to_numpy())
        replay_duration = replay_timestamps[-1] - replay_timestamps[0]

        # Scale replay timestamps to match the demo timescale
        scaled_replay_timestamps = replay_timestamps * (demo_duration / replay_duration)

        replay_positions = replay_df[['position_x', 'position_y', 'position_z']].to_numpy()

        for i, label in enumerate(['X', 'Y', 'Z']):
            plt.plot(scaled_replay_timestamps, replay_positions[:, i], label=f"Replay {idx + 1} {label}")

    plt.title("Trajectories Over Time (Aligned and Normalized Time)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()



# File paths
demo_file = "/home/toine/catkin_ws/src/realsense/4feb_dense_demo.csv"
replay_files = [
    "/home/toine/catkin_ws/src/traj_csvs/5feb_arm_nonrigid_2.csv",
    "/home/toine/catkin_ws/src/traj_csvs/5feb_arm_nonrigid_3.csv",
    "/home/toine/catkin_ws/src/traj_csvs/5feb_arm_nonrigid_4.csv",
    "/home/toine/catkin_ws/src/traj_csvs/5feb_arm_nonrigid_5.csv",
]
pcd_file = "/policy_transportation/apply_laplacian_editing/4feb_dense_arm.pcd"

# Visualize the trajectories and point cloud with alignment
visualize_trajectories_and_pcd_with_alignment(demo_file, replay_files, pcd_file)

