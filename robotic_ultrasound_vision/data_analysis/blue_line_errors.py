import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import re
from scipy.linalg import svd
import matplotlib.pyplot as plt

# Path to the directory containing CSV files
CSV_DIR = "error_csvs_18feb"


def load_csv(file_path):
    """Loads a CSV file and returns the data as a NumPy array."""
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        return data[:, 1:4]  # Extract only x, y, z columns
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_initialdemo_csv(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        return data[:, 0:3]  # Extract only x, y, z columns
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compute_arc_length(trajectory):
    """Computes cumulative arc length of a trajectory."""
    deltas = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    arc_length = np.concatenate(([0], np.cumsum(deltas)))
    return arc_length


def resample_trajectory(trajectory, num_points):
    """Resamples a trajectory to have `num_points` based on arc length."""
    arc_length = compute_arc_length(trajectory)
    target_arc_length = np.linspace(0, arc_length[-1], num_points)

    # Interpolate X, Y, Z separately along the arc length
    interp_x = interp1d(arc_length, trajectory[:, 0], kind='linear', fill_value="extrapolate")
    interp_y = interp1d(arc_length, trajectory[:, 1], kind='linear', fill_value="extrapolate")
    interp_z = interp1d(arc_length, trajectory[:, 2], kind='linear', fill_value="extrapolate")

    resampled_traj = np.vstack((interp_x(target_arc_length),
                                interp_y(target_arc_length),
                                interp_z(target_arc_length))).T
    return resampled_traj


def compute_similarity_metrics(initial, adapted, raw_initial):
    """Computes LTE similarity metrics: E2 (Rotational), E3 (Scaling-Invariant)."""

    n = min(initial.shape[0], adapted.shape[0])  # Ensure same number of points
    initial = initial[:n]
    adapted = adapted[:n]
    raw_initial = raw_initial[:n]  # Ensure same cropping

    # Compute local displacements (before resampling)
    raw_init_diff = np.linalg.norm(np.diff(raw_initial, axis=0), axis=1)
    init_diff = np.linalg.norm(np.diff(initial, axis=0), axis=1)
    adapt_diff = np.linalg.norm(np.diff(adapted, axis=0), axis=1)

    # Compute E2: Rotationally Adapted Deformation Measure
    E2 = 0
    E3 = 0
    for i in range(n - 1):  # Iterate through trajectory segments
        # Solve for optimal rotation using SVD
        H = np.outer(initial[i + 1] - initial[i], adapted[i + 1] - adapted[i])
        U, S, Vt = svd(H)
        R_opt = Vt.T @ U.T

        # Apply rotation to initial displacement
        rotated_diff = R_opt @ (initial[i + 1] - initial[i]).reshape(3, 1)

        # Compute E2 deviation
        E2 += np.linalg.norm(rotated_diff.flatten() - (adapted[i + 1] - adapted[i])) ** 2

        # Compute E3: Scaling-Invariant Similarity using raw segment lengths
        scale_factor = raw_init_diff[i] / init_diff[i] if init_diff[i] > 0 else 1
        E3 += np.linalg.norm(scale_factor * rotated_diff.flatten() - (adapted[i + 1] - adapted[i])) ** 2

    return E2, E3


def process_trajectories(blue_line_data, traj_data):
    """
    Filters and resamples the blue line and transported trajectory
    to ensure consistency for RMSE calculation and similarity metrics.
    """
    if blue_line_data is None or traj_data is None:
        return None, None

    # Determine key Z-values from the blue line
    max_z_blue = np.max(blue_line_data[:, 2])
    min_z_blue = np.min(blue_line_data[:, 2])
    last_blue_z = blue_line_data[-1, 2]

    # Compute Z differences to detect sharp increase
    z_diff = np.diff(traj_data[:, 2], prepend=traj_data[0, 2])

    # Filter transported trajectory:
    traj_data_filtered = traj_data[
        (traj_data[:, 2] >= min_z_blue) &
        (traj_data[:, 2] <= max_z_blue) &
        ((traj_data[:, 2] <= last_blue_z) | (z_diff <= 0))
        ]

    if traj_data_filtered.size == 0:
        return None, None

    # Resample the blue line to match the transported trajectory's number of points
    blue_line_resampled = resample_trajectory(blue_line_data, traj_data_filtered.shape[0])

    return blue_line_resampled, traj_data_filtered

def plot_initial_vs_transported(initial, transported, title):
    """Plots the initial trajectory vs. the transported trajectory."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.plot(initial[:, 0], initial[:, 1], initial[:, 2], 'g-', label="Initial Demonstration")
    ax.plot(transported[:, 0], transported[:, 1], transported[:, 2], 'r-', label="Transported Trajectory")

    ax.legend()
    ax.set_title(title)
    plt.show()

def calculate_similarity_metrics(initial_file, blue_line_file, traj_file, e2_list, e3_list, step_list, rmse_list):
    """Calculates LTE similarity metrics and RMSE for processed trajectories."""
    initial_data = load_initialdemo_csv(initial_file)
    blue_line_data = load_csv(blue_line_file)
    traj_data = load_csv(traj_file)

    if initial_data is None or blue_line_data is None or traj_data is None:
        print(f"Skipping metrics calculation for {blue_line_file} and {traj_file} due to invalid data.")
        return None

    # Resample initial trajectory to match transported trajectory's number of points
    initial_resampled = resample_trajectory(initial_data, traj_data.shape[0])
    blue_line_resampled, traj_data_filtered = process_trajectories(blue_line_data, traj_data)

    if blue_line_resampled is None or traj_data_filtered is None:
        return None

    # Compute RMSE (Blue Line vs. Transported)
    rmse = np.sqrt(mean_squared_error(blue_line_resampled, traj_data_filtered))

    # Compute similarity metrics using both resampled and raw initial trajectories
    E2, E3 = compute_similarity_metrics(initial_resampled, traj_data_filtered, initial_data)

    # # Compute E4: Sum of Squared Cartesian Differences (Transported vs. Blue Line)
    # E4 = np.sum(np.linalg.norm(blue_line_resampled - traj_data_filtered, axis=1) ** 2)

    print(f"Metrics for {os.path.basename(traj_file)}:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  E2 (Rotational Adaptation): {E2:.6f}")
    print(f"  E3 (Scaling-Invariant): {E3:.6f}")

    # Plot Initial vs Transported Trajectory
    # plot_initial_vs_transported(initial_resampled, traj_data_filtered,
    #                             f"Initial vs Transported - {os.path.basename(traj_file)}")

    # Store values for drift analysis
    rmse_list.append(rmse)
    e2_list.append(E2)
    e3_list.append(E3)
    step_list.append(step_list[-1] + 1 if step_list else 1)  # Start from 1

    return rmse, E2, E3

def find_csv_pairs():
    """Finds all matching CSV files for initial demo, blue line, and transported trajectory."""
    blue_line_files = sorted(glob.glob(os.path.join(CSV_DIR, "18feb_blue_line_*.csv")), key=natural_sort_key)
    traj_files = sorted(glob.glob(os.path.join(CSV_DIR, "18feb_transported_trajectory_*.csv")), key=natural_sort_key)
    # initial_demo = "/home/toine/catkin_ws/src/realsense/18feb_demo.csv"
    initial_demo = "/home/toine/catkin_ws/src/realsense/data_analysis/filtered_traj.csv"

    pairs = []
    for blue_file in blue_line_files:
        suffix = re.search(r'_(\d+)\.csv$', blue_file)
        if suffix:
            suffix = suffix.group(1)
            matching_traj = os.path.join(CSV_DIR, f"18feb_transported_trajectory_{suffix}.csv")
            if matching_traj in traj_files:
                pairs.append((initial_demo, blue_file, matching_traj))

    return pairs

# def calculate_similarity_metrics(previous_file, blue_line_file, traj_file, e2_list, e3_list, step_list):
#     """Calculates LTE similarity metrics and RMSE, comparing the current adaptation to the previous one,
#        and stores E2 and E3 values for drift analysis.
#     """
#     prev_data = load_csv(previous_file)  # Load previous adaptation (or initial demo for first step)
#     blue_line_data = load_csv(blue_line_file)  # Ground truth reference
#     traj_data = load_csv(traj_file)  # Current transported trajectory
#
#     if prev_data is None or blue_line_data is None or traj_data is None:
#         print(f"Skipping metrics calculation for {blue_line_file} and {traj_file} due to invalid data.")
#         return None
#
#     # Resample previous trajectory to match the transported trajectory's number of points
#     prev_resampled = resample_trajectory(prev_data, traj_data.shape[0])
#     blue_line_resampled, traj_data_filtered = process_trajectories(blue_line_data, traj_data)
#
#     if blue_line_resampled is None or traj_data_filtered is None:
#         return None
#
#     # Compute RMSE (Blue Line vs. Transported)
#     rmse = np.sqrt(mean_squared_error(blue_line_resampled, traj_data_filtered))
#
#     # Compute similarity metrics using the previous adaptation instead of the initial demonstration
#     E2, E3 = compute_similarity_metrics(prev_resampled, traj_data_filtered, prev_data)
#
#     print(f"Metrics for {os.path.basename(traj_file)} (compared to previous adaptation):")
#     print(f"  RMSE: {rmse:.6f}")
#     print(f"  E2 (Rotational Adaptation): {E2:.6f}")
#     print(f"  E3 (Scaling-Invariant): {E3:.6f}")
#
#     # Store values for drift analysis
#     e2_list.append(E2)
#     e3_list.append(E3)
#     step_list.append(step_list[-1] + 1 if step_list else 1)  # Start from 1
#
#     return rmse, E2, E3
#
#
# def find_csv_pairs():
#     """Finds and orders CSV files to compare each adaptation to the previous one."""
#     traj_files = sorted(glob.glob(os.path.join(CSV_DIR, "18feb_transported_trajectory_*.csv")), key=natural_sort_key)
#     blue_line_files = sorted(glob.glob(os.path.join(CSV_DIR, "18feb_blue_line_*.csv")), key=natural_sort_key)
#
#     # The first adaptation is compared to the initial demo
#     initial_demo = "/home/toine/catkin_ws/src/realsense/data_analysis/filtered_traj.csv"
#
#     pairs = []
#     previous_file = initial_demo  # Start with the initial demonstration
#
#     for traj_file, blue_file in zip(traj_files, blue_line_files):
#         pairs.append((previous_file, blue_file, traj_file))  # Compare each trajectory to the previous one
#         previous_file = traj_file  # Update previous trajectory for next comparison
#
#     return pairs


def natural_sort_key(filename):
    """Extracts numerical suffix from filename for proper sorting."""
    match = re.search(r'_(\d+)\.csv$', filename)
    return int(match.group(1)) if match else float('inf')


def plot_drift(step_list, rmse_list):
    """Plots RMSE over adaptation steps with a horizontal mean line for readability in paper format."""
    plt.clf()  # Clears previous figures
    plt.close('all')  # Ensures no stale plots interfere

    fig, ax = plt.subplots(figsize=(10, 6))

    mean_rmse = np.mean(rmse_list)

    ax.plot(step_list, rmse_list, marker='o', linestyle='-', color='r', label="RMSE")
    ax.axhline(y=mean_rmse, color='black', linestyle='--', linewidth=2, label="Mean RMSE")  # Mean RMSE line

    # Increase font sizes
    ax.set_xlabel("Adaptation Step", fontsize=35)
    ax.set_ylabel("RMSE (m)", fontsize=35)
    # ax.set_title("RMSE Deviation between  Adaptation Steps", fontsize=28)
    ax.legend(fontsize=28)
    ax.grid(True)

    # Increase tick font size
    ax.tick_params(axis='both', which='major', labelsize=26)

    plt.tight_layout()
    plt.show(block=True)  # Ensures figure is fully rendered
if __name__ == "__main__":
    csv_pairs = find_csv_pairs()

    if not csv_pairs:
        print("No matching CSV files found.")
    else:
        print(f"Found {len(csv_pairs)} matching CSV sets.")

        # Lists to store drift values
        rmse_list = []
        e2_list = []
        e3_list = []
        step_list = []

        for previous_file, blue_file, traj_file in csv_pairs:
            try:
                calculate_similarity_metrics(previous_file, blue_file, traj_file, e2_list, e3_list, step_list, rmse_list)
            except Exception as e:
                print(f"Error processing {blue_file} and {traj_file}: {e}")

        # Plot drift over time
        print(f'Mean RMSE: {np.mean(rmse_list)}, Std RMSE: {np.std(rmse_list)}')
        plot_drift(step_list, rmse_list)

