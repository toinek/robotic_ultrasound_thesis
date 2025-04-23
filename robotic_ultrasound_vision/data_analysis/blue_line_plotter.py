import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import re

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

def process_trajectories(blue_line_data, traj_data):
    """
    Filters and resamples the blue line and transported trajectory
    to ensure consistency for both RMSE calculation and visualization.
    """
    if blue_line_data is None or traj_data is None:
        return None, None

    # Determine key Z-values from the blue line
    max_z_blue = np.max(blue_line_data[:, 2])
    min_z_blue = np.min(blue_line_data[:, 2])
    last_blue_z = blue_line_data[-1, 2]  # Last Z-value of the blue line

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

def calculate_rmse(blue_line_file, traj_file):
    """Calculates RMSE using the processed (filtered & resampled) trajectories."""
    blue_line_data = load_csv(blue_line_file)
    traj_data = load_csv(traj_file)

    blue_line_resampled, traj_data_filtered = process_trajectories(blue_line_data, traj_data)

    if blue_line_resampled is None or traj_data_filtered is None:
        print(f"Skipping RMSE calculation for {blue_line_file} and {traj_file} due to invalid data.")
        return None

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(blue_line_resampled, traj_data_filtered))
    print(f"RMSE for {os.path.basename(blue_line_file)} and {os.path.basename(traj_file)}: {rmse:.6f}")
    return rmse

def plot_trajectories(blue_line_file, traj_file):
    """Plots a single pair of blue line and transported trajectory using consistent processing."""
    blue_line_data = load_csv(blue_line_file)
    traj_data = load_csv(traj_file)

    blue_line_resampled, traj_data_filtered = process_trajectories(blue_line_data, traj_data)

    if blue_line_resampled is None or traj_data_filtered is None:
        print(f"Skipping plot for {blue_line_file} and {traj_file} due to invalid data.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X", fontsize=25)
    ax.set_ylabel("Y", fontsize=25)
    ax.set_zlabel("Z", fontsize=25)
    #ax.set_title(f"Trajectory Pair: {os.path.basename(blue_line_file)} & {os.path.basename(traj_file)}")

    # Plot resampled blue line
    ax.plot(blue_line_resampled[:, 0], blue_line_resampled[:, 1], blue_line_resampled[:, 2], 'b-', label="Ground Truth Blue Line")

    # Plot filtered transported trajectory
    ax.plot(traj_data_filtered[:, 0], traj_data_filtered[:, 1], traj_data_filtered[:, 2], 'r-', label="LTE Transported Trajectory")

    ax.legend(fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust label size for x and y axes
    ax.tick_params(axis='z', which='major', labelsize=18)  # Adjust label size for z axis
    plt.show(block=True)  # Wait until figure is closed

def natural_sort_key(filename):
    """Extracts numerical suffix from filename for proper sorting."""
    match = re.search(r'_(\d+)\.csv$', filename)  # Extract number before ".csv"
    return int(match.group(1)) if match else float('inf')  # Sort numerically

def find_csv_pairs():
    """Finds all matching blue line and transported trajectory CSV files in 'error_csvs_18feb' directory."""
    blue_line_files = sorted(glob.glob(os.path.join(CSV_DIR, "18feb_blue_line_*.csv")), key=natural_sort_key)
    traj_files = sorted(glob.glob(os.path.join(CSV_DIR, "18feb_transported_trajectory_*.csv")), key=natural_sort_key)

    pairs = []
    for blue_file in blue_line_files:
        suffix = re.search(r'_(\d+)\.csv$', blue_file)  # Extract number suffix
        if suffix:
            suffix = suffix.group(1)  # Extract matched number
            matching_traj = os.path.join(CSV_DIR, f"18feb_transported_trajectory_{suffix}.csv")
            if matching_traj in traj_files:
                pairs.append((blue_file, matching_traj))

    return pairs

if __name__ == "__main__":

    ### Uncomment the following block to process all CSV pairs in 'error_csvs_18feb' directory
    # csv_pairs = find_csv_pairs()
    #
    # if not csv_pairs:
    #     print("No matching blue line and transported trajectory CSV files found in 'error_csvs_18feb'.")
    # else:
    #     print(f"Found {len(csv_pairs)} matching CSV pairs in 'error_csvs_18feb'.")
    #     total_rmse = 0
    #     for blue_file, traj_file in csv_pairs:
    #         file_num = re.search(r'_(\d+)\.csv$', blue_file).group(1)
    #         # if file_num in ["36", "37","38","39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50"]:
    #         try:
    #             rmse = calculate_rmse(blue_file, traj_file)
    #             print((f"RMSE for file {file_num}: {rmse:.6f}"))
    #             total_rmse += rmse
    #             plot_trajectories(blue_file, traj_file)
    #         except Exception as e:
    #             print(f"Error processing {blue_file} and {traj_file}: {e}")
    #             continue
    #     print("\nAverage RMSE for all pairs:", total_rmse / len(csv_pairs))


    ### Uncomment the following block to calculate the initial error in the given demo
    initial_blue_lines = sorted(glob.glob(os.path.join("initial_blueline_18feb", "blue_line_*.csv")), key=natural_sort_key)
    initial_red_line = os.path.join("initial_blueline_18feb", "18feb_demo.csv")
    total_rmse = 0
    rmse_list = []
    step = 0
    for blue_file in initial_blue_lines:
        try:
            rmse = calculate_rmse(blue_file, initial_red_line)
            total_rmse += rmse
            # plot_trajectories(blue_file, initial_red_line)
            if rmse < 0.03:
                step += 1
                rmse_list.append(rmse)
        except Exception as e:
            print(f"Error processing {blue_file} and {initial_red_line}: {e}")
            continue
    print("\nAverage RMSE for initial demo:", total_rmse / len(initial_blue_lines))

    average_rmse = np.sum(rmse_list) / len(rmse_list)
    print("\nAverage RMSE for initial demo:", average_rmse)
    print(f'mean RMSE: {np.mean(rmse_list)}')
    print(f'std dev: {np.std(rmse_list)}')
    # plot RMSE per step
    x = np.arange(1, step + 1)
    plt.plot(x, rmse_list, marker='o', linestyle='-')
    plt.axhline(y=np.mean(rmse_list), color='r', linestyle='--', label="Mean RMSE")
    plt.xlabel("Step")
    plt.ylabel("RMSE")
    plt.title("RMSE per Step")
    plt.legend()
    plt.show()

