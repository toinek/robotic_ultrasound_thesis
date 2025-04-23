import open3d as o3d
import numpy as np
import pandas as pd
import copy

from realsense.realsense_vision.registration_utils import PointCloudUtility, TrajectoryUtility, VisualizeUtility
from realsense.realsense_vision.pcd_helpers import ICPMatcher
from src.gaussian_process_transportation.policy_transportation.transportation.laplacian_editing_transportation import \
    LaplacianEditingTransportation as Transport

def filter_by_z(pcd, z_threshold):
    """
    Filters out points in the point cloud where the z-value is below the given threshold.

    :param pcd: open3d.geometry.PointCloud
    :param z_threshold: float, the minimum allowed z-value
    :return: open3d.geometry.PointCloud, filtered point cloud
    """
    points = np.asarray(pcd.points)
    mask = points[:, 2] >= z_threshold  # Keep points with z >= z_threshold
    filtered_pcd = pcd.select_by_index(np.where(mask)[0])
    return filtered_pcd

if __name__ == '__main__':
    pcdutil = PointCloudUtility()
    trajutil = TrajectoryUtility()

    # Load point clouds
    file_path1 = "20jan_static_pointcloud.pcd"
    file_path2 = "male_pointcloud.pcd"

    original_target = o3d.io.read_point_cloud(file_path2)
    pcd1 = o3d.io.read_point_cloud(file_path1)
    pcd2 = o3d.io.read_point_cloud(file_path2)

    # Filter out points with z < 0.03
    pcd1 = filter_by_z(pcd1, 0.03)
    pcd2 = filter_by_z(pcd2, -0.03)

    print(f'shapes: {np.asarray(pcd1.points).shape}, {np.asarray(pcd2.points).shape}')

    # Load the trajectory
    csv_file_path = '20jan_static_demo.csv'
    df = pd.read_csv(csv_file_path)
    traj = df[['position_x', 'position_y', 'position_z', 'orientation_x', 'orientation_y', 'orientation_z',
                'orientation_w']].values[::10]
    old_traj = copy.deepcopy(traj)

    # Visualize the point clouds and trajectory
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(old_traj[:, :3])
    traj_colors = np.tile([1.0, 0.0, 0.0], (old_traj[:, :3].shape[0], 1))  # Red color
    traj_pcd.colors = o3d.utility.Vector3dVector(traj_colors)

    # Combine the trajectory point cloud with the original point cloud
    combined_pcd = pcd1 + traj_pcd
    o3d.visualization.draw_geometries([combined_pcd, pcd2], window_name="Original Point Clouds")

    # Compute normals for the point clouds
    pcd1.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd2.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Downsample and denoise the point clouds
    voxel_size = 0.02
    pcd1 = pcdutil.downsample_and_denoise(pcd1, voxel_size=voxel_size)
    pcd2 = pcdutil.downsample_and_denoise(pcd2, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Downsampled and Denoised Point Clouds")
    pcd2_copy = copy.deepcopy(pcd2)
    matched_pcd1, matched_pcd2, tf_param = pcdutil.apply_cpd(pcd1, pcd2, tf_type_name="nonrigid", maxiter=100, tol=1e-5)

    # Do the matching
    matcher = ICPMatcher()
    result = matcher.run_icp(matched_pcd1, matched_pcd2)
    reordered_source, reordered_target = matcher.reorder_point_clouds_based_on_correspondences(pcd1, pcd2_copy)
    # Create the correspondence lines
    correspondence_lines = pcdutil.create_correspondence_lines(reordered_source, reordered_target)

    # Visualize reordered source, target, and the correspondence lines
    o3d.visualization.draw_geometries([reordered_source, reordered_target, correspondence_lines],
                                      window_name="Reordered Source, Target, and Correspondences")

    # Initialize the transportation object
    transport = Transport()
    new_traj = trajutil.apply_transport(reordered_source, reordered_target, traj, transport)

    VisualizeUtility.visualize_all(reordered_source, reordered_target, original_target, old_traj, new_traj, transport, correspondence_lines)