import copy
import time
import open3d as o3d
import probreg
import numpy as np

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from realsense.data_analysis.computation_logger import timing_logger
# from gaussian_process_transportation.policy_transportation.transportation.gaussian_process_transportation import GaussianProcessTransportation

class PointCloudUtility:
    def __init__(self):
        pass

    @staticmethod
    def filter_outer_points(pcd, radius_threshold):
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        distances = np.linalg.norm(np.asarray(pcd.points) - centroid, axis=1)
        outer_points = distances > radius_threshold
        outer_pcd = o3d.geometry.PointCloud()
        outer_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[outer_points])
        outer_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[outer_points])
        return outer_pcd

    @staticmethod
    def apply_cpd(pcd1, pcd2, tf_type_name="rigid", maxiter=40, tol=1e-6):
        tf_param = probreg.cpd.registration_cpd(pcd1, pcd2, tf_type_name=tf_type_name, maxiter=maxiter, tol=tol)
        transformed_points = tf_param.transformation.transform(np.asarray(pcd1.points))
        pcd1_transformed = o3d.geometry.PointCloud()
        pcd1_transformed.points = o3d.utility.Vector3dVector(transformed_points)
        pcd1_transformed.colors = o3d.utility.Vector3dVector(np.asarray(pcd1.colors))
        return pcd1_transformed, pcd2, tf_param

    @staticmethod
    def reorder_point_clouds_based_on_cpd(source_pcd, target_pcd, transformation):
        transformed_source_points = transformation.transform(np.asarray(source_pcd.points))
        target_points = np.asarray(target_pcd.points)
        target_kdtree = cKDTree(target_points)
        matched_target_indices = set()
        reordered_source_points = []
        reordered_source_colors = []
        reordered_target_points = []
        reordered_target_colors = []

        for i, transformed_src_point in enumerate(transformed_source_points):
            distance, nearest_target_idx = target_kdtree.query(transformed_src_point)
            if nearest_target_idx in matched_target_indices:
                continue

            matched_target_indices.add(nearest_target_idx)
            reordered_source_points.append(np.asarray(source_pcd.points)[i])
            reordered_source_colors.append(np.asarray(source_pcd.colors)[i])
            reordered_target_points.append(target_points[nearest_target_idx])
            reordered_target_colors.append(np.asarray(target_pcd.colors)[nearest_target_idx])

        reordered_source = o3d.geometry.PointCloud()
        reordered_source.points = o3d.utility.Vector3dVector(reordered_source_points)
        reordered_source.colors = o3d.utility.Vector3dVector(reordered_source_colors)

        reordered_target = o3d.geometry.PointCloud()
        reordered_target.points = o3d.utility.Vector3dVector(reordered_target_points)
        reordered_target.colors = o3d.utility.Vector3dVector(reordered_target_colors)
        return reordered_source, reordered_target

    @staticmethod
    def center_point_cloud(pcd):
        """Centers the point cloud by translating it to the origin."""
        points = np.asarray(pcd.points)
        centroid = points.mean(axis=0)
        pcd.translate(-centroid)
        return pcd

    @staticmethod
    # Compute initial transformation using RANSAC
    def coarse_alignment(source, target, voxel_size, translate=False):
        # Step 1: Compute centroids
        source_centroid = np.asarray(source.points).mean(axis=0)
        target_centroid = np.asarray(target.points).mean(axis=0)

        # Step 2: Translate target to the source's centroid
        target.translate(source_centroid - target_centroid)
        translation_vector = source_centroid - target_centroid

        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100)
        )
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 8,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            #estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            ransac_n=4,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 8)  # Distance check
                      ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
        )
        R = result_ransac.transformation[:3, :3]
        if np.linalg.det(R) < 0:
            print("Flipped transformation detected. Applying correction.")
            result_ransac.transformation[:3, :3] *= -1  # Flip back
        if translate:
            return result_ransac.transformation, translation_vector
        return result_ransac.transformation

    @staticmethod
    def downsample_and_denoise(point_cloud, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
        downsampled_pc = point_cloud.voxel_down_sample(voxel_size)
        cl, ind = downsampled_pc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        denoised_pc = downsampled_pc.select_by_index(ind)
        return denoised_pc

    @staticmethod
    def create_correspondence_lines(source_pcd, target_pcd, color=[0, 1, 0], intensity=0.5):
        points_combined = np.vstack((np.asarray(source_pcd.points), np.asarray(target_pcd.points)))
        lines = [[i, i + len(source_pcd.points)] for i in range(len(source_pcd.points))]#[::30]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points_combined)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # paint the colours see through but in a greenish tone
        line_set.colors = o3d.utility.Vector3dVector(
            np.tile(color, (len(lines), 1)) * intensity)  # Reduce brightness

        return line_set

    @staticmethod
    def apply_breathing_effect(pcd, amplitude=0.4, sigma=0.5):
        points = np.asarray(pcd.points)
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        distances = np.sqrt((points[:, 0] - x_center) ** 2 + (points[:, 1] - y_center) ** 2)
        max_distance = distances.max()
        normalized_distances = distances / max_distance
        gaussian_weights = np.exp(-normalized_distances ** 2 / (2 * sigma ** 2))
        points[:, 2] += amplitude * gaussian_weights
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(points)
        transformed_pcd.colors = pcd.colors
        return transformed_pcd

    @staticmethod
    def chamfer_distance(pcd1_points, pcd2_points):
        tree1 = cKDTree(pcd1_points)
        tree2 = cKDTree(pcd2_points)
        d1 = np.mean(tree1.query(pcd2_points)[0])
        d2 = np.mean(tree2.query(pcd1_points)[0])
        return d1 + d2

    @staticmethod
    def calculate_point_to_surface_metrics(source, target):
        """
        Calculate mean and maximum point-to-surface distances.
        """
        distances = target.compute_point_cloud_distance(source)
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)
        return mean_dist, max_dist

    @staticmethod
    def calculate_overlap_ratio(source, target, threshold=0.02):
        """
        Calculate the overlap ratio between source and target clouds.
        """
        source_points = np.asarray(source.points)
        target_kdtree = o3d.geometry.KDTreeFlann(target)

        overlap_count = 0
        for point in source_points:
            _, _, distances = target_kdtree.search_knn_vector_3d(point, 1)
            if distances[0] <= threshold:
                overlap_count += 1

        overlap_ratio = overlap_count / len(source_points)
        return overlap_ratio

class TrajectoryUtility:
    def __init__(self):
        pass

    @staticmethod
    def apply_transport(reordered_source, reordered_target, traj, transport, distance_threshold=0.05, current_spline_index=0, keypoint=False):
        old_policy = copy.deepcopy(traj)

        delta_trajectory = np.diff(old_policy[:, :3], axis=0)
        delta_trajectory = np.vstack([delta_trajectory, np.zeros((1, delta_trajectory.shape[1]))])

        transport.source_distribution = np.asarray(reordered_source.points)
        transport.target_distribution = np.asarray(reordered_target.points)

        if type(transport).__name__ == "LaplacianEditingTransportation":
            transport.training_traj = traj
            transport.training_delta = delta_trajectory
            # 1. Measure time for the fitting process
            start_fit_time = time.perf_counter()
            transport.fit_transportation(do_scale=True, do_rotation=True, distance_threshold=distance_threshold,
                                         current_spline_index=current_spline_index, keypoint=keypoint)

            end_fit_time = time.perf_counter()
            timing_logger.log_timing("LTE/Fitting", end_fit_time - start_fit_time)
        else:
            print('Fitting other type Process Transportation')
            transport.training_traj = traj[:, :3]
            transport.training_delta = delta_trajectory[:, :3]
            transport.fit_transportation()

        # 2. Measure time for applying the transportation
        start_apply_time = time.perf_counter()
        transport.apply_transportation()
        end_apply_time = time.perf_counter()

        timing_logger.log_timing("LTE/Trajectory Update", end_apply_time - start_apply_time)

        new_trajectory = transport.training_traj
        if type(transport).__name__ != "LaplacianEditingTransportation":
            new_trajectory = np.hstack([new_trajectory, old_policy[:, 3:]])
        else:
            new_trajectory[:, 3:] = old_policy[:, 3:]
        return new_trajectory

    @staticmethod
    def align_trajectory_to_point_cloud(traj, pcd, distance_threshold):
        first_point, last_point = False, False
        point_cloud_points = np.asarray(pcd.points)
        kdtree = cKDTree(point_cloud_points)
        updated_traj_pos = traj.copy()
        for i, coord in enumerate(traj[:, :3]):
            distance, index = kdtree.query(coord)
            if not first_point:
                first_point_index = i
                first_point_distance = distance
            if distance <= distance_threshold:
                first_point = True
                last_point_index = i
                last_point_distance = distance
                updated_traj_pos[i, 2] = point_cloud_points[index, 2]
        updated_traj_pos[:first_point_index, 2] -= first_point_distance
        updated_traj_pos[last_point_index + 1:, 2] -= last_point_distance

        return updated_traj_pos

    @staticmethod
    def create_arrow_with_orientation(position, quaternion, scale=0.05):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.01 * scale,
            cone_radius=0.02 * scale,
            cylinder_height=0.05 * scale,
            cone_height=0.1 * scale
        )
        arrow.paint_uniform_color([0, 1, 0])
        r = R.from_quat(quaternion)
        rotation_matrix = r.as_matrix()
        arrow.rotate(rotation_matrix, center=np.array([0, 0, 0]))
        arrow.translate(position)
        return arrow

    @staticmethod
    def visualize_trajectory_with_orientation(traj):
        geometries = []
        for pose in traj:
            position = pose[:3]
            quaternion = pose[3:]
            arrow = TrajectoryUtility.create_arrow_with_orientation(position, quaternion)
            geometries.append(arrow)
        return geometries

    @staticmethod
    def surface_normals(traj, pcd, distance_threshold):
        if not pcd.has_normals():
            raise ValueError("Point cloud must have precomputed normals. Use pcd.estimate_normals() to calculate them.")

        pcd_points = np.asarray(pcd.points)
        pcd_normals = np.asarray(pcd.normals)
        updated_traj = copy.deepcopy(traj)

        # Create a KDTree for the point cloud and query the nearest points for each trajectory point
        kdtree = cKDTree(pcd_points)
        distances, indices = kdtree.query(traj[:, :3])
        mask_traj = np.where(distances < distance_threshold)[0]  # Indices meeting the threshold
        skipped_num = len(traj) - len(mask_traj)

        for i in mask_traj:
            normal = pcd_normals[indices[i]]

            if normal[2] < 0:
                normal = -normal

            z_axis = -normal / np.linalg.norm(normal)
            ref_vector = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            x_axis = np.cross(ref_vector, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
            quaternion = R.from_matrix(rotation_matrix).as_quat()
            updated_traj[i, 3:] = quaternion

        print(f'Skipped percentage of points: {skipped_num / len(traj) * 100:.2f}%')
        return updated_traj

class VisualizeUtility:
    def __init__(self):
        pass

    @staticmethod
    def visualize_simple(old_traj, new_traj, source, target, transport, axes_size=0.5, axes_origin=[0, 0, 0]):
        # Increase z height by 0.12 of source and target
        # all source points blue, all target points yellow
        # source.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (source.points.__len__(), 1)))
        # target.colors = o3d.utility.Vector3dVector(np.tile([1, 1, 0], (target.points.__len__(), 1)))

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size, origin=axes_origin)

        old_traj_pcd = o3d.geometry.PointCloud()
        old_traj_pcd.points = o3d.utility.Vector3dVector(old_traj[:, :3])
        old_traj_colors = np.tile([1.0, 0.0, 0.0], (old_traj[:, :3].shape[0], 1))  # Red color
        old_traj_colors[transport.mask_traj] = [1.0, 1.0, 0.0]  # Yellow color for matched points
        old_traj_pcd.colors = o3d.utility.Vector3dVector(old_traj_colors)

        new_traj_pcd = o3d.geometry.PointCloud()
        new_traj_pcd.points = o3d.utility.Vector3dVector(new_traj[:, :3])
        new_traj_colors = np.tile([0.0, 1.0, 0.0], (new_traj[:, :3].shape[0], 1))  # Green color
        new_traj_colors[transport.mask_traj] = [1.0, 1.0, 0.0]  # Yellow color for matched points
        new_traj_pcd.colors = o3d.utility.Vector3dVector(new_traj_colors)

        # homog_traj_pcd = o3d.geometry.PointCloud()
        # homog_traj_pcd.points = o3d.utility.Vector3dVector(homog_traj[:, :3])
        # homog_traj_colors = np.tile([0.0, 0.0, 1.0], (homog_traj[:, :3].shape[0], 1))  # Blue color
        # homog_traj_pcd.colors = o3d.utility.Vector3dVector(homog_traj_colors)

        original_arrows = TrajectoryUtility.visualize_trajectory_with_orientation(old_traj)
        new_arrows = TrajectoryUtility.visualize_trajectory_with_orientation(new_traj)
        correspondence_lines = PointCloudUtility.create_correspondence_lines(source, target)
        o3d.visualization.draw_geometries([source, target, old_traj_pcd, new_traj_pcd, axes] + original_arrows + new_arrows, window_name="Source, Target, and Trajectories")
        o3d.visualization.draw_geometries([source, target, axes, correspondence_lines], window_name="Source, Target, and Trajectories")

    @staticmethod
    def plot_with_correspondence_lines_and_legend(reordered_source, reordered_target, correspondence_lines):
        """
        Visualize the source and target point clouds with correspondence lines and add a legend.

        Args:
            reordered_source: Source point cloud with reordered points.
            reordered_target: Target point cloud with reordered points.
            correspondence_lines: Open3D LineSet object showing correspondences.
        """
        # Legend geometry: small point clouds for the legend
        legend_points = np.array([[-0.3, -0.2, 0], [-0.3, -0.25, 0]])  # Positions for legend dots
        legend_colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Red for source, blue for target
        legend = o3d.geometry.PointCloud()
        legend.points = o3d.utility.Vector3dVector(legend_points)
        legend.colors = o3d.utility.Vector3dVector(legend_colors)

        # Add text to the legend using Open3D text geometries
        text_legend_source = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        text_legend_source.translate([-0.25, -0.2, 0])  # Position text next to red dot
        text_legend_source.paint_uniform_color([1.0, 0.0, 0.0])

        text_legend_target = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        text_legend_target.translate([-0.25, -0.25, 0])  # Position text next to blue dot
        text_legend_target.paint_uniform_color([0.0, 0.0, 1.0])

        # Combine all geometries for visualization
        geometries = [
            reordered_source,
            reordered_target,
            correspondence_lines,
            legend,
            text_legend_source,
            text_legend_target,
        ]

        # Visualize with Open3D
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Correspondence Lines with Legend")
        for geom in geometries:
            vis.add_geometry(geom)

        vis.run()
        vis.destroy_window()

    @staticmethod
    def visualize_all(reordered_source, reordered_target, original_target, old_traj, new_traj, transport,
                      correspondence_lines):
        """
        Visualizes all surfaces and trajectories in different contexts (with arrows, masked points, and lines).

        Parameters:
            reordered_source (o3d.geometry.PointCloud): Source point cloud after reordering.
            reordered_target (o3d.geometry.PointCloud): Target point cloud after reordering.
            original_target (o3d.geometry.PointCloud): Original target point cloud.
            old_traj (np.ndarray): Original trajectory points (Nx7).
            new_traj (np.ndarray): New transported trajectory points (Nx7).
            transport (object): Transport object containing masks for source and trajectory.
        """

        # ----------- Trajectories as Point Clouds -----------
        old_traj_pcd = o3d.geometry.PointCloud()
        old_traj_pcd.points = o3d.utility.Vector3dVector(old_traj[:, :3])
        old_traj_colors = np.tile([1.0, 1.0, 0.0], (old_traj[:, :3].shape[0], 1))  # Yellow for old trajectory
        old_traj_colors[transport.mask_traj] = [0.0, 1.0, 1.0]  # Cyan
        old_traj_pcd.colors = o3d.utility.Vector3dVector(old_traj_colors)

        new_traj_pcd = o3d.geometry.PointCloud()
        new_traj_pcd.points = o3d.utility.Vector3dVector(new_traj[:, :3])
        new_traj_colors = np.tile([0.0, 1.0, 0.0], (new_traj[:, :3].shape[0], 1))  # Green for new trajectory
        new_traj_colors[transport.mask_traj] = [1, 0, 1]  # Magenta
        new_traj_pcd.colors = o3d.utility.Vector3dVector(new_traj_colors)

        # ----------- Visualization with arrows -----------
        original_arrows = TrajectoryUtility.visualize_trajectory_with_orientation(old_traj)
        new_arrows = TrajectoryUtility.visualize_trajectory_with_orientation(new_traj)

        reordered_source.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (reordered_source.points.__len__(), 1)))
        reordered_target.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (reordered_target.points.__len__(), 1)))
        o3d.visualization.draw_geometries(
            [reordered_source, reordered_target, correspondence_lines],
            window_name="Source, Target, and Trajectories as Point Clouds"
        )

        axes_size = 0.25
        axes_origin = [0, 0, 0]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size, origin=axes_origin)

        # ----------- Visualize point clouds and trajectories -----------
        o3d.visualization.draw_geometries(
            [reordered_source, reordered_target, old_traj_pcd, new_traj_pcd, correspondence_lines, axes],
            window_name="Source, Target, and Trajectories as Point Clouds"
        )

        o3d.visualization.draw_geometries(
            [original_target, new_traj_pcd] + new_arrows,
            window_name="Full resolution target with new Trajectory as Point Cloud"
        )

        # ----------- Visualization with masked points -----------
        mask_source, mask_traj = transport.mask_source, transport.mask_traj

        masked_source_points = np.asarray(reordered_source.points)[mask_source]
        masked_traj_points = old_traj[mask_traj, :3]

        masked_source_pcd = o3d.geometry.PointCloud()
        masked_source_pcd.points = o3d.utility.Vector3dVector(masked_source_points)
        masked_source_pcd.paint_uniform_color([0, 1, 0])  # Green for masked source points

        masked_traj_pcd = o3d.geometry.PointCloud()
        masked_traj_pcd.points = o3d.utility.Vector3dVector(masked_traj_points)
        masked_traj_pcd.paint_uniform_color([1, 0, 1])  # Magenta for masked trajectory points

        masked_target_points = np.asarray(reordered_target.points)[mask_source]
        masked_new_traj_points = new_traj[mask_traj, :3]

        masked_target_pcd = o3d.geometry.PointCloud()
        masked_target_pcd.points = o3d.utility.Vector3dVector(masked_target_points)
        masked_target_pcd.paint_uniform_color([0, 1, 1])  # Cyan for masked

        masked_new_traj_pcd = o3d.geometry.PointCloud()
        masked_new_traj_pcd.points = o3d.utility.Vector3dVector(masked_new_traj_points)
        masked_new_traj_pcd.paint_uniform_color([0, 1, 1])  # Cyan for masked

        #masked_correspondence_lines = PointCloudUtility.create_correspondence_lines(masked_source_pcd, masked_target_pcd, color = [1, 1, 0], intensity=1)
        traj_masked_correspondence_lines = PointCloudUtility.create_correspondence_lines(masked_traj_pcd, masked_new_traj_pcd, color = [0, 0, 1], intensity=1)


        o3d.visualization.draw_geometries(
            [reordered_target, new_traj_pcd, masked_source_pcd, masked_target_pcd, masked_traj_pcd,
             masked_new_traj_pcd, traj_masked_correspondence_lines] + original_arrows + new_arrows,
            window_name="Masked points on Target and Trajectory Points as Point Clouds"
        )

        o3d.visualization.draw_geometries(
            [reordered_target, new_traj_pcd, old_traj_pcd, masked_source_pcd, masked_target_pcd, masked_traj_pcd,
             masked_new_traj_pcd, traj_masked_correspondence_lines] + original_arrows + new_arrows,
            window_name="Masked points on Target and Trajectory Points with All Trajectories as Point Clouds"
        )

        o3d.visualization.draw_geometries(
            [new_traj_pcd, old_traj_pcd, traj_masked_correspondence_lines, axes] + original_arrows + new_arrows,
            window_name="Masked points on Old and Target Trajectories"
        )
