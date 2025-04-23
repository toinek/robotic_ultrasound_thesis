#!/usr/bin/env python
import copy
import rospy
import open3d as o3d
import time
import numpy as np

from policy_transportation.transportation.laplacian_editing_transportation import \
    LaplacianEditingTransportation as Transport

from realsense.realsense_vision.pcd_helpers import ICPMatcher, PointCloudConverter
from apply_transport_rigid import ApplyTransport
from realsense.realsense_vision.registration_utils import PointCloudUtility
from realsense.data_analysis.computation_logger import timing_logger

class ApplyTransportDynamic(ApplyTransport):
    def __init__(self, technique, first_pcd_path, debug=False, keypoints=False):
        super().__init__(technique, first_pcd_path, debug)
        self.matcher = ICPMatcher()  # Overwrite matcher with CPD-based approach
        self.pcd_converter = PointCloudConverter(x_min=-5, x_max=5, y_min=-5, y_max=5, z_min=0.03, z_max = 1.5)
        self.keypoints = keypoints
        self.last_update_time = 0

    def pcd_cb(self, pcd_msg):
        """ Callback function for handling incoming point cloud messages
        and applying the transportation technique to the point clouds.
        """

        start_time = time.perf_counter()
        if self.target_surface is None:
            self.target_surface = self.first_pcd
            # o3d.visualization.draw_geometries([self.target_surface], window_name="First point cloud")
            # self.target_surface = self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg)
            return

        if self.old_policy is None:
            rospy.logerr("Registration callback waiting for the old policy...")
            return

        end_time = time.perf_counter()
        timing_logger.log_timing("PCD_dynamic/PCD Message Handling & Preprocessing", end_time - start_time)

        if self.keypoints:
            # No matching needed, only spare keypoints
            start_time = time.perf_counter()
            keypoint = True

            if len(self.target_surface.points) != 4 or len(self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg).points) != 4:
                rospy.logwarn(f"Not enough keypoints detected. Skipping transportation. Sizes: {len(self.target_surface.points)} {len(self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg).points)}")
                return

            # Compute pairwise Euclidean distance
            distance = np.sum(np.linalg.norm(
                np.array(self.target_surface.points) - np.array(self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg).points), axis=1
            ))
            # Compute time difference
            current_time = time.time()

            # Skip transportation if keypoints are similar or time difference is less than 0.5 seconds
            if distance < 0.1 or current_time - self.last_update_time < 0.5:
                # rospy.loginfo("Keypoints are similar. Skipping transportation.")
                return None

            # Update source and target surfaces and send to transportation
            self.source_surface = self.target_surface
            self.target_surface = self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg)
            reordered_source, reordered_target = self.source_surface, self.target_surface
            self.last_update_time = time.time()
            end_time = time.perf_counter()
            timing_logger.log_timing("PCD_keypoints/Keypoints entire pipeline", end_time - start_time)

        else:
            # Perform dense pcd matching
            start_time = time.perf_counter()
            keypoint=False

            target_pcd = self.pcd_converter.convert_pc2msg_to_pcd(pcd_msg)
            voxel_size = 0.02
            source_down = self.pcdutil.downsample_and_denoise(self.target_surface, voxel_size=voxel_size)
            target_down = self.pcdutil.downsample_and_denoise(target_pcd, voxel_size=voxel_size)

            chamfer_dist = self.pcdutil.chamfer_distance(source_down.points, target_down.points)
            # rospy.loginfo(f"Chamfer Distance: {chamfer_dist}")
            if chamfer_dist < 0.075:
                # rospy.loginfo("Source and target surfaces are similar. Skipping transportation.")
                return None

            self.source_surface = self.target_surface
            self.target_surface = target_pcd

            self.target_surface.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # Make a copy of the target
            target_copy = copy.deepcopy(target_down)

            if self.debug:
                o3d.visualization.draw_geometries([source_down, target_down], window_name='Source + Target Point Clouds')

            end_time = time.perf_counter()
            timing_logger.log_timing("PCD_dynamic/Downsampling & Chamfer Distance", end_time - start_time)

            # Step 2: Coherent Point Drift (CPD) Registration
            start_time = time.perf_counter()
            matched_pcd1, matched_pcd2, tf_param = self.pcdutil.apply_cpd(
                source_down, target_down, tf_type_name='nonrigid', maxiter=100, tol=1e-5
            )
            end_time = time.perf_counter()
            timing_logger.log_timing("PCD_dynamic/CPD Registration", end_time - start_time)
            if self.debug:
                o3d.visualization.draw_geometries([matched_pcd1, matched_pcd2], window_name='CPD Registration Result')

            # Step 3: ICP Fine Alignment (after CPD)
            start_time = time.perf_counter()
            self.matcher.run_icp(matched_pcd1, matched_pcd2)
            matched_pcd1.transform(self.matcher.registration_result.transformation)
            end_time = time.perf_counter()
            timing_logger.log_timing("PCD_dynamic/ICP Fine Alignment", end_time - start_time)
            if self.debug:
                o3d.visualization.draw_geometries([matched_pcd1, matched_pcd2], window_name='ICP Registration Result')

            ### Calculate Registration Metrics
            mean_dist, max_dist = PointCloudUtility.calculate_point_to_surface_metrics(matched_pcd1, matched_pcd2)
            overlap_ratio = PointCloudUtility.calculate_overlap_ratio(matched_pcd1, matched_pcd2, threshold=0.02)

            rospy.loginfo(
                f"Final Registration Metrics -> Mean Distance: {mean_dist} m, Max Distance: {max_dist} m, Overlap Ratio: {overlap_ratio} m")
            ### Finished calculating registration metrics

            # Step 4: Reordering Point Clouds Based on Correspondence
            start_time = time.perf_counter()
            reordered_source, reordered_target = self.matcher.reorder_point_clouds_based_on_correspondences(
                source_down, target_copy
            )
            end_time = time.perf_counter()
            timing_logger.log_timing("PCD_dynamic/Reordering Point Clouds", end_time - start_time)
        self.apply_transport(reordered_source, reordered_target, keypoint=keypoint)



if __name__ == '__main__':
    rospy.init_node('policy_transportation_dynamic', anonymous=True)
    technique = Transport()
    transport = ApplyTransportDynamic(technique, "18feb_sparse_arm.pcd", debug=False, keypoints=True)

    rospy.spin()
