"""
Authors: Giovanni Franzese and Ravi Prakash, May 2023
Email: r.prakash-1@tudelft.nl, g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""


import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from geometry_msgs.msg import PoseStamped
import open3d as o3d
import pickle
from policy_transportation.models.torch.stocastic_variational_gaussian_process import StocasticVariationalGaussianProcess

class Surface_PointCloud_Detector(): 
    def __init__(self):
        super(Surface_PointCloud_Detector, self).__init__() 

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.base_frame = "panda_link0"

        

        # For Cleaning experiment
        self.view_marker = PoseStamped()
        self.view_marker.header.frame_id = "panda_link0"
        self.view_marker.pose.position.x = 0.37216857
        self.view_marker.pose.position.y = -0.07206429
        self.view_marker.pose.position.z = 0.71190887
        self.view_marker.pose.orientation.w = 0.02090975
        self.view_marker.pose.orientation.x =  0.99741665
        self.view_marker.pose.orientation.y =  0.02242987
        self.view_marker.pose.orientation.z = 0.06492219


        rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pointcloud_subscriber_callback, queue_size=1)
        self.point_cloud = o3d.geometry.PointCloud()
        self.source_distribution = None
        self.target_distribution = None



    def pointcloud_subscriber_callback(self, msg):
        try:
            # Retrieve the transform between the camera frame and the robot frame

            transform = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(1.0))

            # Convert the point cloud to the robot frame
            msg_in_robot_frame = do_transform_cloud(msg, transform)

            # Convert PointCloud2 message to numpy array
            pc_data = pc2.read_points(msg_in_robot_frame, skip_nans=True, field_names=("x", "y", "z"))

            # Create Open3D point cloud from numpy array
            self.point_cloud.points = o3d.utility.Vector3dVector(pc_data)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # rospy.logerr('Error occurred during point cloud transformation: %s', str(e))
            pass


        
    def pick_points(self,pcd):
        print("")
        print("1) Please pick the corner points of the PCD using [shift + left click]")
        print("   Press [shift + right click] to undo point picking")
        print("2) Afther picking points, press q for close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        print("selected points:",vis.get_picked_points())
        return vis.get_picked_points()

    def meshgrid(self,picked_points):
        # Define the 4 corner points of the quadrilateral

        A = np.array(picked_points[0,:-1])
        B = np.array(picked_points[1,:-1])
        C = np.array(picked_points[2,:-1])
        D = np.array(picked_points[3,:-1])


        # Define the number of points in the x and y directions of the grid
        nx = 20
        ny = 20

        # Create the linspace arrays for AB and CD
        AB = np.linspace(A, B, nx)
        CD = np.linspace(D, C, nx)

        # Initialize the meshgrid array
        data=np.empty([0,2])
        # Create the meshgrid by linearly interpolating between AB and CD
        for i in range(nx):
            line=np.linspace(AB[i], CD[i], ny)
            data=np.vstack([data,line])

        # print("data.shape",data.shape)    

        return data



    def crop_geometry(self, picked_points_distribution, pcd):
        min_bound = np.min(picked_points_distribution, axis=0)
        max_bound = np.max(picked_points_distribution, axis=0)
    
        # Set z-bounds to positive and negative infinity
        min_bound[2] = -np.inf
        max_bound[2] = np.inf
    
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        # Crop the point cloud using the bounding box
        cropped_pcd = pcd.crop(bbox)

        return cropped_pcd

    def record_distribution(self, distribution):
        
        print("Visualize np Distribution and Grid")
        picked_id_distribution = self.pick_points(distribution)
        picked_points_distribution = np.asarray(distribution.points)[picked_id_distribution]

        cropped_distribution = self.crop_geometry(picked_points_distribution, distribution)

        meshgrid_distribution = self.meshgrid(picked_points_distribution)

        down_distribution = cropped_distribution.voxel_down_sample(voxel_size=0.02)

        distribution_np = np.asarray(down_distribution.points)
        fig = plt.figure()
        ax = plt.axes(projection ='3d') 
        ax.scatter(distribution_np[:,0], distribution_np[:,1], distribution_np[:,2])
        # np.savez(str(pathlib.Path().resolve())+'/data/point_cloud_distribution.npz', point_cloud_distribution=distribution_np)
 
        print("Find the points corresponding of the selected grid")
        gp_distribution=StocasticVariationalGaussianProcess(distribution_np[:,:2], distribution_np[:,2].reshape(-1,1), num_inducing=1000)
        gp_distribution.fit(num_epochs=5) 
        newZ,_ = gp_distribution.predict(meshgrid_distribution)

        distribution_surface=np.hstack([meshgrid_distribution,newZ.reshape(-1,1)])
        # print("distribution_surface.shape",distribution_surface.shape)
        ax.scatter(distribution_surface[:,0], distribution_surface[:,1], distribution_surface[:,2], 'r')
        plt.show()
        return distribution_surface

    def convert_distribution_to_array(self):
        pass

    def record_source_distribution(self):
        source_cloud = self.point_cloud
        self.source_distribution = self.record_distribution(source_cloud)

    def record_target_distribution(self):
        target_cloud = self.point_cloud
        self.target_distribution = self.record_distribution(target_cloud)

    def save_distributions(self):
        # create a binary pickle file 
        f = open("distributions/source.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.source_distribution,f)
        # close file
        f.close()

    # create a binary pickle file 
        f = open("distributions/target.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.target_distribution,f)
        # close file
        f.close()

    def load_distributions(self):
        try:
            with open("distributions/source.pkl","rb") as source:
                self.source_distribution = pickle.load(source)
        except:
            print("No source distribution saved")

        try:
            with open("distributions/target.pkl","rb") as target:
                self.target_distribution = pickle.load(target)
        except:
            print("No target distribution saved")    
