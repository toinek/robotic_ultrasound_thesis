import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel as C, RBF
from policy_transportation import GaussianProcessTransportation as Transport
from policy_transportation.plot_utils import draw_error_band
import warnings
import similaritymeasures
warnings.filterwarnings("ignore")
class Multiple_Reference_Frames_GPT:
    def __init__(self):
        k_transport = C(constant_value=np.sqrt(10))  * RBF(20*np.ones(1), [10,50]) + WhiteKernel(0.01 , [0.0000001, 0.000001])
        self.transport=Transport(kernel_transport=k_transport)

    def generate_distribution_from_frames(self, A,b, use_extra_points=True):
        if use_extra_points==True:
            distribution_training_set=np.zeros((len(A),10,2))
        else:
            distribution_training_set=np.zeros((len(A),4,2)) 
        frame_dim=5
        for i in range(len(A)):
            distribution_training_set[i,0,:]=b[i][0][0]
            distribution_training_set[i,1,:]=b[i][0][0]+A[i][0][0] @ np.array([ 0, frame_dim])
            distribution_training_set[i,2,:]=b[i][0][1]
            distribution_training_set[i,3,:]=b[i][0][1]+A[i][0][1] @ np.array([ 0, -frame_dim])
            #Extra points
            if use_extra_points==True:
                distribution_training_set[i,4,:]=b[i][0][0]+A[i][0][0] @ np.array([ 0, -frame_dim])
                distribution_training_set[i,5,:]=b[i][0][1]+A[i][0][1] @ np.array([ 0, frame_dim])
                distribution_training_set[i,6,:]=b[i][0][0]+A[i][0][0] @ np.array([ +frame_dim, 0])
                distribution_training_set[i,7,:]=b[i][0][1]+A[i][0][1] @ np.array([ +frame_dim, 0])
                distribution_training_set[i,8,:]=b[i][0][0]+A[i][0][0] @ np.array([ -frame_dim, 0])
                distribution_training_set[i,9,:]=b[i][0][1]+A[i][0][1] @ np.array([ -frame_dim, 0])
        return distribution_training_set   
    
    def load_dataset(self, filename = 'reach_target', use_extra_points=True):
        

        demos = np.load(filename + '.npy', allow_pickle=True, encoding='latin1')[()]

        ### Trajectory data
        demos_x = demos['x'] # position

        ### Coordinate systems transformation
        demos_A = [d for d in demos['A']]
        demos_b = [d for d in demos['b']]

        distribution_training_set=np.zeros((len(demos_x),10,2))
        final_distance=np.zeros((len(demos_x),2))
        final_orientation=np.zeros((len(demos_x),1))
        # index=2
        distribution_training_set=self.generate_distribution_from_frames(demos_A,demos_b, use_extra_points=use_extra_points)
        for i in range(len(demos_x)):
            final_distance[i]=  np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:] - demos_b[i][0][1])

            final_delta=np.linalg.inv(demos_A[i][0][1]) @ (demos_x[i][-1,:]-demos_x[i][-2,:])
            final_orientation[i]= np.arctan2(final_delta[1],final_delta[0])

        self.demos_x=demos_x
        self.demos_A=demos_A
        self.demos_b=demos_b
        self.distribution_training_set=distribution_training_set
        self.final_distance=final_distance
        self.final_orientation=final_orientation    

    def load_test_dataset(self, test_A, test_b, use_extra_points=True):

        distribution_test_set=np.zeros((len(test_A),10,2))

        distribution_test_set=self.generate_distribution_from_frames(test_A,test_b, use_extra_points=use_extra_points)     
        self.distribution_test_set=distribution_test_set
        self.test_A=test_A
        self.test_b=test_b

    def reproduce(self, index_source, index_target, ax=None, compute_metrics=False, linear=False, plot_bounds=True):
        X=self.demos_x[index_source]

        self.transport.source_distribution=self.distribution_training_set[index_source,:,:]
        self.transport.target_distribution=self.distribution_training_set[index_target,:,:]
        self.transport.training_traj=X

        self.transport.fit_transportation(do_scale=True, do_rotation=True)
        self.transport.apply_transportation()
        std=self.transport.std

        X1=self.transport.training_traj
        

        if ax is not None:
            if index_source == index_target:
                self.plot(X1, std, self.distribution_training_set[index_target,:,:], ax, c_frames=['#FFD700', '#FFD700'])
            else:
                self.plot(X1, std, self.distribution_training_set[index_target,:,:], ax)
            ax.plot(self.demos_x[index_target][:,0],self.demos_x[index_target][:,1], 'k--')


        if compute_metrics==True:    
        # Discrete Frechet distance
            df = similaritymeasures.frechet_dist(self.demos_x[index_target], X1)

            # quantify the difference between the two curves using
            # area between two curves
            area = similaritymeasures.area_between_two_curves(self.demos_x[index_target], X1)

            # quantify the difference between the two curves using
            # Dynamic Time Warping distance
            dtw, d = similaritymeasures.dtw(self.demos_x[index_target], X1)

            # fde=np.linalg.norm(demos_x[i][-1]-X1[-1])
            fd=  np.linalg.inv(self.demos_A[index_target][0][1]) @ (X1[-1] - self.demos_b[index_target][0][1])
            fde=np.linalg.norm(self.final_distance[index_target]-fd)

            final_vel=  np.linalg.inv(self.demos_A[index_target][0][1]) @ (X1[-1] - X1[-5])

            final_angle= np.arctan2(final_vel[1], final_vel[0])

            final_angle_distance= np.abs(final_angle - self.final_orientation[index_target])  

            print("Final Point Distance  : ", fde)
            print("Frechet Distance      : ", df)
            print("Area between two curves: ", area)
            print("Dynamic Time Warping  : ", dtw)
            print("Final Angle Distance  : ", final_angle_distance[0])
            return df, area, dtw, fde, final_angle_distance[0]

    def generalize(self, index_source, index_target, ax=None, compute_metrics=False, linear=False):
        X=self.demos_x[index_source].reshape(-1,2)

        self.transport.source_distribution=self.distribution_training_set[index_source,:,:].reshape(-1,2)
        self.transport.target_distribution=self.distribution_test_set[index_target,:,:].reshape(-1,2)
        self.transport.training_traj=X

        self.transport.fit_transportation(do_scale=True, do_rotation=True)
        self.transport.apply_transportation()
        std=self.transport.std
        X1=self.transport.training_traj
        if ax is not None:
            self.plot(X1, std, self.distribution_test_set[index_target,:,:], ax=ax)


        if compute_metrics==True:    
            # fde=np.linalg.norm(transport.target_distribution[2,:]-X1[-1])
            fd=  np.linalg.inv(self.test_A[index_target][0][1]) @ (X1[-1] - self.test_b[index_target][0][1])
            fde=np.linalg.norm(self.final_distance[index_target]-fd)

            final_vel=  np.linalg.inv(self.test_A[index_target][0][1]) @ (X1[-1] - X1[-5])

            final_angle= np.arctan2(final_vel[1], final_vel[0])

            final_angle_distance= np.abs(final_angle - self.final_orientation[index_target])

            print("Final Point Distance  : ", fde)
            print("Final Angle Distance  : ", final_angle_distance[0])
            return fde, final_angle_distance[0]
    
    def plot(self, X1, std, distribution, ax=None, plot_bounds=True, c_frames=['green', [30.0/256.0,144.0/256.0,255.0/256.0]]):
        if plot_bounds==True:
            draw_error_band(ax, X1[:,0], X1[:,1], err=std, facecolor= [255.0/256.0,140.0/256.0,0.0], edgecolor="none", alpha=.8)
        ax.plot(distribution[0:2,0],distribution[0:2,1], linewidth=10, alpha=0.9, c=c_frames[0])
        ax.scatter(distribution[0,0],distribution[0,1], linewidth=10, alpha=0.9, c=c_frames[0])
        ax.plot(distribution[2:4,0],distribution[2:4,1], linewidth=10, alpha=0.9, c=c_frames[1])
        ax.scatter(distribution[2,0],distribution[2,1], linewidth=10, alpha=0.9, c=c_frames[1])
        ax.plot(distribution[:,0],distribution[:,1], 'b*',  linewidth=0.2)
        ax.plot(X1[:,0],X1[:,1], c= [255.0/256.0,20.0/256.0,147.0/256.0])