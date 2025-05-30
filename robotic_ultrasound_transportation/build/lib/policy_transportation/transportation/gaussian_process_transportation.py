"""
Authors: Giovanni Franzese June 2024
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
from src.gaussian_process_transportation.policy_transportation import GaussianProcess
from src.gaussian_process_transportation.policy_transportation.transportation.policy_transportation import PolicyTransportation
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import numpy as np  
class GaussianProcessTransportation():
    def __init__(self, kernel_transport=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0001)):
        super(GaussianProcessTransportation, self).__init__()
        self.method=PolicyTransportation(GaussianProcess(kernel=kernel_transport))

    
    def fit_transportation(self, do_scale=False, do_rotation=True):
        self.method.fit(self.source_distribution, self.target_distribution, do_scale=do_scale, do_rotation=do_rotation)

    def accuracy(self):
        source_transported=self.method.transport(self.source_distribution)[0]
        #compute rmse
        error=np.sqrt(np.mean((source_transported-self.target_distribution)**2))
        # error = np.max(np.abs(source_transported-self.target_distribution))
        # error = error/np.sqrt(np.sum((self.source_distribution-self.target_distribution)**2))
        return error
    def apply_transportation(self):
        self.training_traj_old=self.training_traj
        self.training_traj, self.std=self.method.transport(self.training_traj_old)

        if hasattr(self, 'training_delta'):
            self.training_delta, self.var_vel_transported =self.method.transport_velocity(self.training_traj_old, self.training_delta)
            self.diffeo_mask=self.method.diffeo_mask
        if hasattr(self, 'training_ori'):
            self.training_ori=self.method.transport_orientation(self.training_traj_old, self.training_ori)

    def sample_transportation(self):
        samples=self.method.sample_transportation(self.training_traj_old)
        return samples


