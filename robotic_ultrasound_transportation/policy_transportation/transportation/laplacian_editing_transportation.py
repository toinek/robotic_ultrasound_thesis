"""
Authors: Giovanni Franzese
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
import numpy as np
from copy import copy, deepcopy
from policy_transportation.models.affine_trasformation import AffineTransform
from policy_transportation.models.laplacian_editing import Laplacian_Editing
from scipy.spatial.transform import Rotation as R
from numpy.linalg import qr

class LaplacianEditingTransportation():
    def __init__(self):
        super(LaplacianEditingTransportation, self).__init__()
        self.mask_traj=None
        self.mask_source=None
    
    def fit_transportation(self, distance_threshold, do_scale=True, do_rotation=True, current_spline_index=0, keypoint=False):
        print('Fitting Transportation')
        # Initialize the affine transformation and the transportation classes
        self.affine_transform=AffineTransform(do_scale=do_scale, do_rotation=do_rotation)
        self.transportation=Laplacian_Editing()

        # Fit the affine transformation to the source and target distributions
        self.affine_transform.fit(self.source_distribution, self.target_distribution)

        # Apply the affine transformation to the source distribution and the training trajectory
        source_distribution=self.affine_transform.predict(self.source_distribution)
        self.transportation.training_traj_old = copy(self.training_traj)
        self.training_traj[:, :3] = self.affine_transform.predict(self.training_traj[:, :3])

        # Fit the transportation model to the transformed source distribution and the training trajectory
        if self.mask_traj is None and self.mask_source is None:
            self.transportation.fit(source_distribution, self.target_distribution, self.training_traj[:, :3],
                                    distance_threshold, current_spline_index=current_spline_index, keypoint=keypoint)
        else:
            self.transportation.fit(source_distribution, self.target_distribution, self.training_traj[:, :3],
                                    distance_threshold, current_spline_index=current_spline_index, keypoint=keypoint,
                                    mask_traj=self.mask_traj, mask_source=self.mask_source)
        self.mask_traj=self.transportation.mask_traj
        self.mask_source=self.transportation.mask_source
        # print("The mask of the source is:", self.mask_source)
        # print("The mask of the traj is:", self.mask_traj)


    def apply_transportation(self):
              
        # Deform Trajactories
        self.training_traj_old=self.training_traj
        self.traj_rotated=self.affine_transform.predict(self.training_traj[:, :3])
        self.training_traj[:, :3], self.std= self.transportation.predict(self.traj_rotated, return_std=True)

        # Calculate Jacobian J for orientation transformation based on positions
        if hasattr(self, 'training_delta'):

            # Calculate the Jacobian matrix J of the transformation
            delta = self.training_traj[1:, :3, np.newaxis] - self.training_traj_old[:-1, :3, np.newaxis]
            delta_hat = self.training_traj_old[1:, :3, np.newaxis] - self.training_traj_old[:-1, :3, np.newaxis]

            J = delta_hat @np.linalg.pinv(delta)
            J = np.concatenate((J, J[-1:, :, :]), axis=0)

            # Apply the Jacobian transformation to the trajectory’s orientations using matrix multiplication
            adjusted_orientations = []
            for i in range(len(self.training_traj)):
                original_orientation = R.from_quat(self.training_traj_old[i, 3:])
                new_orientation_matrix = J[i] @ original_orientation.as_matrix()
                adjusted_orientation = R.from_matrix(new_orientation_matrix).as_quat()
                adjusted_orientations.append(adjusted_orientation)

            # Update the orientations in the transformed trajectory
            self.training_traj[:, 3:] = np.array(adjusted_orientations)

            # # Apply Jacobian on training_delta if it exists
            # if J.shape[0] > self.training_delta.shape[0]:
            #     J = J[:self.training_delta.shape[0], :, :]
            # self.training_delta = (J @ self.training_delta[:, :, np.newaxis])[:, :, 0]

    def accuracy(self):
        return self.transportation.accuracy    


    def sample_transportation(self):
        training_traj_samples= self.transportation.samples(self.traj_rotated)
        return training_traj_samples
