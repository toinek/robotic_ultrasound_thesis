import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.interpolate import griddata
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as Transport

def create_random_surface(X, Y, length_scale):
    X_ = X.reshape(-1, 1)
    Y_ = Y.reshape(-1, 1)
    X_train = np.hstack([X_, Y_])

    # Define a random kernel
    kernel = C(1.0) * RBF(length_scale=length_scale)

    # Compute the covariance matrix
    K = kernel(X_train, X_train) + 0.0001 * np.eye(X_train.shape[0])

    # Cholesky decomposition to sample from the Gaussian process
    L = np.linalg.cholesky(K)

    # Sample from the GP
    u = np.random.normal(loc=0, scale=1, size=len(X_train))
    Z = np.dot(L, u)

    return Z.reshape(X.shape)

def apply_transport(trajectory_old, old_surface, new_surface):
    # Transport the trajectory from the old surface to the new surface
    delta_trajectory = np.zeros((len(trajectory_old),3))
    for j in range(len(trajectory_old)-1):
        delta_trajectory[j,:]=(trajectory_old[j+1,:]-trajectory_old[j,:])
    transport = Transport()
    transport.source_distribution=old_surface
    transport.target_distribution=new_surface
    transport.training_traj=trajectory_old
    transport.training_delta=delta_trajectory

    transport.fit_transportation(do_scale=True, do_rotation=True)
    transport.apply_transportation()
    new_trajectory=transport.training_traj
    new_delta=transport.training_delta

    return new_trajectory


if __name__ == '__main__':
    # Generate the grid
    X = np.arange(-5, 5, 1)
    Y = np.arange(-5, 5, 1)
    X, Y = np.meshgrid(X, Y)

    # Create two random surfaces using different length scales
    old_surface = create_random_surface(X, Y, length_scale=1.0)
    new_surface = create_random_surface(X, Y, length_scale=1.5)

    # Create a smooth connected trajectory on the old surface
    # Define a straight line trajectory along one axis (e.g., diagonal path)
    num_trajectory_points = 200
    x_start, x_end = -5, 5
    y_start, y_end = -5, 5

    # Generate 200 points along a diagonal line in the X-Y plane
    trajectory_x = np.linspace(x_start, x_end, num_trajectory_points)
    trajectory_y = np.linspace(y_start, y_end, num_trajectory_points)

    # Flatten the grid for interpolation
    points = np.vstack((X.ravel(), Y.ravel())).T

    # Interpolate Z values for the trajectory on the old surface
    trajectory_z_old = griddata(points, old_surface.ravel(), (trajectory_x, trajectory_y), method='cubic')

    # Interpolate Z values for the trajectory on the new surface
    trajectory_z_new = griddata(points, new_surface.ravel(), (trajectory_x, trajectory_y), method='cubic')

    # Stack the trajectory points into arrays
    trajectory_old = np.vstack((trajectory_x, trajectory_y, trajectory_z_old)).T


    # Transport the trajectory from the old surface to the new surface
    delta_trajectory = np.zeros((len(trajectory_old),3))
    for j in range(len(trajectory_old)-1):
        delta_trajectory[j,:]=(trajectory_old[j+1,:]-trajectory_old[j,:])
    transport = Transport()
    transport.source_distribution=old_surface
    transport.target_distribution=new_surface
    transport.training_traj=trajectory_old
    transport.training_delta=delta_trajectory

    transport.fit_transportation(do_scale=True, do_rotation=True)
    transport.apply_transportation()
    new_trajectory=transport.training_traj
    new_delta=transport.training_delta



    # Plot the surfaces and the trajectories
    fig = plt.figure(figsize=(14, 7))

    # Plot old surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, old_surface, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
    ax1.plot(trajectory_x, trajectory_y, trajectory_z_old, 'ro-', label='Trajectory on Old Surface', linewidth=2)
    ax1.set_title('Old Surface with Trajectory')
    ax1.legend()
    #fig.colorbar(surf1, shrink=0.5, aspect=5)

    # Plot new surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, new_surface, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5)
    ax2.plot(trajectory_x, trajectory_y, trajectory_z_new, 'go-', label='Trajectory on New Surface', linewidth=2)
    ax2.set_title('New Surface with Transported Trajectory')
    ax2.legend()
    fig.colorbar(surf2, shrink=0.5, aspect=5)

    plt.show()
