import matplotlib.pyplot as plt
import numpy as np
from policy_transportation.transportation.laplacian_editing_transportation import LaplacianEditingTransportation as Transport

def create_2D_surfaces():
    x1 = np.linspace(-50, 50, 100)
    y1 = np.linspace(-50, 50, 100)

    x2 = np.linspace(-100, 100, 100)
    y2 = np.linspace(-0, 0, 100)

    trajectory_x = np.linspace(-20, 0, 20)
    trajectory_y = np.linspace(-20, 0, 20)

    old_surface = np.vstack((x1 , y1)).T
    new_surface = np.vstack((x2 , y2)).T
    trajectory = np.vstack((trajectory_x, trajectory_y)).T
    return old_surface, new_surface, trajectory

def create_3D_surfaces():
    x1 = np.linspace(-50, 50, 100)
    y1 = np.linspace(-50, 50, 100)
    z1 = np.linspace(-50, 50, 100)

    x2 = np.linspace(-100, 100, 100)
    y2 = np.linspace(-0, 0, 100)
    z2 = np.linspace(-0, 0, 100)

    trajectory_x = np.linspace(-20, 0, 20)
    trajectory_y = np.linspace(-20, 0, 20)
    trajectory_z = np.linspace(-20, 0, 20)

    old_surface = np.vstack((x1 , y1, z1)).T
    new_surface = np.vstack((x2 , y2, z2)).T
    trajectory = np.vstack((trajectory_x, trajectory_y, trajectory_z)).T
    return old_surface, new_surface, trajectory

def create_eclipse_surfaces():
    # Parameters for the 3D circle (old surface)
    radius = 50
    theta = np.linspace(0, 2 * np.pi, 100)  # Parametric angle for circle and ellipse

    x1 = radius * np.cos(theta)
    y1 = radius * np.sin(theta)
    z1 = np.zeros_like(theta)  # Z is 0 for the circle in the XY plane

    # Parameters for the 3D ellipse (new surface)
    a = 100  # Semi-major axis
    b = 50   # Semi-minor axis
    x2 = a * np.cos(theta)
    y2 = b * np.sin(theta)
    z2 = np.zeros_like(theta)  # Z is also 0 for the ellipse in the XY plane

    # Trajectory: a part of the 3D circle (e.g., the first quarter)
    trajectory_theta = np.linspace(0, np.pi / 2, 20)  # First quarter of the circle
    trajectory_x = radius * np.cos(trajectory_theta)
    trajectory_y = radius * np.sin(trajectory_theta)
    trajectory_z = np.zeros_like(trajectory_theta)  # Z is 0

    # Stack them into surface and trajectory arrays
    old_surface = np.vstack((x1, y1, z1)).T
    new_surface = np.vstack((x2, y2, z2)).T
    trajectory = np.vstack((trajectory_x, trajectory_y, trajectory_z)).T

    return old_surface, new_surface, trajectory

def create_sphere_surfaces():
    # Number of points on the sphere
    num_points = 100

    # Parameters for the first 3D sphere (old surface)
    radius1 = np.random.uniform(20, 50)  # Random radius between 20 and 50
    theta1 = np.linspace(0, np.pi, num_points)  # Polar angle
    phi1 = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle
    theta1, phi1 = np.meshgrid(theta1, phi1)

    x1 = radius1 * np.sin(theta1) * np.cos(phi1)
    y1 = radius1 * np.sin(theta1) * np.sin(phi1)
    z1 = radius1 * np.cos(theta1)

    # Flatten arrays for easier manipulation
    old_surface = np.vstack((x1.ravel(), y1.ravel(), z1.ravel())).T

    # Parameters for the second 3D sphere (new surface)
    radius2 = np.random.uniform(50, 100)  # Random radius between 50 and 100
    theta2 = np.linspace(0, np.pi, num_points)  # Polar angle
    phi2 = np.linspace(0, 2 * np.pi, num_points)  # Azimuthal angle
    theta2, phi2 = np.meshgrid(theta2, phi2)

    x2 = radius2 * np.sin(theta2) * np.cos(phi2)
    y2 = radius2 * np.sin(theta2) * np.sin(phi2)
    z2 = radius2 * np.cos(theta2)

    # Apply a random translation to the new sphere
    translation_vector = np.random.uniform(-100, 100, size=3)  # Random translation in each axis
    new_surface = np.vstack((x2.ravel(), y2.ravel(), z2.ravel())).T + translation_vector

    # Trajectory: Create a path along the surface of the first sphere (along a great circle)
    # Let's take a constant azimuthal angle (phi = 0), and vary the polar angle theta
    num_trajectory_points = 20
    trajectory_theta = np.linspace(0, np.pi / 2,
                                   num_trajectory_points)  # Polar angle for the trajectory (quarter of the sphere)
    phi_constant = 0  # Constant azimuthal angle for a great circle path

    trajectory_x = radius1 * np.sin(trajectory_theta) * np.cos(phi_constant)
    trajectory_y = radius1 * np.sin(trajectory_theta) * np.sin(phi_constant)
    trajectory_z = radius1 * np.cos(trajectory_theta)

    # Stack the trajectory points into an array
    trajectory = np.vstack((trajectory_x, trajectory_y, trajectory_z)).T

    return old_surface, new_surface, trajectory



old_surface, new_surface, trajectory = create_sphere_surfaces()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(old_surface[:,0], old_surface[:,1], old_surface[:,2])
ax.plot(new_surface[:,0], new_surface[:,1], new_surface[:,2])
ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 'ro')


delta_trajectory = np.zeros((len(trajectory),3))
for j in range(len(trajectory)-1):
    delta_trajectory[j,:]=(trajectory[j+1,:]-trajectory[j,:])

transport = Transport()
transport.source_distribution=old_surface
transport.target_distribution=new_surface
transport.training_traj=trajectory
transport.training_delta=delta_trajectory

transport.fit_transportation(do_scale=False, do_rotation=True)
transport.apply_transportation()
new_trajectory=transport.training_traj
new_delta=transport.training_delta

plt.plot(new_trajectory[:,0], new_trajectory[:,1], new_trajectory[:,2], 'ys')
plt.show()