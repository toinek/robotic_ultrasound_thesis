"""
Authors:  Giovanni Franzese and Ravi Prakash, Dec 2022
Email: g.franzese@tudelft.nl, r.prakash-1@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""

#%%
import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
import matplotlib.pyplot as plt
from policy_transportation import GaussianProcess as GPR
from policy_transportation import GaussianProcessTransportation as Transport
import pathlib
from policy_transportation.plot_utils import plot_vector_field, plot_vector_field_minvar
from policy_transportation.utils import resample
import warnings
warnings.filterwarnings("ignore")
#%% Load the drawings

source_path = str(pathlib.Path(__file__).parent.absolute())  
data =np.load(source_path+ '/data/'+str('example7')+'.npz')
X=data['demo'] 
S=data['floor'] 
S1=data['newfloor']
X=resample(X, num_points=200)
source_distribution=resample(S, num_points=20)
target_distribution=resample(S1, num_points=20)

print(source_distribution)
print(target_distribution)

#%% Calculate deltaX
deltaX = np.zeros((len(X),2))
for j in range(len(X)-1):
    deltaX[j,:]=(X[j+1,:]-X[j,:])


#%% Fit a dynamical system to the demo and plot it
k_deltaX = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), nu=2.5) + WhiteKernel(0.01) 
gp_deltaX=GPR(kernel=k_deltaX)
gp_deltaX.fit(X, deltaX)
x_grid=np.linspace(np.min(X[:,0]-10), np.max(X[:,0]+10), 100)
y_grid=np.linspace(np.min(X[:,1]-10), np.max(X[:,1]+10), 100)
plot_vector_field(gp_deltaX, x_grid,y_grid,X,target_distribution)
plt.quiver(X[:,0], X[:,1], deltaX[:,0], deltaX[:,1], color=[1,0,0],  scale=None)
fig = plt.figure(figsize = (12, 7))
plt.xlim([-50, 50-1])
plt.ylim([-50, 50-1])
plt.scatter(X[:,0],X[:,1], color=[1,0,0]) 
plt.scatter(source_distribution[:,0],source_distribution[:,1], color=[0,1,0])   
plt.scatter(target_distribution[:,0],target_distribution[:,1], color=[0,0,1]) 
plt.legend(["Demonstration","Surface","New Surface"])
#%% Transport the dynamical system on the new surface
k_transport = C(constant_value=10)  * RBF(1*np.ones(2), [0.1,5]) + WhiteKernel(0.01 )

transport=Transport(kernel_transport= k_transport)
transport.source_distribution=source_distribution 
transport.target_distribution=target_distribution
transport.training_traj=X
transport.training_delta=deltaX

print('Transporting the dynamical system on the new surface')
transport.fit_transportation(do_scale=False, do_rotation=True)
transport.apply_transportation()
X1=transport.training_traj
deltaX1=transport.training_delta


# Fit the Gaussian Process dynamical system   
print('Fitting the GP dynamical system on the transported trajectory')
k_deltaX1 = C(constant_value=np.sqrt(0.1))  * Matern(1*np.ones(2), [5,200], nu=2.5) + WhiteKernel(0.01)    
gp_deltaX1=GPR(kernel=k_deltaX1)
mask = ~transport.diffeo_mask

gp_deltaX1.fit(X1[mask], deltaX1[mask])
# gp_deltaX1.fit(X1, deltaX1)
x1_grid=np.linspace(np.min(X1[:,0]-10), np.max(X1[:,0]+10), 100)
y1_grid=np.linspace(np.min(X1[:,1]-10), np.max(X1[:,1]+10), 100)
plot_vector_field_minvar(gp_deltaX1, x1_grid,y1_grid,X1[mask],target_distribution)
plt.scatter(X1[transport.diffeo_mask,0],X1[transport.diffeo_mask,1], color=[0,0,1], marker='x', s=100)
# plt.quiver(X1[:,0], X1[:,1], deltaX1[:,0], deltaX1[:,1], color=[1,0,0],  scale=None)
plt.show()