import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

class Laplacian_Editing():
    def __init__(self):
        self.training_traj_old = None

    def create_graph(self, training_traj):
        # Create a chain graph
        num_nodes = training_traj.shape[0]
        #check the distance between the first and the last point
        #if it is smaller than a threshold, create a cycle graph

        # The threshold should be the max distance between two consecutive points
        threshold_distance = np.max(np.linalg.norm(training_traj[1:]-training_traj[:-1], axis=1)) # removed *5
        if np.linalg.norm(training_traj[0]-training_traj[-1])<threshold_distance:
            G = nx.cycle_graph(num_nodes)
            print("Cycle graph")
        else:
            G = nx.path_graph(num_nodes)    
            print("Path graph")
        # Compute the graph Laplacian matrix that is the discrete analog of the Laplace-Beltrami operator.
        # It can be computed as the difference between the degree matrix and the adjacency matrix.
        # The degree matrix of an undirected graph is a diagonal matrix which contains information about the degree of each vertex—that is
        #, the number of edges attached to each vertex. The adjacency matrix of an undirected graph is a square matrix with dimensions
        # equal to the number of vertices in the graph. The elements of the matrix indicate whether pairs of vertices are adjacent or
        # not in the graph.
        self.L = nx.laplacian_matrix(G).toarray()
        self.L = self.L 
        # Rather than working in absolute Cartesian coordinates, the discrete Laplace-Beltrami operator specifies the local path properties,
        # called Laplacian coordinates Delta that can be calculated as the product of the graph Laplacian and the training trajectory
        self.DELTA= self.L @ training_traj

        return self.L, self.DELTA
    
    def find_matching_waypoints_hungarian(self, source_distribution, training_traj, distance_threshold):
       # ceate cdist matrix
        distance_matrix = np.linalg.norm(training_traj[:, None] - source_distribution, axis=2)


        # Apply the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        # The Hungarian algorithm is a combinatorial optimization algorithm that solves the assignment problem in polynomial time.
        # The algorithm has many applications in combinatorial optimization, for example in problems of matching supply and demand
        # in transportation networks, or in finding the minimum cost assignment in job scheduling.
        # The algorithm is also known as the Kuhn-Munkres algorithm.

        # The optimal assignment and the corresponding costs
        optimal_assignment = list(zip(row_ind, col_ind))
        assignment_costs = distance_matrix[row_ind, col_ind]

        # Determine a threshold for outliers (e.g., mean + 2 standard deviations)
        threshold = np.mean(assignment_costs) + 1.5 * np.std(assignment_costs)

        # Identify outliers
        inliers = [pair for pair, cost in zip(optimal_assignment, assignment_costs) if not(cost > threshold)]
        inliers = np.array(inliers)
        return inliers[:,0], inliers[:,1]

    def find_matching_waypoints(self, source_distribution, training_traj, distance_threshold):
        # Find the closest points in source_distribution to each point in training_traj
        distances, indices = cKDTree(source_distribution).query(training_traj)

        # Select indices in training_traj that meet the distance threshold
        mask_traj = np.where(distances < distance_threshold)[0]  # Indices in training_traj
        mask_source = indices[mask_traj]  # Corresponding indices in source_distribution

        # Only the pairs within the threshold are included, and the indices are ordered by mask_traj
        return mask_traj[::6], mask_source[::6] #, only choose certain points to sample
        # return mask_traj, mask_source
        # return mask_traj[[0, len(mask_traj) // 2, -1]], mask_source[[0, len(mask_source) // 2, -1]]  # Only choose the first and last points


    def fit(self, source_distribution, target_distribution, training_traj, distance_threshold,
            current_spline_index=0, keypoint=False, mask_traj=None, mask_source=None):
        self.training_traj=training_traj

        diff=np.zeros_like(training_traj)
        constraint= np.zeros_like(training_traj)

        if keypoint:
            if mask_traj is None:
                print("Using Hungarian")
                mask_traj, mask_source = self.find_matching_waypoints_hungarian(source_distribution, training_traj, distance_threshold)
            else:
                # continue
                print("Mask traj and mask source set outside function")
        else:
            mask_traj, mask_source= self.find_matching_waypoints(source_distribution, training_traj, distance_threshold)
        # print("Mask traj: ", mask_traj)
        # print("Mask source: ", mask_source)
        # Set class variables
        self.mask_traj = mask_traj
        self.mask_source = mask_source

        # Define constraints based on the difference between the target and source distributions
        diff[mask_traj]=target_distribution[mask_source] - source_distribution[mask_source]
        constraint[mask_traj]= training_traj[mask_traj]+ diff[mask_traj]

        # Using current_spline_index to constrain the point the robot is at
        index_to_constrain = current_spline_index
        constraint[index_to_constrain] = self.training_traj_old[index_to_constrain][:3]
        mask_traj = np.insert(mask_traj, 0, index_to_constrain)
        # Done using current_spline_index

        # Create the graph Laplacian and the Laplacian coordinates
        L, DELTA = self.create_graph(training_traj)

        # make a vector that has 1 in the index of mask_traj
        vect= np.zeros(len(training_traj))
        vect[mask_traj]=1
        P_hat=np.diag(vect)
        # P_hat is a diagonal matrix that has 1 in the index of mask_traj and 0 otherwise

        # We are now solving the following optimization problem
        # min ||L P_hat - DELTA||^2 + ||P_hat - constraint||^2 and this can be written as
        # min ||A P_s - B||^2 where A= [L; P_hat] and B= [DELTA; constraint] and P_s is the solution of the optimization problem.
        # Since it is a linear system with more constraint than variables, the solution is given by the pseudo inverse of A @ B
        weight_delta= 1
        weight_constraint= 5
        A = np.vstack([L * weight_delta, P_hat * weight_constraint])
        B = np.vstack([DELTA * weight_delta, constraint * weight_constraint])
        
        self.P_s, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

        # The solution is the new trajectory where the assigned nodes to the source are moved by the quantity specified by the difference between the target
        # and the source distribution

        # Check out much the solution of the syste respected the orginal contraint in B
        # print("Residuals: ", A @ self.P_s - B)
        # print("Residuals on Delta", 100*(L @ self.P_s - DELTA)/DELTA)
        # print("Residuals on constraint", 100*(P_hat @ self.P_s - constraint)/constraint)
        self.accuracy = np.sqrt(np.mean((P_hat @ self.P_s - constraint)**2))

    
    def predict(self, X, return_std=False):
        # assert that X adn self.training_traj have the same 
        # assert(np.allclose(X,self.training_traj)), " Laplacian editing can only predict the training trajectory"
        mean=self.P_s
        eps=1e-6
        if return_std:
            std = eps*np.ones_like(mean)
            return mean, std
        return mean
    
    def samples(self, X):
        # laplacian editing is deterministic, then we return the same sample
        predictions = [self.predict(X) for i in range(10)]
        predictions = np.array(predictions)  # Shape: (n_estimators, n_samples)
        return predictions