import numpy as np
import torch
from scipy.spatial.distance import cdist

def clip_eigenvalue(A, epsilon=1e-5):
    """ By eigen decomposition """
    A = A.astype(np.float64)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A) # eigen value of symmetric matrix
    
    # Find the smallest eigenvalue
    lambda_min = np.min(eigenvalues)
    
    # Find index of non-positive eigen values
    msk_neg = eigenvalues <= 0 
    
    # Modify the smallest eigenvalue to epsilon (ensure it's positive)
    modified_lambda_min = max(epsilon, lambda_min)
    
    # Create a diagonal matrix with modified eigenvalues
    eigenvalues[msk_neg] = modified_lambda_min
    
    # Reconstruct the matrix with modified eigenvalues
    # modified_A = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
    modified_A = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))

    eigenvalues, eigenvectors = np.linalg.eigh(modified_A)
    if np.min(eigenvalues) <= 0:
        print(np.min(eigenvalues))
        import ipdb;ipdb.set_trace() # breakpoint 28
    return modified_A

def angular_distance(rot_matrices):
    # NOTE: normalize rot matrices
    # q, r = torch.linalg.qr(rot_matrices)
    # d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    # q *= d.unsqueeze(-1)
    # rot_matrices = q

    # rot_matrices is expected to be of shape [n, 3, 3]
    n = rot_matrices.shape[0]
    
    # Compute the transpose of all rotation matrices
    rot_trans = rot_matrices.transpose(1, 2)  # shape becomes [n, 3, 3]
    
    # Compute the product of each pair of rotation matrices
    # This results in a tensor of shape [n, n, 3, 3]
    product = torch.matmul(rot_trans[:, None, :, :], rot_matrices[None, :, :, :])

    # Compute the trace for each product matrix, resulting in shape [n, n]
    traces = torch.einsum('nijj->ni', product) 
    
    # Calculate the angular distance
    cos_sim = torch.clamp((traces - 1) / 2.0, -1, 1) # remove floating point inprecision
    cos_sim_normalized = (cos_sim + 1) / 2
    # angular_distances = torch.acos(cos_sim)
    
    return cos_sim_normalized

def dist_gaussian_kernel(X, sigma):
    pairwise_sq_dists = cdist(X, X, 'sqeuclidean')
    return np.exp(-pairwise_sq_dists / (2. * sigma ** 2))