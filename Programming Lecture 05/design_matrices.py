"""
This file holds functions that form design matrices
"""
import numpy as np

def gaussian_rbf(x, z, scale = 1):
    return np.exp(-np.linalg.norm(x-z)**2/scale**2))
    

def form_gaussian_rbf_design_matrix(X, Z):
    """
    This forms the design matrix using Guassian radial basis functions
    
    :param X: a numpy 2D array containing inputs. Each row corresponds to
        a data point. Each column corresponds to an input variable
        
    :param Z: a numpy 2D array containiing the fixed points for the Guassian
    radial basis function. For each fixed point, we get a basis function.
    
    :returns: a design matrix using Gaussian RBFs as basis functions
    """

    # the number of basis functions = number of fixed points
    n = len(Z)
    
    # the number of data points
    m = len(X)
    
    # allocate an m x n matrix
    Phi = np.zeros([m, n])
    
    # iterate over points
    for i,x in enumerate(X):
        
        # iterate over basis functions 
        for j,z in enumerate(Z):
            
            # the (i,j)-th element of Phi is the j-th basis function
            # applied to the i-th data point
            
            Phi[i,j] = gaussian_rbf(x, z)
    
    return Phi
