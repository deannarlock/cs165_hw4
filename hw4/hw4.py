#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy

def read_data(path):
    """
    Read the input file and store it in data_set.

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        path: path to the dataset

    Returns:
        data_set: n_samples x n_features
            A list of data points, each data point is itself a list of features.
    """

    data_set = []
    data_point = []
    
    file = open(path,"r")
    lines = file.readlines()

    for line in lines:
        
        loc = line.rfind(',')

        feature = ''
        for i in range(loc):
            char = line[i]
            if(char != ','):
                feature = feature + char
            else:
                data_point.append(float(feature))
                feature = ''

        data_point.append(float(feature))
        data_point.append(float(line[loc + 1:-1]))
        
        data_set.append(data_point)
        data_point = []
    
    return data_set
    
    
def pca(data_set, n_components):
    """
    Perform principle component analysis and dimentinality reduction.
    
    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: n_samples x n_features
            The dataset, as generated in read_data.
        n_components: int
            The number of components to keep. If n_components is None, all components should be kept.

    Returns:
        components: n_components x n_features
            Principal axes in feature space, representing the directions of maximum variance in the data. 
            They should be sorted by the amount of variance explained by each of the components.
    """
    
    #zero center the data
    mean = []
    for i in range(len(data_set[0])):
        mean.append(0.0)
        
    for point in data_set:
        for i in range(len(point)):
            mean[i] = mean[i] + point[i];
    
    amt = len(data_set)
    for i in range(len(mean)):
        mean[i] = mean[i] / amt
        
    
    new_data = []
    for point in data_set:
        new_point = []
        for i in range(len(point)):
            new_point.append(point[i] - mean[i])
        new_data.append(new_point)
        
    
    #calculate covariance matrix
    transposed = []
    for feature in new_data[0]:
        blank = []
        blank.append(feature)
        transposed.append(blank)
    
    for point in new_data[1:]:
        for i in range(len(point)):
            transposed[i].append(point[i])
    
    
    S = np.cov(transposed, bias=True)
    
    
    #find eigenvectors and eigenvalues of covariance matrix
    V, U = np.linalg.eig(S)
    
    
    #find n PC's
    sorted_v = []
    for i in range(len(V)):
        sorted_v.append(V[i])
        
    sorted_v.sort(reverse=True)
    list_v = list(V)
    
    
    PC = []
    for i in range(n_components):
        loc = list_v.index(sorted_v[i])
        PC.append(list(U[:,loc]))
        
    
    return np.transpose(PC)
    

def dim_reduction(data_set, components):
    """
    perform dimensionality reduction (change of basis) using the components provided.

    DO NOT CHANGE SIGNATURE OF THIS FUNCTION

    Args:
        data_set: n_samples x n_features
            The dataset, as generated in read_data.
        components: n_components x n_features
            Principal axes in feature space, representing the directions of maximum variance in the data. 
            They should be sorted by the amount of variance explained by each of the components.

    Returns:
        transformed: n_samples x n_components
            Return the transformed values.
    """
    
    return np.dot(data_set, components)
    