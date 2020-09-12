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


# You may put code here to test your program. They will not be run during grading.
if __name__ == '__main__':
    pass