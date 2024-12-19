### TODO 1: Importing the necessary libraries - numpy, matplotlib and time
import pandas as pd
import numpy as np
import matplotlib as plt
import time 


### TODO 2
### Load data from data_path
### Check the input file spice_locations.txt to understand the Data Format
### Return : np array of size Nx2
def load_data(data_path):
    data = np.loadtxt(data_path , dtype = int)
    return data 
### TODO 3.1
### If init_centers is None, initialize the centers by selecting K data points at random without replacement
### Else, use the centers provided in init_centers
### Return : np array of size Kx2
def initialise_centers(data, K, init_centers=None):
    if init_centers is not None:
        return  init_centers
    indices = np.random.choice(data.shape[0],size = K,replace = False)
    return data[indices]

### TODO 3.2
### Initialize the labels to all ones to size (N,) where N is the number of data points
### Return : np array of size N
def initialise_labels(data):
    N = data.shape[0]
    labels = np.ones(N)
    return labels
    

### TODO 4.1 : E step
### For Each data point, find the distance to each center
### Return : np array of size NxK
def calculate_distances(data, centers):
    distances = np.zeros(data.shape[0],centers.shape[0])
    for i in range(data.shape[0]):
        for j in range(centers.shape[0]):
            distances[i][j] = np.sum((centers[j]-data[i])**2)
    return distances

### TODO 4.2 : E step
### For Each data point,ign the label of the nearest center
### Return : np array of size N
def update_labels(distances):
    return np.argmin(distances,axis=1)

### TODO 5 : M step
### Update the centers to the mean of the data pointsigned to it
### Return : np array of size Kx2
def update_centers(data, labels, K):
    centers = np.zeros((K, 2))
    for k in range(K):
        points = data[labels == k]
        if len(points) > 0:
           centers[k] = np.mean(points, axis=0)
        else:
          centers[k] = np.random.rand(2)
    return centers

### TODO 6 : Check convergence
### Check if the labels have changed from the previous iteration
### Return : True / False
def check_termination(labels1, labels2):
     return np.array_equal(labels1,labels2)

### simulate the algorithm in the following function. run.py will call this
### function with given inputs.
def kmeans(data_path:str, K:int, init_centers):
    '''
    Input :
        data (type str): path to the file containing the data
        K (type int): number of clusters
        init_centers (type numpy.ndarray): initial centers. shape = (K, 2) or None
    Output :
        centers (type numpy.ndarray): final centers. shape = (K, 2)
        labels (type numpy.ndarray): label of each data point. shape = (N,)
        time (type float): time taken by the algorithm to converge in seconds
    N is the number of data points each of shape (2,)
    '''
    data = load_data(data_path)
    start_time = time.time()
    centers = initialise_centers(data, K, init_centers)
    labels = initialise_labels(data)
    old_labels = None
    while True:
        old_labels = labels.copy()
        distances = calculate_distances(data, centers)
        labels = update_labels(distances)
        centers = update_centers(data, labels, K)
        if check_termination(labels, old_labels):
            break
    
    execution_time = time.time() - start_time
    return centers, labels, execution_time


### to visualise the final data points and centers.
def visualise(data_path, labels, centers):
    data = load_data(data_path)

    # Scatter plot of the data points
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    return plt
