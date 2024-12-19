from k_means import *

data_path = 'data.txt'
K, init_centers = 5, None  
centers, labels, time_taken = kmeans(data_path, K, init_centers)
print('Time taken for the algorithm to converge:', time_taken)
visualise(data_path, labels, centers)