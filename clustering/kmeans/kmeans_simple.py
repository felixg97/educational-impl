
# Required packages
import math
import random
import numpy as np
from sklearn import cluster


# definition of the eucledian distance measure / norm
def eucledian_distance(x_1, x_2):
    sum_squared_differences = 0

    # summation of the squared differences
    for i in range(len(x_1)):
        sum_squared_differences += (x_1[i] - x_2[i])**2

    # taking the square root of the sum of squared differences
    distance = math.sqrt(sum_squared_differences)

    return distance



# k-Means cluster algorithm
def k_means(data, num_k, max_iterations, distance_measure):

    # arbitrarily choose k objects from D as the initial cluster centers
    random_indices = random.sample(range(len(data)), k=num_k)
    cluster_centers = data[random_indices] 

    # repeat until no change
    clusters = None
    changed = True
    counter = 0
    while changed:
        counter += 1
        
        # initialize an array with num_k empty arrays
        clusters = [[] for i in range(num_k)] # CRITIQUE: List comprehension

        k_means = [] # new cluster centroids/centers

        # (re)assign each object to the cluster to which the object is most 
        # similar, based on the mean value of the objects in the cluster
        for object in data:
            # determine the most similar cluster center
            distances_to_cluster_centers = []
            for cluster_center in cluster_centers:
                distance = distance_measure(cluster_center, object)
                distances_to_cluster_centers.append(distance)
            
            # determine cluster (index)
            num_cluster = np.argmin(distances_to_cluster_centers)

            # assign the object to the cluster with the most similar cluster 
            # center
            clusters[num_cluster].append(object)

        # calculate the cluster centers by averaging the cluster elements
        for cluster in clusters:
            cluster_mean = np.mean(cluster, axis=0) # takes mean of each column
            k_means.append(cluster_mean)

        # determine if cluster centers have changed
        distances_cluster_centers = []
        for i in range(len(cluster_centers)):
            distance = distance_measure(cluster_centers[i], k_means[i])
            distances_cluster_centers.append(distance)

        # taking the sum of the elements to determine the overall change
        cluster_change = np.sum(distances_cluster_centers)
        
        # update the clusters
        cluster_centers = k_means

        if counter == max_iterations or cluster_change == 0:
            changed = False
        
    return clusters




if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.cluster import KMeans

    iris = datasets.load_iris()["data"]

    clusters = k_means(iris, 4, None, 100, eucledian_distance)

    print(clusters)

    # k_means = KMeans(n_clusters=4)

    # clusters = k_means.fit_predict(iris)

    # print(clusters)



