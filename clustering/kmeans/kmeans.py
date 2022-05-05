import random
import numpy as np

from distance_measures import eucledian_distance
from distance_measures import manhattan_distance


class KMeans:

    def __init__(self, num_k, distance_measure=None, max_iter=None, verbose=False):
        self.verbose = verbose
        self.num_k = num_k
        self.max_iter = max_iter
        
        if distance_measure == "eucledian" or None:
            self.distance_measure = eucledian_distance
        elif distance_measure == "manhattan":
            self.distance_measure = manhattan_distance
            

    def fit_predict(self, data):

        # arbitrarily choose k objects from D as the initial cluster centers
        random_indices = random.sample(range(len(data)), k=self.num_k)
        cluster_centers = data[random_indices] 

        # repeat until no change
        clusters = None
        changed = True
        counter = 0
        while changed:
            counter += 1
            
            # initialize an array with num_k empty arrays
            clusters = [[] for i in range(self.num_k)] # CRITIQUE: List comprehension 

            k_means = [] # new cluster centroids/centers

            # (re)assign each object to the cluster to which the object is most 
            # similar, based on the mean value of the objects in the cluster
            for object in data:
                # determine the most similar cluster center
                distances_to_cluster_centers = []
                for cluster_center in cluster_centers:
                    distance = self.distance_measure(cluster_center, object)
                    distances_to_cluster_centers.append(distance)
                
                # determine cluster (index)
                num_cluster = np.argmin(distances_to_cluster_centers)

                # assign the object to the cluster with the most similar cluster 
                # center
                clusters[num_cluster].append(object)

            # determine if cluster centers have changed
            distances_cluster_centers = []
            for i in range(len(cluster_centers)):
                distance = self.distance_measure(cluster_centers[i], k_means[i])
                distances_cluster_centers.append(distance)

            # taking the sum of the elements to determine the overall change
            cluster_change = np.sum(distances_cluster_centers)

            # update the cluster centers by averaging the cluster elements
            for cluster in clusters:
                cluster_mean = np.mean(cluster, axis=0) # takes mean of each column
                k_means.append(cluster_mean)

            cluster_centers = k_means

            if counter == self.max_iter or cluster_change == 0:
                changed = False
            
        return clusters


    def fit_predict_with_history(self, data):

        history = []

        # arbitrarily choose k objects from D as the initial cluster centers
        random_indices = random.sample(range(len(data)), k=self.num_k)
        cluster_centers = data[random_indices] 

        # repeat until no change
        clusters = None
        changed = True
        counter = 0
        while changed:
            counter += 1
            
            # initialize an array with num_k empty arrays
            clusters = [[] for i in range(self.num_k)] # CRITIQUE: List comprehension 

            k_means = [] # new cluster centroids/centers

            # (re)assign each object to the cluster to which the object is most 
            # similar, based on the mean value of the objects in the cluster
            for object in data:
                # determine the most similar cluster center
                distances_to_cluster_centers = []
                for cluster_center in cluster_centers:
                    distance = self.distance_measure(cluster_center, object)
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
                distance = self.distance_measure(cluster_centers[i], k_means[i])
                distances_cluster_centers.append(distance)

            # taking the sum of the elements to determine the overall change
            cluster_change = np.sum(distances_cluster_centers)
            
            # update the clusters
            cluster_centers = k_means


            # save the current state
            history.append({
                "iteration": counter,
                "cluster_centers": cluster_centers,
                "clusters": clusters,
                "cluster_change": cluster_change,
            })

            if counter == self.max_iter or cluster_change == 0:
                changed = False
            
        return clusters


