
# Required packages
import random
import numpy as np



# k-Means cluster algorithm
def k_means(data, num_k, max_iter):

    # define distance norm
    dist_eucledian = lambda x1, x2: np.sqrt(np.sum(np.square(np.subtract(x1, x2))))

    # arbitrarily choose k objects from D as the initial cluster centers
    cluster_centers = np.array(data[random.sample(range(len(data)), k=num_k)]) # TODO: is there a np func?

    # repeat until no change
    clustering = None  
    counter = 0
    changed = True
    while changed:
        counter += 1

        clustering = np.array([]) 

        # (re)assign each object to the cluster to which the object is most 
        # similar, based on the mean value of the objects in the cluster
        for object in data:
            # determine the most similar cluster center
            distances = np.array([])

            for center in cluster_centers:
                #print(dist_eucledian(center, object))
                distances = np.append(distances, dist_eucledian(center, object))

            # determine cluster (index)
            num_cluster = np.argmin(distances)
            
            # assign the object to the cluster with the most similar cluster center
            clustering = np.append(clustering, num_cluster)


        k_means = np.empty(shape=(len(cluster_centers), len(data[0]))) # new cluster centroids/centers

        # calculate the cluster centers by averaging the cluster elements
        for index in range(len(cluster_centers)):
            k_means[index] = np.mean(data[np.argwhere(clustering == index)], axis = 0)


        # determine if cluster centers have changed
        distances_cluster_centers = np.array([])
        for i in range(len(cluster_centers)):
            distance = dist_eucledian(cluster_centers[i], k_means[i])
            distances_cluster_centers = np.append(distances_cluster_centers, distance)

        # taking the sum of the elements to determine the overall change
        cluster_change = np.sum(distances_cluster_centers)
        
        # update the clusters
        cluster_centers = k_means

        if counter == max_iter or cluster_change == 0:
            changed = False

    return clustering




if __name__ == "__main__":
    from sklearn import datasets

    iris = datasets.load_iris()["data"]

    clustering = k_means(iris, num_k=3, max_iter=5) 

    print(clustering)


