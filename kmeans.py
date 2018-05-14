import scipy.spatial
import numpy as np


class KMeans(object):
    """docstring for Kmeans"""
    def __init__(self, n_clusters, fixed_centroids = {}, max_iters=500):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        # {index:[x1, x2, x3, ... xN], index:[x1, x2, x3, ... xN], ...} 
        # e.g. {0:[0,0]} only fix the first centroid at (0,0) for two-dimensional data
        self.fixed_centroids = fixed_centroids
        self.centroids = None
        self.distance = None
        self.X = None
        self.labels = None
        self.max_iters = max_iters
        
    def compute_distance(self):
        self.distance = scipy.spatial.distance.cdist(self.X, self.centroids)
        self.labels = self.distance.argmin(axis=1)

    def update_centroids(self):
        for i in range(self.centroids.shape[0]):
            idx, = np.where(self.labels == i)
            if idx.any():
                self.centroids[i] = self.X[idx].mean(axis=0)

    def fit(self, X):
        self.X = X
        iter_cnt = 0
        sorted_X = self.X[np.argsort(np.linalg.norm(self.X, axis=1))]
        self.centroids = sorted_X[[(i+2)*len(sorted_X)/(self.n_clusters+2) for i in range(self.n_clusters)]]
        while iter_cnt < self.max_iters:
            for idx in self.fixed_centroids.keys():
                self.centroids[idx] = np.array(self.fixed_centroids[idx])
            old_centroids = self.centroids.copy()
            self.compute_distance()
            self.update_centroids()
            if (self.centroids == old_centroids).all():
                break
            iter_cnt += 1

        for idx in self.fixed_centroids.keys():
            self.centroids[idx] = np.array(self.fixed_centroids[idx])
        self.compute_distance()

    def predict(self, X):
        return scipy.spatial.distance.cdist(X, self.centroids).argmin(axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels
