import os, json
import random
import numpy

class Cluster(object):

    def __init__(self, dim_count):
        ''' 
            dim_count: int
                The number of dimensions per point (i.e. vector3 would have 3 dims)
        '''
        self.dim_count = dim_count
        self.points = []
        self.point_ids = []
        self.centroid = None

    def addPoint(self, point):
        self.point_ids.append(point[0])
        self.points.append(point[1])

    def setNewCentroid(self):
        # Sum the points
        self.centroid = [0 for i in range(self.dim_count)]
        for point in self.points:
            for dim_index, value in enumerate(point):
                self.centroid[dim_index] += value

        # divide it by the number of points
        for i, _ in enumerate(self.centroid):
            self.centroid[i] /= len(self.points)

        return self.centroid

    def clearPoints(self):
        self.points = []
        self.point_ids = []

    def SortByLabel(self):
        pass

class Kmeans(object):
    
    def __init__(self, k=4, max_iterations=12, min_distance=0.1, size=200, verbose=0):
        ''' 
            Params:
            ------
            k: int
                The number of clusters
            ------
            max_iterations: int
                The maximum number of iterations the K means clustering should go through
            ------
            min_distance: float
                Theshold distance between the cluster iterations where it could stop running early
            ------
            size: int
                The number of dimensions per point (i.e. vector3 would have 3 dims)
            ------
            verbose:

        '''
        self.k = k
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.size = size
        self.verbose = verbose

    def run(self, points):
        ''' 
            Params
            ------
            points: List(tuple(id, point))
        '''
        # Grab just the points
        self.points = points
        # self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)

        self.clusters = []
        self.prev_centroids = None

        # initialize random points
        random_ponts = random.sample(list(list(zip(*self.points))[1]), self.k)
        for k_i in range(self.k):
            self.clusters.append(Cluster(self.size))
            self.clusters[k_i].centroid = random_ponts[k_i]

        iterations = 0

        while not self.shouldExit(iterations):
            for cluster in self.clusters:
                cluster.clearPoints()
            
            self.prev_centroids = [cluster.centroid for cluster in self.clusters]

            print("iter:", iterations)

            for point in self.points:
                self.assignClusters(point)

            for cluster in self.clusters:
                cluster.setNewCentroid()

            iterations += 1

        return [cluster.centroid for cluster in self.clusters]

    def assignClusters(self, point):
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(numpy.array(cluster.centroid), numpy.array(point[1]))
            if distance < shortest:
                shortest = distance
                nearest = cluster

        nearest.addPoint(point)

    def calcDistance(self, a, b):
        ''' Find the distances between two numpy points
        '''
        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    def shouldExit(self, iterations):

        if self.prev_centroids is None:
            return False

        total_dist = 0
        for idx in range(self.k):
            dist = self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.prev_centroids[idx])
            )
            total_dist += dist
        if total_dist < self.min_distance:
            return True

        if iterations <= self.max_iterations:
            return False

        return True



if __name__ == '__main__':
    km = Kmeans(k=2, size=3)

    points = [
        (0, [1,1,1]),
        (1, [2,1,1]),
        (2, [10,10,1]),
        (3, [11,10,1])
    ]

    clusters = km.run(points)
    # print("clusters", clusters)
    for cluster in km.clusters:
        # print(cluster.point_ids)
        for point in cluster.points:
            print(cluster.centroid, point[0])
