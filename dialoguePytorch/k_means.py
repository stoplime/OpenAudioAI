from PIL import Image, ImageChops
import os, json, cv2
import random
import numpy
from tkinter import Tk, filedialog


class Cluster(object):

    def __init__(self):
        self.pixels = []
        self.centroid = None

    def addPoint(self, pixel):
        self.pixels.append(pixel)

    def setNewCentroid(self):

        R = [colour[0] for colour in self.pixels]
        G = [colour[1] for colour in self.pixels]
        B = [colour[2] for colour in self.pixels]

        if len(R)!=0:
            R = sum(R) / len(R)
        else:
            R = 0
        if len(G)!=0:
            G = sum(G) / len(G)
        else:
            G = 0
        if len(B)!=0:
            B = sum(B) / len(B)
        else:
            B = 0

        self.centroid = (R, G, B)
        self.pixels = []

        return self.centroid


class Kmeans(object):

    def __init__(self, k=4, max_iterations=12, min_distance=0.1, size=1000, verbose=0):
        self.k = k
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.size = (size, size)
        self.verbose = verbose

    def run(self, image):
        self.image = image
        self.image.thumbnail(self.size)
        self.pixels = numpy.array(image.getdata(), dtype=numpy.uint8)

        self.clusters = [None for i in range(self.k)]
        self.oldClusters = None

        randomPixels = random.sample(list(self.pixels), self.k)

        for idx in range(self.k):
            self.clusters[idx] = Cluster()
            self.clusters[idx].centroid = randomPixels[idx]

        iterations = 0

        while self.shouldExit(iterations) is False:

            self.oldClusters = [cluster.centroid for cluster in self.clusters]

            print(iterations)

            for pixel in self.pixels:
                self.assignClusters(pixel)

            for cluster in self.clusters:
                cluster.setNewCentroid()

            iterations += 1

        return [cluster.centroid for cluster in self.clusters]

    def assignClusters(self, pixel):
        shortest = float('Inf')
        for cluster in self.clusters:
            distance = self.calcDistance(cluster.centroid, pixel)
            if distance < shortest:
                shortest = distance
                nearest = cluster

        nearest.addPoint(pixel)

    def calcDistance(self, a, b):

        result = numpy.sqrt(sum((a - b) ** 2))
        return result

    def shouldExit(self, iterations):

        if self.oldClusters is None:
            return False

        for idx in range(self.k):
            dist = self.calcDistance(
                numpy.array(self.clusters[idx].centroid),
                numpy.array(self.oldClusters[idx])
            )
            if dist < self.min_distance:
                return True

        if iterations <= self.max_iterations:
            return False

        return True

    def showImage(self):
        if self.verbose > 2:
            self.image.show()

        return self.image

    def showCentroidColours(self):

        for cluster in self.clusters:
            print('tuple cluster.centroid:', cluster.centroid)
            print('list cluster.centroid:', list(cluster.centroid))
            cluster.centroid = [ int(x) for x in list(cluster.centroid) ]
            print('int list cluster.centroid:', cluster.centroid)
            print('int list cluster.centroid oneliner:', tuple([int(x) for x in list(cluster.centroid)]))
            print('int tuple cluster.centroid:', tuple(cluster.centroid))
            image = Image.new("RGB", (200, 200), tuple([int(x) for x in list(cluster.centroid)]))
            image.show()

    def showClustering(self):

        localPixels = [None] * len(self.image.getdata())

        for idx, pixel in enumerate(self.pixels):
            shortest = float('Inf')
            for cluster in self.clusters:
                distance = self.calcDistance(
                    cluster.centroid,
                    pixel
                )
                if distance < shortest:
                    shortest = distance
                    nearest = cluster

            localPixels[idx] = nearest.centroid

        w, h = self.image.size
        localPixels = numpy.asarray(localPixels)\
            .astype('uint8')\
            .reshape((h, w, 3))

        colourMap = Image.fromarray(localPixels)
        # colourMap.show()

        return colourMap

if __name__ == '__main__':
    suffix = '.JPG'
    Tk().withdraw()
    path_to_json = filedialog.askdirectory() # show file explorer and return the path to the selected file
    if path_to_json is not None:
        json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
        for index, js in enumerate(json_files):
            img = Image.open(os.path.join(path_to_json, os.path.splitext(js)[0] + suffix))
            image = Kmeans()
            # img = cv2.imread(os.path.join(path_to_json, os.path.splitext(js)[0] + suffix))
            image.run(img)
            original = image.showImage()
            colourMap = image.showClustering()
            # image.showCentroidColours()
            diff = ImageChops.subtract(original, colourMap)
            diff.show()
            input("Press Enter to continue...")
    # img = Image.open(r'C:\Users\malessan\Downloads\giraffe.JPG')
    # image = Kmeans()
    # image.run(img)
    # original = image.showImage()
    # colourMap = image.showClustering()
    # # image.showCentroidColours()
    # input("Press Enter to continue...")