from pprint import pprint
from math import sqrt, floor
import numpy as np


def initial_centroids(k, data):
    n = np.shape(data)[1]
    # print(n)

    # The centroids
    centroids = np.mat(np.zeros((k, n)))
    # print(centroids)

    # Create random centroids (get min, max attribute values, randomize in that range)
    for j in range(n):
        min_centroid = min(data[:, j])
        max_centroid = max(data[:, j])
        range_centroid = float(max_centroid - min_centroid)
        centroids[:, j] = min_centroid + range_centroid * np.random.rand(k, 1)

    # pprint(centroids)
    # Return centroids as numpy array
    return centroids


def euclidean_distance(a, b):
    temp = (a - b) ** 2
    return np.sqrt(np.sum(temp))


def measure_distance(data, centroid):
    # distance = []

    for i in range(len(data)):
        for j in range(len(centroid)):
            # pprint(data[i])
            # pprint(centroid[j])
            A = np.squeeze(np.asarray(centroid[j]))
            # print(A)
            distance = euclidean_distance(data[i], A)
            print(distance)
    pass


def main():
    data = np.loadtxt("breast_cancer_data.txt", usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), delimiter=",")
    # pprint(data)
    k = [2, 3, 4, 5, 6, 7, 8]

    for i in range(1):
        print(k[i])
        centroid = initial_centroids(k[i], data)
        measure_distance(data, centroid)
        print(centroid)


if __name__ == "__main__" :
    main()