from pprint import pprint
from math import sqrt, floor
import numpy as np
import operator


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


def assign_cluster(distance):
    return max(distance.items(), key=operator.itemgetter(1))[0]


def measure_distance(data, centroid):
    # count1 = 0
    # count2 = 0
    for i in range(len(data)):
        distance = {}
        for j in range(len(centroid)):
            # pprint(data[i])
            # pprint(centroid[j])
            A = np.squeeze(np.asarray(centroid[j]))
            # print(A)
            distance[j] = euclidean_distance(data[i], A)
            print(distance)
        a = assign_cluster(distance)
        print(a)

    #     if a == 0:
    #         count1 = count1 + 1
    #     else:
    #         count2 = count2 + 1
    #
    # print(count1)
    # print(count2)


    pass


def main():
    data = np.loadtxt("breast_cancer_data.txt", usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), delimiter=",")
    # pprint(data)
    k = [2, 3, 4, 5, 6, 7, 8]

    for i in range(7):
        print(k[i])
        centroid = initial_centroids(k[i], data)
        measure_distance(data, centroid)
        print(centroid)


if __name__ == "__main__" :
    main()