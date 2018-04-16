from pprint import pprint
from math import sqrt, floor

import numpy as np
import operator
import matplotlib.pyplot as plt



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
    return min(distance.items(), key=operator.itemgetter(1))[0]


def find(assign_centroid):
    for k, v in assign_centroid.items():
        if not (len(v)>0):
            print(len(v))
    return {k: sum(v) / len(v) for k, v in assign_centroid.items()}
    pass


def measure_distance(data, centroid, k):
    # count1 = 0
    # count2 = 0
    # assigned_cluster = [[[] for i in range(k)]]

    assign_centroid = {}
    for i in range(len(data)):
        distance = {}
        for j in range(len(centroid)):
            # pprint(data[i])
            # pprint(centroid[j])
            if (len(np.asarray(centroid[j]))) > 0:
                A = np.squeeze(np.asarray(centroid[j]))
            distance[j] = euclidean_distance(data[i], A)
            # print(distance)
        a = assign_cluster(distance)
        # print(a)
        assign_centroid.setdefault(a, []).append(data[i])
        # assigned_cluster[a-1] = assign_centroid
        # print(assign_centroid)
    new_centroid = find(assign_centroid)




    # print(type(assign_centroid))
    #
    # m = np.asmatrix(assign_centroid)
    # pprint(m)
    #     assign_centroid = data[i]
    #     print(assign_centroid)

    #     if a == 0:
    #         count1 = count1 + 1
    #     else:
    #         count2 = count2 + 1
    #
    # print(count1)
    # print(count2)
    return new_centroid, assign_centroid


# def compareCentroid(previous_centroid, new_centroid):
#     # print("hi")
#     print(previous_centroid)
#     # print(len(previous_centroid))
#     print(new_centroid)
#     # print(len(new_centroid))
#     # for key, value in previous_centroid:
#     #     if key in new_centroid.items():
#     for key in previous_centroid.keys() & new_centroid.keys():
#         print("hello")
#         return False
#     return True

def compareCentroid(previous_centroid, new_centroid):
    # print(previous_centroid)
    # print(new_centroid)

    # print("HOLAaaa")
    # print(set(new_centroid) == set(previous_centroid))
    for key, value in previous_centroid.items():
        if not all(new_centroid[key] == value):
            return False
    return True


def calculateCentroids(data, new_centroid, k):
    previous_centroid = new_centroid
    new_centroid, revised_data = measure_distance(data, new_centroid, k)
    return previous_centroid, new_centroid


def checkCentroid(data, previous_centroid, new_centroid, k):
    # print(previous_centroid)
    # print(len(previous_centroid))
    # print(new_centroid)
    # print(len(new_centroid))
    flag = compareCentroid(previous_centroid, new_centroid)
    while True:
        # print(flag)
        if flag == False:
            previous_centroid, new_centroid = calculateCentroids(data, new_centroid, k)
            flag = compareCentroid(previous_centroid, new_centroid)
        else:
            break
    pprint(previous_centroid)
    return new_centroid


def calculatePotentialFunction(k, final_centroid, revised_data):
    potentialFunction = 0

    for i in range(k):
        for j in revised_data[i]:
            z = 0
            z = final_centroid[i] - j
            z = np.linalg.norm(z)
            z = z * z
            potentialFunction += z

    # print(potentialFunction)
    return potentialFunction


def graphPlot(graph,k):
    plt.plot(graph, 'bs-', label="K-Means")
    xdatapoints=[i for i in range(0, len(k))]
    plt.ylabel('Potential Function')
    plt.xlabel('Value of K')
    plt.xticks(xdatapoints,k)
    plt.legend()
    plt.show()


def main():
    data = np.loadtxt("breast_cancer_data.txt", usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), delimiter=",")
    # pprint(data)
    k = [2, 3, 4, 5, 6, 7, 8]
    potentialfunction = []
    for i in range(7):
        print(k[i])
        centroid = initial_centroids(k[i], data)
        # print(data)
        previous_centroid,revised_data = measure_distance(data, centroid, k[i])
        # print(previous_centroid)
        new_centroid, revised_data = measure_distance(data, previous_centroid,k[i])
        # print(new_centroid)
        # print(new_centroid)
        final_centroid = checkCentroid(data, previous_centroid, new_centroid,k[i])
        final_centroid_ans, revised_data = measure_distance(data, final_centroid, k[i])
        print("final")
        pprint(final_centroid_ans)
        # pprint(len(revised_data[0]))
        potentialfunction.append(calculatePotentialFunction(k[i], final_centroid_ans, revised_data))

    print(potentialfunction)
    graphPlot(potentialfunction, k)



        # while(checkCentroid(data,previous_centroid,k[i])):
        #     continue


if __name__ == "__main__" :
    main()