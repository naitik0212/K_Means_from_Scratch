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
        range_centroid = float(max_centroid - min_centroid*1.2)
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
            pass
    return {k: sum(v) / len(v) for k, v in assign_centroid.items()}
    pass


def splitLargest(assign_centroid,list):
    for i in list:
        max_index = -1
        max_length = -1
        for cluster in assign_centroid:
            if(len(assign_centroid(cluster))>max_length):
                max_index = cluster
                max_length = len(assign_centroid(cluster))
        # print(max_length,max_index)
        temp = assign_centroid[max_index]
        del assign_centroid[max_index]
        checkpoint = len(temp)/2
        assign_centroid[i]=temp[:checkpoint]
        assign_centroid[max_index]=temp[checkpoint:]
        # print(type(temp))
    return assign_centroid
    pass


def measure_distance(data, centroid, k):

    assign_centroid = {}
    for i in range(len(data)):
        distance = {}
        for j in range(len(centroid)):
            # pprint(data[i])
            # pprint(centroid[j])
            # print(len(np.asarray(centroid[j])))
            A = np.squeeze(np.asarray(centroid[j]))
            # print(type(centroid[j]),np.shape(centroid[j]))
            # print(type(data[i]),np.shape(data[i]))
            distance[j] = euclidean_distance(data[i], A)
            # print(distance)
        a = assign_cluster(distance)
        # print(a)

        assign_centroid.setdefault(a, []).append(data[i])

    # print(assign_centroid)
    list = []
    for cluster in assign_centroid:
        # print(len(assign_centroid[cluster]))
        # print(len(assign_centroid[i]))
        # print(all(assign_centroid[i]))
        if(len(assign_centroid[cluster])==0):
            list.append(cluster)
            del assign_centroid[cluster]

    if(len(list)>0):
        splitLargest(assign_centroid,list)



    # list_length = [0 in range(k)]
    #
    # for i in range(k):
    #     # print(max(list_length))
    #     if(len(assign_centroid[i]))==0 :
    #         assign_centroid[i] = assign_centroid[list_length.index(max(list_length))]
    #     list_length.append(len(assign_centroid[i]))

    # for i in range(k):
    #     if(min(list_length)==0):
    #         assign_centroid[i] = assign_centroid[list_length.index(max(list_length))]
    #
    # print(list_length)
    # print(list_length.index(max(list_length)))

    # if len(list_length[i])!= 0:
        #     pass
        # else:
        #     assign_centroid[i] = assign_centroid[i-1]
    new_centroid = find(assign_centroid)

    return new_centroid, assign_centroid


def compareCentroid(previous_centroid, new_centroid):

    for key, value in previous_centroid.items():
        if not all(new_centroid[key] == value):
            return False
    return True


def calculateCentroids(data, new_centroid, k):
    previous_centroid = new_centroid
    new_centroid, revised_data = measure_distance(data, new_centroid, k)
    return previous_centroid, new_centroid


def checkCentroid(data, previous_centroid, new_centroid, k):

    flag = compareCentroid(previous_centroid, new_centroid)
    while True:
        if flag == False:
            previous_centroid, new_centroid = calculateCentroids(data, new_centroid, k)
            flag = compareCentroid(previous_centroid, new_centroid)
        else:
            break
    # pprint(previous_centroid)
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
    plt.xticks(xdatapoints, k)
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
        previous_centroid, revised_data = measure_distance(data, centroid, k[i])
        new_centroid, revised_data = measure_distance(data, previous_centroid, k[i])
        final_centroid = checkCentroid(data, previous_centroid, new_centroid, k[i])
        final_centroid_ans, revised_data = measure_distance(data, final_centroid, k[i])
        # print("final")
        pprint(final_centroid_ans)
        potentialfunction.append(calculatePotentialFunction(k[i], final_centroid_ans, revised_data))

    print(potentialfunction)
    graphPlot(potentialfunction, k)


if __name__ == "__main__" :
    main()