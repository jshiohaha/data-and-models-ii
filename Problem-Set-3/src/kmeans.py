import sys
import copy
import math
import time
import random
import pandas as pd
import numpy as np


def k_means_clustering(dataframe, k, max_iterations, epsilon, seed):
    num_iterations = 1
    rows, columns = dataframe.shape
    old_sse, new_sse = sys.maxsize, 0
    df_as_arr = dataframe.reset_index().values
    min_max = [(dataframe[column].min(), dataframe[column].max()) for column in list(dataframe)]

    np.random.seed(seed)
    centroids = {
        i: [np.random.uniform(min_max[idx][0], min_max[idx][1]) for idx in range(len(list(dataframe)))]
        for i in range(k)
    }
    
    original_centroids = copy.deepcopy(centroids)

    start = time.time()
    while(num_iterations <= max_iterations):
        clusters = {}

        for instance in df_as_arr:
            instance_idx = instance[0]
            instance = instance[1:]

            min_dist, cluster_id = dist(instance, centroids)
            if cluster_id not in clusters.keys():
                clusters[cluster_id] = list()
            clusters[cluster_id].append([instance_idx, instance])

        old_sse = new_sse
        new_sse = 0
        for k,list_of_instances in clusters.items():
            centroid_sum = np.zeros(columns)
            centroid_size = len(list_of_instances)

            for instance in list_of_instances:
                instance = instance[1:]
                centroid_sum = np.add(centroid_sum, instance)

            centroids[k] = [(centroid_sum[i] / centroid_size) for i in range(len(centroid_sum))]
            new_sse += np.sum([np.linalg.norm((np.array(instance[1:])-centroids[k]))**2 for instance in list_of_instances])

        if(math.fabs(old_sse - new_sse) < epsilon):
            end = time.time()
            return clusters, original_centroids, centroids, num_iterations, (end-start), new_sse

        num_iterations += 1

    end = time.time()
    return clusters, original_centroids, centroids, num_iterations, (end-start), new_sse


def dist(instance, centroids):
    min_dist = sys.maxsize
    min_dist_key = 0

    for k,v in centroids.items():
        d = np.linalg.norm(np.array(instance).reshape(1,-1)-np.array(v).reshape(1,-1))

        if d < min_dist:
            min_dist = d
            min_dist_key = k

    return min_dist, min_dist_key
