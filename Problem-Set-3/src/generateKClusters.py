import sys
import csv
import json
import time
import pprint
import copy

import kmeans as kmeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arff2pandas import a2p
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from sklearn import preprocessing


def runKMeans(filename):
    k = 3
    epsilon = 0.01
    max_iterations = 10
    seed = 10
    normalize = True
    dataframe, classes, original_dataframe = parse_arff_file(filename, normalize)

    rows, columns = dataframe.shape
    clusters, original_centroids, final_centroids, num_iterations, runtime, error = kmeans.k_means_clustering(dataframe, k, max_iterations, epsilon, seed)

    print_k_means_data(original_dataframe, clusters, original_centroids, final_centroids, num_iterations, runtime, error, classes)
    plot_results(clusters)


def print_k_means_data(original_dataframe, clusters, original_centroids, final_centroids, num_iterations, runtime, error, classes):
    print("\nkMeans")
    print("======")
    print("\nNumber of iterations: {}".format(num_iterations))
    print("Within cluster sum of squared errors: {}".format(round(error, 3)))
    print("\nInitial starting points (random):")

    rows, columns = original_dataframe.shape
    headers = list()
    full_data_averages = list()
    for col in list(original_dataframe):
        headers.append(col)
        full_data_averages.append(original_dataframe[col].mean())

    for k,v in original_centroids.items():
        centroid_coordinates = ""
        for item in v:
            centroid_coordinates += (str(round(item, 2)) + ",")
        print("Cluster {}: {}".format(k+1, centroid_coordinates[:-1]))

    print("\nFinal cluster centroids:")
    table_str = "Attribute\tFull Data"
    for i in range(k+1):
        table_str += "\t{}".format(i+1)
    print(table_str)

    table_str = "\t\t({})\t".format(rows)
    for i in range(k+1):
        if i not in clusters:
            table_str += "\t({})".format(0)
        else:
            table_str += "\t({})".format(len(clusters[i]))
    print(table_str)

    print("=========================================================")

    for i in range(len(headers)):
        current_attribute = headers[i].split("@")[0]

        row_str = current_attribute + "\t" + str(round(full_data_averages[i], 2)) + "\t"
        for j in range(k+1):
            row_str += "\t" + str(round(final_centroids[j][0][i], 2))
        print(row_str)

    print("\nTime taken to build model (full training data) : {} seconds".format(round(runtime, 5)))
    
    print("\nClustered Instances")
    for i in range(k+1):
        if i not in clusters:
            print("{}\t{} ({} %)".format(i+1, 0, 0))
        else:
            print("{}\t{} ({} %)".format(i+1, len(clusters[i]), round(((len(clusters[i])/rows)*100), 2)))


def plot_results(clusters, features=['sepallength', 'sepalwidth', 'petallength']):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    species_arr = ['setosa', 'versicolor', 'virginica']
    colors = ['green', 'red', 'blue']

    for k,v in clusters.items():
        x_plt, y_plt, z_plt = [],[],[]

        for instance in v:
            x_plt.append(instance[1][0])
            y_plt.append(instance[1][1])
            z_plt.append(instance[1][2])
        print("Added {} instances to cluster {} to print in color {}.".format(len(x_plt),k,colors[k]))
        ax.scatter(x_plt, y_plt, z_plt, color=colors[k])

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])

    ax.legend()
    plt.show()


def parse_arff_file(filename, normalize):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    classes = []
    with open(filename) as file:
        df = a2p.load(file)

        try:
            df.iloc[:,-1] = df.iloc[:,-1].apply(int)
            classes = None
        except:
            le = preprocessing.LabelEncoder()
            le.fit(df.iloc[:,-1])
            classes = list(le.classes_)
            df.iloc[:,-1] = le.transform(df.iloc[:,-1]) 

        df = df.select_dtypes(include=numerics)

        original_dataframe = copy.deepcopy(df)
        if normalize:
            headers = df.columns
            x = df.values
            scaler = preprocessing.Normalizer()
            scaled_df = scaler.fit_transform(df)
            df = pd.DataFrame(scaled_df)
            df.columns=headers

        return df, classes, original_dataframe


if __name__ == '__main__':
    main()