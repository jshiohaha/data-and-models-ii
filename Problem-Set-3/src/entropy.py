import sys
import csv
import math

import numpy as np
import pandas as pd


'''
Prints the information gain values for each partition (or column).
'''
def runEntropy(df):

    # Get Y and labels:
    labels = list(df.axes[1])
    y_label = labels[0]
    Y = df[y_label]
    del labels[0]

    #Get information gain and loss for each partition:
    data_str = "Partition\t\tInformation Gain"
    for i in labels:
        data_str += "\n{}:\t\t\t{}".format(i, infoGainLoss(df[i],Y))

    #Print data
    print(data_str)

'''
Computes the entropy of the Y column.
'''
def getSetEntropy(Y):
    partitionEntropy = 0
    p_1 = sum(Y)/len(Y)
    p_2 = 1-p_1
    if p_2 == 0:
        partitionEntropy = 0
    elif p_1 == 0:
        partitionEntropy = 1
    else:
        partitionEntropy = -((p_1*math.log(p_1,2)) + (p_2*math.log(p_2,2)))
    return(partitionEntropy)

'''
Computes the information gain, which is simply the entropy of the set minus
the combined entropies of each result in a partition (or a column).
'''
def infoGainLoss(X,Y):
    setEntropy = getSetEntropy(Y)
    part_dict = getPartitionEntropy(X,Y)
    set_xresults = set(X)
    part_fracs = set()
    for p in set_xresults:
        total = 0
        numPart = 0
        for i in range(len(X)):
            total += 1
            if X[i] == p:
                numPart += 1
        try:
            if not math.isnan(float(p)):
                part_fracs.add((numPart/total)*part_dict[p])
        except:
            part_fracs.add((numPart/total)*part_dict[p])
    return(setEntropy - sum(part_fracs))

'''
Gets Entropy for each value in a partitioned column and returns
them in a dictionary used to compute the combined partition entropy
in infoGainLoss.
'''
def getPartitionEntropy(X,Y):
    set_xresults = set(X)
    partitionEntropy = 0
    part_dict = {}
    for p in set_xresults:
        numTotal = 0
        numCorrect = 0
        for i in range(len(X)):
            if X[i] == p:
                numTotal += 1
                if Y[i] == 1:
                    numCorrect += 1
        if numTotal == 0:
            p_1 = 0
            p_2 = 1
        else:
            p_1 = numCorrect/numTotal
            p_2 = 1-p_1

        if p_2 == 0:
            partitionEntropy = 0
        elif p_1 == 0:
            partitionEntropy = 1
        else:
            partitionEntropy = -((p_1*math.log(p_1,2)) + (p_2*math.log(p_2,2)))

        part_dict[p] = partitionEntropy
    return part_dict
