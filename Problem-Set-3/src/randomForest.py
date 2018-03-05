import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def runRandomForest():
    # read train data
    train = pd.read_csv("../data/mnist/train.csv")
    train_df, test_df = train_test_split(train, test_size = 0.25)

    # Print labels to ensure equal numbers
    print(Counter(list(train_df['label'])))

    # read test data
    test = pd.read_csv("../data/mnist/test.csv")

    # Use only split training set for problem set
    rand_for1 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state = 1)
    rand_for1.fit(train_df.drop('label', axis=1),train_df['label'])
    predictions1 = rand_for1.predict(test_df.drop('label', axis=1))

    # Find accuracy
    test_results = list(test_df['label'])
    correct = 0
    for i in range(len(predictions1)):
        if predictions1[i] == test_results[i]:
            correct += 1
    accuracy = correct/len(test_df)
    data_str = "Accuracy:\t{}".format(accuracy)
    print(data_str)

    # Create confusion table
    cm = confusion_matrix(y_true = list(test_df['label']), y_pred = predictions1, labels=[0,1,2,3,4,5,6,7,8,9], sample_weight=None)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len([0,1,2,3,4,5,6,7,8,9]))
    plt.xticks(tick_marks, [0,1,2,3,4,5,6,7,8,9], rotation=45)
    plt.yticks(tick_marks, [0,1,2,3,4,5,6,7,8,9])

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Use all training set data for Kaggle challenge
    rand_for2 = RandomForestClassifier(n_estimators=100, n_jobs=2, random_state = 1)
    rand_for2.fit(train.drop('label', axis=1),train['label'])
    predictions2 = rand_for2.predict(test)

    numId = []
    for i in range(len(test)):
        numId.append(i+1)
    answer = np.column_stack([numId, predictions2])

    np.savetxt("../data/digit_recognizer_output_2.csv", answer, fmt="%.0f", delimiter=",", header="ImageId,Label")
