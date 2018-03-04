import sys
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.cluster import KMeans

import generateKClusters


'''
    TODO:

    1.  Write a general function that computes the information entropy for a data set in a parent node and the
        aggregate information entropy and information gain for any number of partitions (subsets) of that data set in
        child nodes. The response variable is Bernoulli e.g., play tennis or do not play tennis.
        Test your function and output results for each partitioning in the example in this video. Note that Prof Patrick
        Winston is a super famous MIT professor. And yes, you will have to watch the entire lecture.
        (https://www.youtube.com/watch?v=SXBG3RGr_Rc)
    3.  Do the Kaggle / MNIST digit recognizer challenge (or at least a subset that will run on your computer)
        (http://yann.lecun.com/exdb/mnist/) See the assignment document for the details.

    4.  Use the kmeans() unction. An extra point will be awarded for the team that has the best home grown code performance (time 
        to solution) relative to kmeans() run on the same computer. Provide a complete narrative of your data science and 
        machine learning solution process. Provide a study of the optimal number of clusters using the total within-ness 
        mean squared error. Show all code and display the clusters using 3-D scatterplots.

    5.  Use the mushroom data set from the UCI Machine Learning Repository to predict whether mushrooms are edible or poisonous.
'''
def main():
    # titanic()
    # customKMeans()
    # builtInKMeans()
    naiveBayes()
    return


''' ----- BEGIN TITANIC ----- '''
def titanic():
    # Get training data
    filename = "../data/titanic_train.csv"
    train = load_data_frame(filename)

    # Split to X and Y
    X_train, Y_train = split_data_frame(train, "Survived")

    X_train_imp = prepare_titanic(X_train)

    # Create random Forest Classifier, fit
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_imp, Y_train)

    # Get testing data
    filename = "../data/titanic_test.csv"
    X_test = load_data_frame(filename)

    X_test_imp = prepare_titanic(X_test)

    # Store test predictions as Y
    Y_test = clf.predict(X_test_imp)

    # Convert Pandas Series to Numpy Array
    passenger_id = np.array(X_test["PassengerId"], dtype=pd.Series)

    # Combine Passenger_id with Y_test
    answer = np.column_stack((passenger_id, Y_test))

    # Save output to csv file
    np.savetxt("../data/titanic_output.csv", answer, fmt="%.0f", delimiter=",", header="PassengerId,Survived")

    return


''' ----- TITANIC UTILITY FUNCTIONS ----- '''
def split_data_frame(df, label):
    X = df.drop([label], axis=1)
    Y = df[label]
    return X, Y


def prepare_titanic(df):
    # Encode the Sex and Embarked objects
    df["Sex"] = encode_column(df, "Sex")
    df["Embarked"] = encode_column(df, "Embarked")

    # Convert objects to useable encoded value
    df = convert_title(df)
    df = convert_deck(df)

    # Drop the objects that can't be encoded, convert floats to float32
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1).astype(np.float32)

    # Create and imputer that fills in NaN's with the mean value
    imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imp = imp.fit(df)
    df_imp = imp.transform(df)

    return df_imp


def encode_column(df, label):
    binary_encoding = pd.get_dummies(df[label])
    encoded_vector = binary_encoding.values.argmax(1)
    return encoded_vector


def substrings_in_string(x, substrings):
    for substring in substrings:
        if x.find(substring) != -1:
            return substring


def convert_title(df):
    title_list = ["Mrs", "Mr", "Master", "Miss", "Major", "Rev", "Dr", "Ms", "Mlle", "Col", "Capt",
                  "Mme", "Countess", "Don", "Jonkheer"]

    df["Title"] = df["Name"].map(lambda x: substrings_in_string(x, title_list))

    df["Title"] = df.apply(replace_titles, axis=1)

    df["Title"] = encode_column(df, "Title")

    return df


def convert_deck(df):
    # Fill NaN values with U for unkown
    df["Cabin"] = df["Cabin"].fillna("U")
    df["Deck"] = df["Cabin"].astype(str).str[0]

    df["Deck"] = encode_column(df, "Deck")

    return df


# Replacing all titles with mr, mrs, miss, master
def replace_titles(df):
    title = df["Title"]
    if title in ["Don", "Major", "Capt", "Jonkheer", "Rev", "Col"]:
        return "Mr"
    elif title in ["Countess", "Mme"]:
        return "Mrs"
    elif title in ["Mlle", "Ms"]:
        return "Miss"
    elif title is "Dr":
        if df["Sex"] is "Male":
            return "Mr"
        else:
            return "Mrs"
    else:
        return title
''' ----- END TITANIC ----- '''

def load_data_frame(filename):
    """1. Import data set"""
    dataframe = None
    try:
        dataframe = pd.read_csv(filename)
    except:
        print("No file found for " + filename + ". Exiting now.")
        sys.exit()
    return dataframe


def describe_data_frame(dataframe):
    """
    Aimed at describing characteristics data set for the
    following parts:

    1. How many observations are in the data set?
    2. How many features are in the data set?
    3. What is the data type for each feature and the response?
    4. Print the mean, median, max, and min values for each feature. Suggest using summary()
    """
    print(">> Data Frame Description\n")

    rows, columns = dataframe.shape
    print("num rows (observations): " + str(rows))
    print("num cols (features): " + str(columns))
    print("\n")

    print(str(dataframe.info()))

    print("\nData Set Statistics")
    print("=====================")

    print(str(dataframe.describe()))
    print("\n")


''' ------ BEGIN K-MEANS --------'''
def customKMeans():
    filename = '../data/iris.arff'
    generateKClusters.runKMeans(filename)


def builtInKMeans():
    filename = '../data/iris.arff'
    df, classes, original_dataframe = generateKClusters.parse_arff_file(filename, False)

    X = df.values
    start = time.time()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    end = time.time()

    print("Time: {}".format(end-start))
    score = kmeans.score(X)
    print(score)
    print("Cluster centers: {}".format(kmeans.cluster_centers_))
''' --------- END K-MEANS ---------- '''


def naiveBayes():
    '''
        - What are the dimensions of the data set?
        - What are the response and explanatory variables ? What type are they?
        - How many mushrooms in the data set are edible and poisious ?
        - Should the explanatory variables be scaled ?
        - Split the data set into training and test sets with 75% of data in the training set.
        - Print the dimensions of each set.
        - Develop a model using naiveBayes()
        - Print the conditional probability tables. Choose one and explain the contents.
        - "Predict" the training labels. Print the confusion matrix and accuracy.
        - Predict the test labels. Print the confusion matrix and accuracy.
        - Why is this a good data set for Naive Bayes despite mushrooms not being especially interesting?
    '''
    filename = '../data/agaricus-lepiota.data'
    df = load_data_frame(filename)
    describe_data_frame(df)
    print("todo")

if __name__ == "__main__":
    main()