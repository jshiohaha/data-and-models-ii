import sys
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

import generateKClusters
import entropy

'''
    TODO:
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
    # customEntropy()
    # customKMeans()
    # builtInKMeans()
    # naiveBayes()
    return


''' ----- BEGIN ENTROPY ----- '''

def customEntropy():
    # Get training data
    filename = "../data/romanian.csv"
    romanian = load_data_frame(filename)
    entropy.runEntropy(romanian)

''' ------ END ENTROPY ------ '''


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

    X_test_imp["Deck_T"] = 0
    X_test_imp["Title_Royalty"] = 0

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
    df = dummy_column(df, "Sex")
    df = dummy_column(df, "Embarked")
    df = dummy_column(df, "Pclass")
    df = dummy_column(df, "Title")
    df["Fare"] = df["Fare"].fillna(df.Fare.mean())

    # Convert objects to useable encoded value
    df = convert_title(df)

    _group = df.groupby(["Sex", "Pclass", "Title"])
    group_median = _group.median()

    df["Age"] = df.apply(lambda x: fill_ages(x, group_median) if np.isnan(x["Age"]) else x["Age"], axis=1)

    df = convert_deck(df)

    df = process_family(df)

    # Drop the objects that can't be encoded, convert floats to float32
    df = df.drop(["Ticket", "Embarked", "Sex", "Pclass", "Cabin", "PassengerId", "Name", "Deck", "Title", "FamilySize"], axis=1).astype(np.float32)

    return df


def dummy_column(df, label):
    dummies = pd.get_dummies(df[label], prefix=label)
    df = pd.concat([df, dummies], axis=1)

    return df


def convert_title(df):
    df["Title"] = df["Name"].map(lambda x: x.split(",")[1].split(".")[0].strip())

    title_list = {
        "Mrs": "Mrs",
        "Mr": "Mr",
        "Master": "Master",
        "Miss": "Miss",
        "Major": "Officer",
        "Rev": "Officer",
        "Dr": "Officer",
        "Ms": "Mrs",
        "Mlle": "Miss",
        "Col": "Officer",
        "Capt": "Officer",
        "Mme": "Mrs",
        "Countess": "Royalty",
        "Don": "Royalty",
        "Jonkheer": "Royalty"
    }

    df["Title"] = df["Title"].map(title_list)

    return df


def fill_ages(row, grouped_median):
    if row['Sex'] == 'female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 1, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 1, 'Mrs']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['female', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['female', 1, 'Royalty']['Age']

    elif row['Sex'] == 'female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 2, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 2, 'Mrs']['Age']

    elif row['Sex'] == 'female' and row['Pclass'] == 3:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 3, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 3, 'Mrs']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 1, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 1, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['male', 1, 'Royalty']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 2, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 2, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 2, 'Officer']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 3, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 3, 'Mr']['Age']


def convert_deck(df):
    # Fill NaN values with U for unkown
    df["Cabin"] = df["Cabin"].fillna("U")
    df["Deck"] = df["Cabin"].astype(str).str[0]

    df = dummy_column(df, "Deck")

    return df


def process_family(df):
    df["FamilySize"] = df["Parch"] + df["SibSp"]

    df["Alone"] = df.FamilySize.map(lambda x: 1 if x == 0 else 0)
    df["Small"] = df.FamilySize.map(lambda x: 1 if 0 < x < 4 else 0)
    df["Large"] = df.FamilySize.map(lambda x: 1 if 4 < x else 0)

    return df


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

    print("Time: {}".format(end - start))
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
