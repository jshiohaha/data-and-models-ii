import numpy as np
import pandas as pd
import sys

from sklearn.ensemble import RandomForestClassifier


def main():
    titanic()
    return


def titanic():
    filename = 'data/titanic_train.csv'
    train = load_data_frame(filename).dropna()

    X_train, Y_train = split_data_frame(train, 'Survived')
    X_train = X_train.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1).astype(np.float32)

    describe_data_frame(X_train)

    clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)

    filename = 'data/titanic_test.csv'
    X_test = load_data_frame(filename).dropna()

    X_test = X_test.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1).astype(np.float32)

    print(clf.predict(X_test))

    return


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


def load_data_frame(filename):
    """1. Import data set"""
    dataframe = None
    try:
        dataframe = pd.read_csv(filename)
    except:
        print("No file found for " + filename + ". Exiting now.")
        sys.exit()
    return dataframe


def split_data_frame(df, label):
    X = df.drop([label], axis=1)
    Y = df[label]
    return X, Y


def encode_column(df, label):
    binary_encoding = pd.get_dummies(df[label])
    encoded_vector = binary_encoding.values.argmax(1)
    return encoded_vector

if __name__ == "__main__":
    main()
