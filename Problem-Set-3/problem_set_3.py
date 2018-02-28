import pandas as pd
import sys

from sklearn import tree


def main():
    titanic()
    return


def titanic():
    filename = 'data/titanic_train.csv'
    train = load_data_frame(filename)

    X_train, Y_train = split_data_frame(train, 'Survived')
    X_train["Embarked"] = clean_column(X_train, ["Embarked"])
    X_train["Sex"] = clean_column(X_train, ["Sex"])
    X_train = X_train.drop(["Cabin", "Ticket", "Name"], axis=1)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)

    filename = 'data/titanic_test.csv'
    X_test = load_data_frame(filename)
    X_test["Embarked"] = clean_column(X_test, ["Embarked"])
    X_test["Sex"] = clean_column(X_test, ["Sex"])
    X_test = X_test.drop(["Cabin", "Ticket", "Name"], axis=1)

    clf.predict(X_test)

    return


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


def clean_column(df, label):
    binary_encoding = pd.get_dummies(df[label])
    encoded_vector = binary_encoding.values.argmax(1)
    return encoded_vector

if __name__ == "__main__":
    main()
