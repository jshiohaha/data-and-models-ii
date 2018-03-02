import numpy as np
import pandas as pd
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer


def main():
    titanic()
    return


def titanic():
    # Get training data
    filename = "data/titanic_train.csv"
    train = load_data_frame(filename)

    # Split to X and Y
    X_train, Y_train = split_data_frame(train, "Survived")

    X_train_imp = prepare_titanic(X_train)

    # Create random Forest Classifier, fit
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_imp, Y_train)

    # Get testing data
    filename = "data/titanic_test.csv"
    X_test = load_data_frame(filename)

    X_test_imp = prepare_titanic(X_test)

    # Store test predictions as Y
    Y_test = clf.predict(X_test_imp)

    # Convert Pandas Series to Numpy Array
    passenger_id = np.array(X_test["PassengerId"], dtype=pd.Series)

    # Combine Passenger_id with Y_test
    answer = np.column_stack((passenger_id, Y_test))

    # Save output to csv file
    np.savetxt("data/titanic_output.csv", answer, fmt="%.0f", delimiter=",", header="PassengerId,Survived")

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


def split_data_frame(df, label):
    X = df.drop([label], axis=1)
    Y = df[label]
    return X, Y


def prepare_titanic(df):
    # Encode the Sex and Embarked objects
    df["Sex"] = encode_column(df, "Sex")
    df["Embarked"] = encode_column(df, "Embarked")

    # Convert title to useable encoded value
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
    cabin_list = ["A", "B", "C", "D", "E", "F", "T", "G"]

    df["Deck"] = df["Cabin"].map(lambda x: substrings_in_string(x, cabin_list))

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


if __name__ == "__main__":
    main()
