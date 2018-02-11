import sys
import pydot
import numpy as np
import pandas as pd
import math as math
import matplotlib.pyplot as plot

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score

# LINKS TO HELPFUL INFORMATION ON KNN...
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

def main():
    '''
        Solution for Problem Set 2 in Data & Models II (RAIK 370H)

        All of the functions pertaining to the set of questions are below
        and you should just be able to copy and paste the function name
        and universally include the dataframe as a parameter to see the
        results. Each function itself describes what it's doing along
        with the parts of the problem set question it addresses.
    '''
    plot_flag = True

    filename = 'Data/iris.csv'
    dataframe = load_data_frame(filename)
    # describe_data_frame(dataframe)

    # features = ['Petal.Length', 'Petal.Width', 'Sepal.Length']
    # visualize_data_3d(dataframe, features, plot_flag=plot_flag)

    df_norm = randomize_and_scale_dataset(dataframe)
    # visualize_data_3d(df_norm, features, plot_flag=plot_flag)

    X,Y = create_matrix_and_vector_from_data_frame(df_norm)

    X_train, X_test, y_train, y_test = create_train_test_set(X, Y)
    k = 5
    prediction = knn_model(X_train, y_train, X_test, k)
    confusion_matrix(prediction, y_test)
    use_knn(X_train, y_train, X_test, y_test)

def load_data_frame(filename):
    '''
        1. Import the iris data set from the UCI Data Repository
    '''
    dataframe = None
    try:
        dataframe = pd.read_csv(filename)
    except:
        print("No file found for " + filename + ". Exiting now.")
        sys.exit()
    return dataframe


def describe_data_frame(dataframe):
    '''
        2. Display the data frame dimensions, the structure, summary, the first 5
           and last 5 observations. Which are the explanatory and response variables?
           Comment on the data.
    '''
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

    print(dataframe.corr())

    print("\nFirst 5 obversvations from the dataset")
    print("========================================")
    print(dataframe[:5])

    print("\nLast 5 obversvations from the dataset")
    print("=======================================")
    print(dataframe[-5:])

    return dataframe


def visualize_data_3d(dataframe, features, weights=None, plot_flag=False):
    ''' Aimed at visualizing the data in 3D space and then analyzing
        correlations between variables.

        1. Display several 3D scatterplots using 3 different explanatory variables
        in each plot and different viewing angles. Color code the three iris species
        in the scatter plots. Comment.
    '''
    if not plot_flag:
        return

    print(">> Constructing 3D plot of data set values\n")

    if len(features) != 3:
        print(">> FAILED. Expecting exactly 3 features in the feature array.")

    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    species_arr = ['setosa', 'versicolor', 'virginica']
    colors = ['green', 'red', 'blue']

    for i, f in enumerate(features):
        temp_df = dataframe.loc[dataframe['Species'] == species_arr[i]]
        temp_x = temp_df[features[0]]
        temp_y = temp_df[features[1]]
        temp_z = temp_df[features[2]]
        ax.scatter(temp_x,temp_y,temp_z, color=colors[i], label= species_arr[i] + ' species')


    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])

    ax.legend()
    plot.show()



def find_and_plot_correlation(dataframe):
    '''
        Graphically display the correlation among the features (explanatory variables.)
        Comment.

        TODO...
    '''
    return


def randomize_and_scale_dataset(dataframe):
    '''
        1. Since the observations are grouped by species, randomize the observations for subsequent
        use. Suggest using order().
        2. Scale each feature (column of X) so that each feature observation lies between 0 and 1.
        Verify the scaling. Suggest using summary() or str()
    '''
    print(">> Normalizing the data set between 0 and 1\n")

    explanatory_variables_df = dataframe[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']]
    scaler = MinMaxScaler()
    scaled_data_array = scaler.fit_transform(explanatory_variables_df)
    df_norm = pd.DataFrame(scaled_data_array, index=explanatory_variables_df.index, columns=explanatory_variables_df.columns)
    
    describe_data_frame(df_norm)

    print(">> Randomizing observations for subsequent use\n")
    df_norm = df_norm.sample(frac=1)
    df_norm['Species'] = dataframe['Species']

    return df_norm


def create_matrix_and_vector_from_data_frame(dataframe):
    print(">> Creating X matrix and Y vector from scaled data set\n")

    rows, columns = dataframe.shape
    Y = dataframe['Species'].as_matrix()
    X = dataframe.as_matrix(columns=dataframe.columns[:4])

    return X, Y


def create_train_test_set(X, Y, training_set_size=130, testing_set_size=20):
    '''
        Create a test set using 130 observations and a test set with the other 20 observations.
        Confirm by displaying the dimensions of each set.
    '''
    print(">> Creating testing and training datasets\n")
    test_size = testing_set_size / (training_set_size + testing_set_size)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    print('Length of X training set: ' + str(len(X_train)))
    print('Length of Y training set: ' + str(len(y_train)))

    print('Length of X testing set: ' + str(len(X_test)))
    print('Length of Y testing set: ' + str(len(y_test)) + '\n')

    return X_train, X_test, y_train, y_test


def generate_integer_encoding(vector):
    binary_encoding = pd.get_dummies(vector)
    encoded_vector = binary_encoding.values.argmax(1)
    return encoded_vector


def minkowski_distance(vector0, vector1, p):
    distance = 0
    for idx in range(vector0):
        distance += Math.pow(vector0[idx] - vector1[idx], p)
    return Math.pow(distance, 1/p)


def get_n_neighbors(vector, point, k):
    '''
        For any point, in this case, any row in the representing the explanatory
        variables of the iris dataset, it will take another vector, representing
        a list of sets of explantory variables. It will calculate the closest
        k neighbors according to the L_2 distance, also known as Euclidean distance.
    '''
    distances = []
    for idx in range(len(vector)):
        dist = minkowski_distance(vector[idx], point, 2)
        distances.append((idx, dist))

    # sort array by the distance attribute, at second index
    distances.sort(key=operator.itemgetter(1))
    return distances[:k]


# not sure if this works or not... mainly the max() lambda to get the mode of the list
def get_class_from_n_neighbors(neighbors, vector):
    species = []
    for idx in range(len(neighbors)):
        # append the instance of vector_y to the corresponding
        # instance from vector_x... should be the species related
        # to that specific neighbor
        species.append(vector[neighbors[idx][0]])
    common_species = max(species, key=species.count)
    return common_species


def knn_model(xtrain, ytrain, xtest, k):
    '''
        1. Write code from scratch to predict the species of each observation in the test set using
        KNN. Experiment with the prediction accuracy by changing K, the number of neighbors.
        You might include mention of bias and variance.

        2. Display your comparisons as well as accuracy and error rates. Include a confusion
        matrix as well as a line plot for acccuracy rate (vertical axis) v. K (horizonatal axis). Suggest that table( … , …) be included.
        (Accuracy is determined by comparing your species predictions to the known species.)

        TODO... Question 6

        Q: What is serialized from the model to use in the training set?
    '''
    prediction = []
    for idx in range(len(xtest)):
        neighbors = get_n_neighbors(xtrain, xtest[idx], k)
        prediction.append(get_class_from_n_neighbors(neighbors, ytrain))
    return prediction


def knn_calculate_accuracy(xtrain, ytrain, xtest, ytest, k=10):
    accuracy = []
    for i in range(k):
        prediction = knn_model(xtrain, ytrain, xtest, k)
        print("Confusion matrix with K = {}".format(k))
        print(confusion_matrix(prediction, ytest))
        accuracy.append(accuracy_score(prediction, ytest))

    # TODO: Plot accuracy


def use_knn(X_train, Y_train, X_test, Y_test, n_neighbors=5):
    '''
        Repeat the process from Question 6, but use the knn() function from the class package.
    '''
    print(">> Using KNeighborsRegressor from sklearn.neighbors to calculate " + str(n_neighbors) + " nearest neighbors\n")

    encoded_y_train = generate_integer_encoding(Y_train)
    encoded_y_test = generate_integer_encoding(Y_test)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, encoded_y_train)
    predictions = knn.predict(X_test)

    # Compute the mean squared error of our predictions.
    mse = mean_squared_error(predictions, encoded_y_test)
    print("Mean squared error: " + str(mse))


'''
>> OUTSTANDING PROBLEMS TO CREATE FUNCTIONS FOR + DO...

    Question 8
        - Develop an ANN model for iris species prediction using a library call to a neural network modeler. Suggest
          using neuralnet( ).
        - Since the iris prediction has 3 response categories instead of 2 (its multinomial, not binomial), the species
          factor variable must be split into 3 binary variables. The method is called one hot encoding.
        - Suggest using class.ind( ) from the nnet package for one hot encoding, but this is also easily hand coded.
        - Make a formula using as.formula( ) for use with neuralnet( ) of the form: y1 + y2 + y3 ~ x1 + x2 + x3 + x4
        - Plot the model network
    
    Question 9
        - Predict the species for each observation in the test set. Display the comparisons between actual and
          predicted. Use the numeric, not binary, version of predicted results. Choose a meaningful display type.
        - Display the RMSE error between actual and predicted for each category.
        - Display the actual (numeric) and the activated (0 or 1) predicted values for categories. Suggest using ifelse()
          with a threshold of .5 which is ‘half way’ between 0 and 1.
        - Display the confusion table for each species prediction. Suggest using table(actual = y1test, predicted = …)
        - Display accuracy and error rates from the confusion tables. Suggest using table(), diag(), and sum(). Note that
          table( ) values are stored columnwise.
    
    Question 10
        - Using your ANN model, experiment with the number of hidden nodes. What structure of layers and nodes
          produces the best accuracy on the test set relative to the number of layers, nodes, and model solution time?
        - Carry out this experiment any way you like. Summarize your observations.
'''


if __name__ == "__main__":
    main()