import pandas as panda
import matplotlib.pyplot as plot
import numpy as numpy
import math as math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report


# TODO: Question 2, 7, 8
# Q: What counts as NA in the data set -- a 0 in the ratio column?

# 7. Why was this simple classification model acceptable for this data set ? Discuss.
def main():
    '''
        Solution for Problem Set 1 in Data & Models II (RAIK 370H)

        All of the functions pertaining to the set of questions are below
        and you should just be able to copy and paste the function name
        and universally include the dataframe as a parameter to see the
        results. Each function itself describes what it's doing along
        with the parts of the problem set question it addresses.
    '''
    plot_flag = False

    problem_set_data = panda.read_csv("ProbSet1.csv")
    # visualize_data_3d(problem_set_data, plot_flag)
    # df = problem_set_data.copy(deep=True)
    # visualize_data_3d(problem_set_data, plot_flag)
    normalized_data = normalize_and_visualize_data(problem_set_data, plot_flag)
    X, Y = create_matrix_and_vector_from_data_frame(normalized_data)
    built_in_model(X,Y)
    # weights = build_model(X, Y)
    # test_and_validate_model(X, Y, weights)


def describe_data_frame(df):
    ''' Aimed at describing characteristics of the problem set 1 data set for the
        following parts:

        1. How many observations are in the data set?
        2. How many features are in the data set?
        3. What is the data type for each feature and the response?
        4. Print the mean, median, max, and min values for each feature. Suggest using summary()
    '''
    print(">> Data Frame Description\n")

    rows, columns = df.shape
    print("num rows (observations): " + str(rows))
    print("num cols (features): " + str(columns))
    print("\n")

    print(df.info())

    print("\nData Set Statistics")
    print("=====================")

    print(df.describe())
    print("\n")


def check_nan(df):
    print(">> Checking for NaN Values in Data Set\n")

    temp = df.copy(deep=True)
    # del temp['loan']
    # temp['col_name'].replace(0, numpy.nan);
    # print(temp['ratio'].isnull().sum())
    temp.replace(0, numpy.nan)
    print(temp)


def visualize_data_3d(df, plot_flag=False):
    ''' Aimed at visualizing the data in 3D space and then analyzing
        correlations between variables.

        1. Make a 3D scatterplot of the data. Identify observations as good or bad loans using color. Comment.
        2. Output the correlations between the explanatory variables (the features.) Comment.
    '''
    if not plot_flag:
        return

    print(">> Constructing 3D plot of data set values\n")

    # plot a 2d plane through a 3d graph: https://stackoverflow.com/questions/47835726/plotting-a-2d-plane-through-a-3d-surface
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')

    good_loans = df.loc[df['loan'] == 1]
    good_x = good_loans['fico']
    good_y = good_loans['income']
    good_z = good_loans['ratio']

    ax.scatter(good_x,good_y,good_z, color='green', label='Good Loans')

    bad_loans = df.loc[df['loan'] == 0]
    bad_x = bad_loans['fico']
    bad_y = bad_loans['income']
    bad_z = bad_loans['ratio']
    ax.scatter(bad_x,bad_y,bad_z, color='red', label='Bad Loans')

    ax.set_xlabel('Fico Score')
    ax.set_ylabel('Income')
    ax.set_zlabel('Ratio')

    point  = numpy.array([1, 2, 3])
    normal = numpy.array([1, 1, 2])

    # TODO: how do we plot a plane based on the weights
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = numpy.meshgrid(range(10), range(10))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    surface = fig.gca(projection='3d')
    surface.plot_surface(xx, yy, z)

    ax.legend()
    plot.show()


def normalize_and_visualize_data(df, plot_flag):
    ''' Aimed at scaling each feature to lie in the range from 0 to 1
        and then making a 3d scatter plot of the scaled data.

        1. Print the mean, median, max, and min for the scaled data
        2. Make a 3D scatterplot of the scaled data
    '''
    print(">> Normalizing the data set between 0 and 1\n")

    scaler = MinMaxScaler()
    scaled_data_array = scaler.fit_transform(df)
    df_norm = panda.DataFrame(scaled_data_array, index=df.index, columns=df.columns)
    describe_data_frame(df_norm)
    visualize_data_3d(df, plot_flag)

    return df_norm


def create_matrix_and_vector_from_data_frame(df):
    print(">> Creating X matrix and Y vector from normalized data set\n")

    rows, columns = df.shape
    one_col = numpy.ones((rows,), dtype=int)
    df.insert(0, 'x_0', one_col)

    Y = df['loan'].as_matrix()
    X = df.as_matrix(columns=df.columns[:4])

    return X, Y


def build_model(X, Y, eta=0.01, epochs=100, error_threshold=.001, verbose=False):
    ''' Build a model based on the data frame from the input csv. Perform stochastic gradient descent,
        updating the weights during each iteration so that the root mean squared error of the output
        of the function versus expected output converges.

        1. Create the X matrix
        2. Create the y vector
        3. Solve for and print the weights (parameters). Label the weights.
        4. Print the number of iterations (forward / backward sweeps or epochs)
        TODO...
        5. Plot the data again and include the descision boundary (plane.) Suggest using scatterplot3d() with plane3d().
    '''
    print(">> Building a custom model from matrix X and vector Y\n")

    print("Learning rate (eta): " + str(eta))
    print("Number of epochs: " + str(epochs))
    print("Error threshold: " + str(error_threshold))

    rows, columns = X.shape
    weights = numpy.reshape(numpy.zeros(columns), (columns, 1))
    
    if verbose:
        print("Weights initialized to " + str(weights))
        print("\n")

    y_hat = numpy.reshape(numpy.zeros(rows), (rows, 1))

    for iteration in range(epochs):
        delta_weights = numpy.reshape(numpy.zeros(columns), (4, 1))

        if verbose:
            print("- Iteration " + str(iteration+1))
        
        for i, x in enumerate(X):
            y_hat[i] = simple_activation_function(numpy.dot(X[i],weights))
            error = Y[i] - y_hat[i]
            delta_weights = numpy.add(delta_weights, numpy.reshape(eta*error*X[i], (4, 1)))

        weights = numpy.add(weights, delta_weights)
        mse = mean_squared_error(Y, y_hat)

        if verbose:
            print(">> Mean squared error: " + str(mse))
            print("\n")

        if mse < error_threshold:
            print("\n>> Converged after " + str(iteration) + " iterations.")
            break

    if verbose:
        print("\nFinal weights: " + str(weights) + "\n")

    return weights


def simple_activation_function(x):
    if x < 0:
        return int(0)
    return int(1)


def test_and_validate_model(X, Y, weights):
    ''' The concept of training and testing data sets and accuracy is not really relevant
        since the data is linearly separable and a test set has not been provided. However,
        we still want to validate that our model can reasonably predict responses. This is
        essentially a predict() method based on our weights from the stochastic gradient descent
        sweep on build_model().

        1. Compare predictions for all observations to actual responses
        2. Output and explain the confusion matrix.
    '''
    print(">> Validating the custom model on the data set\n")

    rows, columns = X.shape

    Y_hat = numpy.reshape(numpy.zeros(rows), (rows, 1))
    num_correct = 0
    was_right = False

    for i, x in enumerate(X):
        Y_hat[i] = simple_activation_function(numpy.dot(X[i], weights))
        
        if Y_hat[i] == Y[i]:
            num_correct += 1
            was_right = True
        else:
            was_right = False

        print("-- Test " + str(i+1) + ": " + str("Matched" if was_right else "Non-Match"))

        if not was_right:
            print("\tGuess: " + str(Y_hat[i]))
            print("\tActual: " + str(Y[i]))

    print("\n")
    accuracy = (num_correct/rows) * 100
    print("The model was " + str(accuracy) + "% correct.")

    print("Here is the confusion matrix generated by this model:")
    print(confusion_matrix(Y, Y_hat))
    print("\n")

    acceptable_accuracy(accuracy)
    return


def acceptable_accuracy(accuracy, accuracy_threshold=.95):
    ''' Checking if model validation yielded an acceptable accuracy.
    
        1. Why was this simple classification model acceptable for this
        data set? Discuss.
    '''
    if accuracy >= accuracy_threshold:
        print(">> Accuracy of the model (" + str(accuracy) + ") is above the threshold of " + str(accuracy_threshold) + ".\n")
        return True
    print(">> Accuracy of the model (" + str(accuracy) + ") is not above the threshold of " + str(accuracy_threshold) + ".\n")
    return False


def built_in_model(X, Y):
    ''' In build_model(), we built a custom model and essentially implemented neural_net() for a single node
        since we are just using a single perceptron. Here, we want to use a built-in nerual_net() funciton to
        see how the model compares to our custom one. We will not use any hidden nodes.

        1. Display the model. Suggest using plot( … , rep=“best”)
        2. Compare model predictions with and without the activation function to the actual results. 
           Suggest writing a sum of squared error loop as well as printing some comparisons.
        3. Now, develop the model with 1 hidden layer with 1 node. Display the model and test as in part b above. Comment.
        4. Now, develop the model with 1 hidden layer with 2 nodes. Display the model and test as in part b above. Comment.
    '''
    print(">> Training a Keras Sequential model\n")\

    X = X[:,[1,2,3]]
    rows, columns = X.shape

    X = numpy.reshape(X, (columns, rows))
    model = Sequential()
    model.add(Dense(32, input_shape=(columns,rows)))
    # model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, Y, epochs=50, batch_size=20)
    score = model.evaluate(X, Y, batch_size=20)

    print(score)


if __name__ == "__main__":
    main()