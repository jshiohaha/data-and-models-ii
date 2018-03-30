import sys
import pydot
import numpy as np
import pandas as pd
import math as Math
import matplotlib.pyplot as plt
import operator

import pprint

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.preprocessing import QuantileTransformer, Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score
from sklearn.dummy import DummyRegressor

# dataframe variables...
# Input variables (based on physicochemical tests):
#   1 - fixed acidity
#   2 - volatile acidity
#   3 - citric acid
#   4 - residual sugar
#   5 - chlorides
#   6 - free sulfur dioxide
#   7 - total sulfur dioxide
#   8 - density
#   9 - pH
#   10 - sulphates
#   11 - alcohol
#
#   Output variable (based on sensory data): 
#   12 - quality (score between 0 and 10)

def main():
    '''
        Solution for Problem Set  in Data & Models II (RAIK 370H)

        All of the functions pertaining to the set of questions are below
        and you should just be able to copy and paste the function name
        and universally include the dataframe as a parameter to see the
        results. Each function itself describes what it's doing along
        with the parts of the problem set question it addresses.
    '''

    plot_flag = True

    filename = 'Data/winequality-white.csv'
    df = load_data_frame(filename)

    # plot_feature_histograms(df)
    scaled_df = scale_dataset(df)
    X, Y = create_matrix_and_vector_from_data_frame(scaled_df)
    X_train, X_test, Y_train, Y_test = create_train_test_set(X,Y)
    create_linear_model(X_train, X_test, Y_train, Y_test)
    create_lm_object(X, Y)

def load_data_frame(filename):
    '''
        1. Import the wine quality white data set from the UCI Data Repository
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
    
        TODO: Missing data?
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

    return dataframe


def plot_feature_histograms(dataframe):
    # features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    #             "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",
    #             "quality"]

    # TODO: might need to change this because of all the explanatory variables... it would
    # be simpler to break it out
    plt.figure()
    dataframe.plot.hist(stacked=True)
    plt.show()


def generate_correlation_pairs(dataframe):
    features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

    total = list()
    for i in range(len(features)):
        for j in range(i, len(features)):
            if i == j:
                continue

            pair = (features[i], features[j])

            pair_combo = str(pair[0]+pair[1]).replace(' ', '')
            line1 = "{} = np.correlate(dataframe['{}'], dataframe['{}'])".format(pair_combo,pair[0],pair[1])
            line2 = "print('>> -- Correlation between {} and {}: ' + str({}))".format(pair[0],pair[1],pair_combo)
            print(line1)
            print(line2)
            print("\n")


def find_correlation(dataframe):
    '''
        Find the correlation among the explanatory variables
    '''
    # write a script that that will take these and generate code for correlation...
    # sepalLW = np.correlate(dataframe['Sepal.Length'], dataframe['Sepal.Width'])
    # print(">> -- Correlation between Sepal.Length and Sepal.Width: " + str(sepalLW))
    fixedacidityvolatileacidity = np.correlate(dataframe['fixed acidity'], dataframe['volatile acidity'])
    print('>> -- Correlation between fixed acidity and volatile acidity: ' + str(fixedacidityvolatileacidity))

    fixedaciditycitricacid = np.correlate(dataframe['fixed acidity'], dataframe['citric acid'])
    print('>> -- Correlation between fixed acidity and citric acid: ' + str(fixedaciditycitricacid))

    fixedacidityresidualsugar = np.correlate(dataframe['fixed acidity'], dataframe['residual sugar'])
    print('>> -- Correlation between fixed acidity and residual sugar: ' + str(fixedacidityresidualsugar))

    fixedaciditychlorides = np.correlate(dataframe['fixed acidity'], dataframe['chlorides'])
    print('>> -- Correlation between fixed acidity and chlorides: ' + str(fixedaciditychlorides))

    fixedacidityfreesulfurdioxide = np.correlate(dataframe['fixed acidity'], dataframe['free sulfur dioxide'])
    print('>> -- Correlation between fixed acidity and free sulfur dioxide: ' + str(fixedacidityfreesulfurdioxide))

    fixedaciditytotalsulfurdioxide = np.correlate(dataframe['fixed acidity'], dataframe['total sulfur dioxide'])
    print('>> -- Correlation between fixed acidity and total sulfur dioxide: ' + str(fixedaciditytotalsulfurdioxide))

    fixedaciditydensity = np.correlate(dataframe['fixed acidity'], dataframe['density'])
    print('>> -- Correlation between fixed acidity and density: ' + str(fixedaciditydensity))

    fixedaciditypH = np.correlate(dataframe['fixed acidity'], dataframe['pH'])
    print('>> -- Correlation between fixed acidity and pH: ' + str(fixedaciditypH))

    fixedaciditysulphates = np.correlate(dataframe['fixed acidity'], dataframe['sulphates'])
    print('>> -- Correlation between fixed acidity and sulphates: ' + str(fixedaciditysulphates))

    fixedacidityalcohol = np.correlate(dataframe['fixed acidity'], dataframe['alcohol'])
    print('>> -- Correlation between fixed acidity and alcohol: ' + str(fixedacidityalcohol))

    volatileaciditycitricacid = np.correlate(dataframe['volatile acidity'], dataframe['citric acid'])
    print('>> -- Correlation between volatile acidity and citric acid: ' + str(volatileaciditycitricacid))

    volatileacidityresidualsugar = np.correlate(dataframe['volatile acidity'], dataframe['residual sugar'])
    print('>> -- Correlation between volatile acidity and residual sugar: ' + str(volatileacidityresidualsugar))

    volatileaciditychlorides = np.correlate(dataframe['volatile acidity'], dataframe['chlorides'])
    print('>> -- Correlation between volatile acidity and chlorides: ' + str(volatileaciditychlorides))

    volatileacidityfreesulfurdioxide = np.correlate(dataframe['volatile acidity'], dataframe['free sulfur dioxide'])
    print('>> -- Correlation between volatile acidity and free sulfur dioxide: ' + str(volatileacidityfreesulfurdioxide))

    volatileaciditytotalsulfurdioxide = np.correlate(dataframe['volatile acidity'], dataframe['total sulfur dioxide'])
    print('>> -- Correlation between volatile acidity and total sulfur dioxide: ' + str(volatileaciditytotalsulfurdioxide))

    volatileaciditydensity = np.correlate(dataframe['volatile acidity'], dataframe['density'])
    print('>> -- Correlation between volatile acidity and density: ' + str(volatileaciditydensity))

    volatileaciditypH = np.correlate(dataframe['volatile acidity'], dataframe['pH'])
    print('>> -- Correlation between volatile acidity and pH: ' + str(volatileaciditypH))

    volatileaciditysulphates = np.correlate(dataframe['volatile acidity'], dataframe['sulphates'])
    print('>> -- Correlation between volatile acidity and sulphates: ' + str(volatileaciditysulphates))

    volatileacidityalcohol = np.correlate(dataframe['volatile acidity'], dataframe['alcohol'])
    print('>> -- Correlation between volatile acidity and alcohol: ' + str(volatileacidityalcohol))

    citricacidresidualsugar = np.correlate(dataframe['citric acid'], dataframe['residual sugar'])
    print('>> -- Correlation between citric acid and residual sugar: ' + str(citricacidresidualsugar))

    citricacidchlorides = np.correlate(dataframe['citric acid'], dataframe['chlorides'])
    print('>> -- Correlation between citric acid and chlorides: ' + str(citricacidchlorides))

    citricacidfreesulfurdioxide = np.correlate(dataframe['citric acid'], dataframe['free sulfur dioxide'])
    print('>> -- Correlation between citric acid and free sulfur dioxide: ' + str(citricacidfreesulfurdioxide))

    citricacidtotalsulfurdioxide = np.correlate(dataframe['citric acid'], dataframe['total sulfur dioxide'])
    print('>> -- Correlation between citric acid and total sulfur dioxide: ' + str(citricacidtotalsulfurdioxide))

    citricaciddensity = np.correlate(dataframe['citric acid'], dataframe['density'])
    print('>> -- Correlation between citric acid and density: ' + str(citricaciddensity))

    citricacidpH = np.correlate(dataframe['citric acid'], dataframe['pH'])
    print('>> -- Correlation between citric acid and pH: ' + str(citricacidpH))

    citricacidsulphates = np.correlate(dataframe['citric acid'], dataframe['sulphates'])
    print('>> -- Correlation between citric acid and sulphates: ' + str(citricacidsulphates))

    citricacidalcohol = np.correlate(dataframe['citric acid'], dataframe['alcohol'])
    print('>> -- Correlation between citric acid and alcohol: ' + str(citricacidalcohol))

    residualsugarchlorides = np.correlate(dataframe['residual sugar'], dataframe['chlorides'])
    print('>> -- Correlation between residual sugar and chlorides: ' + str(residualsugarchlorides))

    residualsugarfreesulfurdioxide = np.correlate(dataframe['residual sugar'], dataframe['free sulfur dioxide'])
    print('>> -- Correlation between residual sugar and free sulfur dioxide: ' + str(residualsugarfreesulfurdioxide))

    residualsugartotalsulfurdioxide = np.correlate(dataframe['residual sugar'], dataframe['total sulfur dioxide'])
    print('>> -- Correlation between residual sugar and total sulfur dioxide: ' + str(residualsugartotalsulfurdioxide))

    residualsugardensity = np.correlate(dataframe['residual sugar'], dataframe['density'])
    print('>> -- Correlation between residual sugar and density: ' + str(residualsugardensity))

    residualsugarpH = np.correlate(dataframe['residual sugar'], dataframe['pH'])
    print('>> -- Correlation between residual sugar and pH: ' + str(residualsugarpH))

    residualsugarsulphates = np.correlate(dataframe['residual sugar'], dataframe['sulphates'])
    print('>> -- Correlation between residual sugar and sulphates: ' + str(residualsugarsulphates))

    residualsugaralcohol = np.correlate(dataframe['residual sugar'], dataframe['alcohol'])
    print('>> -- Correlation between residual sugar and alcohol: ' + str(residualsugaralcohol))

    chloridesfreesulfurdioxide = np.correlate(dataframe['chlorides'], dataframe['free sulfur dioxide'])
    print('>> -- Correlation between chlorides and free sulfur dioxide: ' + str(chloridesfreesulfurdioxide))

    chloridestotalsulfurdioxide = np.correlate(dataframe['chlorides'], dataframe['total sulfur dioxide'])
    print('>> -- Correlation between chlorides and total sulfur dioxide: ' + str(chloridestotalsulfurdioxide))

    chloridesdensity = np.correlate(dataframe['chlorides'], dataframe['density'])
    print('>> -- Correlation between chlorides and density: ' + str(chloridesdensity))

    chloridespH = np.correlate(dataframe['chlorides'], dataframe['pH'])
    print('>> -- Correlation between chlorides and pH: ' + str(chloridespH))

    chloridessulphates = np.correlate(dataframe['chlorides'], dataframe['sulphates'])
    print('>> -- Correlation between chlorides and sulphates: ' + str(chloridessulphates))

    chloridesalcohol = np.correlate(dataframe['chlorides'], dataframe['alcohol'])
    print('>> -- Correlation between chlorides and alcohol: ' + str(chloridesalcohol))

    freesulfurdioxidetotalsulfurdioxide = np.correlate(dataframe['free sulfur dioxide'], dataframe['total sulfur dioxide'])
    print('>> -- Correlation between free sulfur dioxide and total sulfur dioxide: ' + str(freesulfurdioxidetotalsulfurdioxide))

    freesulfurdioxidedensity = np.correlate(dataframe['free sulfur dioxide'], dataframe['density'])
    print('>> -- Correlation between free sulfur dioxide and density: ' + str(freesulfurdioxidedensity))

    freesulfurdioxidepH = np.correlate(dataframe['free sulfur dioxide'], dataframe['pH'])
    print('>> -- Correlation between free sulfur dioxide and pH: ' + str(freesulfurdioxidepH))

    freesulfurdioxidesulphates = np.correlate(dataframe['free sulfur dioxide'], dataframe['sulphates'])
    print('>> -- Correlation between free sulfur dioxide and sulphates: ' + str(freesulfurdioxidesulphates))

    freesulfurdioxidealcohol = np.correlate(dataframe['free sulfur dioxide'], dataframe['alcohol'])
    print('>> -- Correlation between free sulfur dioxide and alcohol: ' + str(freesulfurdioxidealcohol))

    totalsulfurdioxidedensity = np.correlate(dataframe['total sulfur dioxide'], dataframe['density'])
    print('>> -- Correlation between total sulfur dioxide and density: ' + str(totalsulfurdioxidedensity))

    totalsulfurdioxidepH = np.correlate(dataframe['total sulfur dioxide'], dataframe['pH'])
    print('>> -- Correlation between total sulfur dioxide and pH: ' + str(totalsulfurdioxidepH))

    totalsulfurdioxidesulphates = np.correlate(dataframe['total sulfur dioxide'], dataframe['sulphates'])
    print('>> -- Correlation between total sulfur dioxide and sulphates: ' + str(totalsulfurdioxidesulphates))

    totalsulfurdioxidealcohol = np.correlate(dataframe['total sulfur dioxide'], dataframe['alcohol'])
    print('>> -- Correlation between total sulfur dioxide and alcohol: ' + str(totalsulfurdioxidealcohol))

    densitypH = np.correlate(dataframe['density'], dataframe['pH'])
    print('>> -- Correlation between density and pH: ' + str(densitypH))

    densitysulphates = np.correlate(dataframe['density'], dataframe['sulphates'])
    print('>> -- Correlation between density and sulphates: ' + str(densitysulphates))

    densityalcohol = np.correlate(dataframe['density'], dataframe['alcohol'])
    print('>> -- Correlation between density and alcohol: ' + str(densityalcohol))

    pHsulphates = np.correlate(dataframe['pH'], dataframe['sulphates'])
    print('>> -- Correlation between pH and sulphates: ' + str(pHsulphates))

    pHalcohol = np.correlate(dataframe['pH'], dataframe['alcohol'])
    print('>> -- Correlation between pH and alcohol: ' + str(pHalcohol))

    sulphatesalcohol = np.correlate(dataframe['sulphates'], dataframe['alcohol'])
    print('>> -- Correlation between sulphates and alcohol: ' + str(sulphatesalcohol))


def scale_dataset(dataframe):
    print(">> Normalizing the data set\n")
    # http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    # looking at the histograms to choose between normal and uniform scaling

    # normal scaling
    scaled_data_array = Normalizer().fit_transform(dataframe)
    df_norm = pd.DataFrame(scaled_data_array, index=dataframe.index, columns=dataframe.columns)

    # describe_data_frame(df_norm)
    # plot_feature_histograms(df_norm)

    # uniform scaling
    # scaled_data_array = QuantileTransformer(output_distribution='uniform').fit_transform(dataframe)
    # df_uni = pd.DataFrame(scaled_data_array, index=dataframe.index, columns=dataframe.columns)

    # describe_data_frame(df_uni)
    # plot_feature_histograms(df_uni)

    return df_norm


def create_matrix_and_vector_from_data_frame(dataframe):
    print(">> Creating X matrix and Y vector from scaled data set\n")

    r, c = dataframe.shape
    Y = dataframe['quality'].as_matrix()
    X = dataframe.as_matrix(columns=dataframe.columns[:c-1])

    return X, Y


def create_train_test_set(X, Y):
    '''
        Create a test set using 130 observations and a test set with the other 20 observations.
        Confirm by displaying the dimensions of each set.
    '''
    print(">> Creating testing and training datasets\n")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.25, random_state=42)

    print('Length of X (and Y) training set: ' + str(len(X_train)))
    print('Length of X (and Y) testing set: ' + str(len(X_test)))

    return X_train, X_test, Y_train, Y_test


def create_linear_model(X_train, X_test, Y_train, Y_test):
    fitModel = DummyRegressor(strategy='mean').fit(X_train, Y_train)
    baseline_score = fitModel.score(X_test, Y_test)
    print("Baseline Score (R^2): {}".format(baseline_score))


def create_lm_object(X, Y):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    print(regr.coef_)
    return

'''
    TODO'S...

    Question 3\
        - Develop an lm() object using all of the explanatory variables
        - Print the model information using summary()
        - Print the model information criterion using AIC(), extractAIC(), and logLik()
        - Predict the wine quality using the test set and compare the accuracy to the actual quality. Comment.
        - Print the parameter estimates and their 95% confidence intervals in a single table. (Suggest using
          confint()), and cbind()

    Question 4
        - Roll your own code to compute model parameters, as well as the model information from the library solver
        - First create the matrix and the vector for the training data (Remember to insert a column of 1’s in
          the matrix)
        - Compute the parameter values (the coefficent estimates in the lm() object )
        - Print the parameters from the lm() model and from your normal solver side by side. Comment. (suggest
          using head())
        - Print the test set quality from the lm() model and from your normal solver side by side. Comment.
          (suggest using head())
        - Print the rmse error between the predicted and actual test qualities

    Question 5
        - Now compute the parameters using the gradient descent solver using the same and
        - First, write a function to compute the scalar value of the cost function
        - Clearly display your learning rate, and your convergence criterion
        - Print the estimated parameters from the lm() model, your normal solver, and your gradient descent
          solver side by side. Comment.
        - Predict the wine quality using the gradient descent parameter using the test set and compare to the
          actual quality in the test set

    Question 6
        - Compare accuracies on the test set to those of a neural net model. Comment.
        - Describe your final neural net model.

    Question 7
        - Now re-compute all of the information from your lm() model using your normal equation model
            - Compute error residuals, e, and plot the histogram of residuals
            - Print the summary() of the error vector, , and compare to lm() model output. Comment
            - Plot histogram of residual errors to check approximate normality. If the errors were not nearly normal
              what might be the problem?
            - Most residual errors are less than |1|, what does that mean ?
            - Compute the residual standard error and the degrees freedom for the residual error
    
    Question 8
        - Create and print a table similar to that in lm() output for your theta values for estimates, , compute
          standard error, T values, and P values.
    
    Question 9
        - Compute R^2
        - Compute R^2_{ADJ}
        - Compute AIC
        - Compute the F statistic for the model
        - Compute degrees of freedom 1 and 2 for the f diistribution
        - Compute the P value for the F, overall model, statistic

    Question 10
        - Reduce the number of explanatory variables in your lm() model one by one to find the best model using
          the AIC criterion (tradeoff between maximum likelihood and number of parameters). (suggest using
          step(lm(),…))
        - Increase the number explanatory variables from the intercept alone in your lm() model one by one to
          find the best model using the AIC criterion
        - Note that step(lm()) uses extractAIC() not AIC()
'''

if __name__ == "__main__":
    main()
