import sys
import pydot
import numpy as np
import pandas as pd
import math as  math
import matplotlib.pyplot as plt
import operator
import itertools

import pprint

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn import linear_model
from sklearn.preprocessing import QuantileTransformer, Normalizer, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, r2_score
from sklearn.dummy import DummyClassifier
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

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

'''
    [TODO] Question 10
        - Reduce the number of explanatory variables in your lm() model one by one to find the best model using
          the AIC criterion (tradeoff between maximum likelihood and number of parameters). (suggest using
          step(lm(),…))
        - Increase the number explanatory variables from the intercept alone in your lm() model one by one to
          find the best model using the AIC criterion
        - Note that step(lm()) uses extractAIC() not AIC()
'''

def main():
    '''
        Solution for Problem Set 4 in Data & Models II (RAIK 370H)

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
    X, Y = create_matrix_and_vector_from_data_frame(df)
    X_train, X_test, Y_train, Y_test = create_train_test_set(X, Y)

    scaled_df = scale_dataset(df)
    scaled_X, scaled_Y = create_matrix_and_vector_from_data_frame(scaled_df)
    scaled_X_train, scaled_X_test, scaled_Y_train, scaled_Y_test = create_train_test_set(scaled_X, scaled_Y)
    # find_baseline(X_train, X_test, Y_train, Y_test)

    # coefficients = create_linear_model(X_train, X_test, Y_train, Y_test)
    # compute_model_parameters(X, Y)
    # gradient_descent_solver(scaled_X_train, scaled_X_test, scaled_Y_train, scaled_Y_test, coefficients)
    # actual, predictions, betas, nobs, p = create_custom_linear_model(X_train, X_test, Y_train, Y_test, coefficients)
    # compute_custom_model_statistics(X_test, nobs, p, betas, actual, predictions)
    find_best_model(X, Y)


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


def find_baseline(X_train, X_test, Y_train, Y_test):
    fitModel = DummyClassifier(strategy='stratified').fit(X_train, Y_train)
    baseline_score = fitModel.score(X_test, Y_test)
    print("Baseline Score (R^2): {}".format(baseline_score))


def create_linear_model(X_train, X_test, Y_train, Y_test):
    ''' TODO...
        - Predict the wine quality using the test set and compare the accuracy to the actual quality. Comment.
        - Print the parameter estimates and their 95% confidence intervals in a single table. (Suggest using
          confint()), and cbind()
    '''
    X_train = add_constant(X_train)
    regressionResult = OLS(Y_train, X_train).fit()
    print(regressionResult.summary())

    # Print various attributes of the OLS fitted model
    # print("R Squared: {}".format(regressionResult.rsquared))
    # print("SSE: {}".format(regressionResult.ess))
    # print("SSR: {}".format(regressionResult.ssr))
    # print("Residual MSE: {}".format(regressionResult.mse_resid))
    # print("Total MSE: {}".format(regressionResult.mse_total))
    # print("Model MSE: {}".format(regressionResult.mse_model))
    # print("F-Value: {}".format(regressionResult.mse_model/regressionResult.mse_resid))
    # print("NOBS: {}".format(regressionResult.nobs))
    # print("Centered TSS: {}".format(regressionResult.centered_tss))
    # print("Uncentered TSS: {}".format(regressionResult.uncentered_tss))
    # print("DF Model: {}".format(regressionResult.df_model))
    # print("DF Resid: {}".format(regressionResult.df_resid))

    predictions = regressionResult.predict(X_train)

    nobs, p = X_train.shape
    eaic = extractAIC(nobs, p, Y_train, predictions)
    print("Extract AIC: {}".format(eaic))

    # create linear regression object to grab coefficients
    reg = linear_model.LinearRegression().fit(X_train, Y_train)

    return reg.coef_


def create_custom_linear_model(X_train, X_test, Y_train, Y_test, coefficients):
    X_train = add_constant(X_train)
    n, p = X_train.shape
    betas = np.zeros(p).reshape(p, 1)
    error_vector = Y_train - X_train.dot(betas)
    betas = np.dot(np.dot(np.linalg.inv(X_train.T.dot(X_train)), X_train.T), Y_train).reshape(p, 1)

    print("Libraray linear model coefficients: {}".format(coefficients))
    print("Custom model parameters: {}".format(betas))

    n_test, p_test = X_test.shape
    X_test = add_constant(X_test)

    predictions = X_train.dot(betas).reshape(n,1)
    rmse = math.sqrt(mean_squared_error(Y_train, predictions))
    print("Root Mean Squared Error: {}".format(rmse))

    return Y_train, predictions, betas, n, p


def compute_custom_model_statistics(X, nobs, p, betas, actual, predictions):
    ''' TODO...
        - [QUESTION] If the errors were not nearly normal what might be the problem?
        - [QUESTION] Most residual errors are less than 1, what does that mean ?
    '''
    plot = False

    actual = actual.reshape(nobs,1)
    residuals = np.subtract(actual, predictions)
    num_lt_1 = sum([1 for r in residuals if math.fabs(r) < 1])
    print("Percent of residuals less than 1: {}%".format((num_lt_1/len(residuals))*100))

    if plot:
        plot_feature_histograms(pd.DataFrame(residuals))

    # -------------------------------------------------------

    create_summary_table(X, nobs, p, betas, actual, predictions)
    return


def compute_model_parameters(X, Y):
    X = add_constant(X)
    return


def gradient_descent_solver(X_train, X_test, Y_train, Y_test, coefficients):
    ''' TODO...
        - Print the estimated parameters from the lm() model, your normal solver, and your gradient descent
          solver side by side. Comment.
        - Predict the wine quality using the gradient descent parameter using the test set and compare to the
          actual quality in the test set
    '''
    def cost(X, Y, B):
        e = np.sum((X.dot(B) - Y[0]) ** 2)
        return e

    learning_rate = 0.00001
    convergence = 1
    print("Learning Rate: {}".format(learning_rate))
    print("Convergence Criterion: {}".format(convergence))

    X_train = add_constant(X_train)
    xtrain_rows, xtrain_columns = X_train.shape
    n = xtrain_columns

    B = np.ones(n).reshape(n, 1)
    Y_train = Y_train.reshape(xtrain_rows, 1)
    c = cost(X_train, Y_train, B)

    while c > convergence:
        dldw = -2 * X_train.T.dot(np.subtract(Y_train[0], X_train.dot(B)))
        B = B - dldw * learning_rate
        c = cost(X_train, Y_train, B)

    print("Gradient Descent Final Cost: {}".format(c))
    print("Gradient Descent Parameters: {}".format(B))
    print("Library linear model coefficients: {}".format(coefficients))

    # Predict the wine quality based on the GD parameters
    predictions = add_constant(X_test).dot(B)
    # for i in range(len(predictions)):
    #     print("Actual quality: {}, Predicted quality: {}".format(Y_test[i], predictions[i]))
    rmse = math.sqrt(mean_squared_error(Y_test, predictions))

    print("Gradient Descent Root Mean Squared Error: {}".format(rmse))

    for i in range(len(predictions)):
        print("Actual quality: {}, Predicted quality: {}".format(Y_test[i], predictions[i]))


def find_best_model(X, Y):
    x_rows, x_columns = X.shape
    aic = {}
    for k in range(1, x_columns):
        for v in itertools.combinations(range(0, x_columns), k):
            # print(v)
            # print(list(v))
            p = X[:, list(v)]
            # print("P: {}".format(p))
            # print("X Shape: {}".format(p.shape))
            # print("Y Shape: {}".format(Y.shape))
            regr = OLS(Y, add_constant(p)).fit()
            aic[v] = regr.aic
            print("V: {} \n AIC: {} \n \n".format(v, aic[v]))
    print(pd.Series(aic).idxmin())
    return


def create_summary_table(X, nobs, p, betas, actual, predictions):
    '''
        - Create and print a table similar to that in lm() output for your theta values
          for estimates, compute standard error, T values, and P values.

          http://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#RegressionResults.ess
    '''
    rows, _ = actual.shape
    x_n, _ = X.shape

    residuals = np.subtract(actual.reshape(rows, 1), predictions)

    print("\n\nRegression Results Summary Table")
    print("Residuals:")
    print("Min\t1Q\tMedian\t3Q\tMax")
    minimum = round(residuals.min(), 3)
    quart_one = round(np.percentile(residuals, 25), 3)
    median = round(np.median(residuals), 3)
    quart_three = round(np.percentile(residuals, 75), 3)
    maximum = round(residuals.max(), 3)

    print("{}\t{}\t{}\t{}\t{}".format(minimum, quart_one, median, quart_three, maximum))

    # ------------------ BEGIN TABLE OUTPUT ------------------

    print("\nCoefficients:")
    print("\t\tEstimate\tStd. Error\tt-value\t\tPr")

    print("(Intercepts)\t{}".format(round(betas[0][0], 3)))
    for i in range(1, p):
        print("x{}\t\t{}".format(i, round(betas[i][0], 3)))

    # ------------------ BEGIN FOOTER OUTPUT ------------------
    sse = calculate_sse(actual, predictions)
    ssr = calculate_ssr(actual, predictions)
    sst = calculate_sst(actual, predictions)

    # print("\nSSE: {}".format(sse))
    # print("SSR: {}".format(ssr))
    # print("SST: {}".format(sst))

    dfr = nobs-p
    dfm = p-1

    aic = AIC(nobs, sse, ssr, dfm, actual, predictions)
    print("Akaike's information criteria (AIC): {}".format(aic))

    rse = sse / dfr
    print("\nResidual Standard Error {} on {} degrees of freedom.".format(rse, dfr))

    r_squared = round(calculate_r_squared(ssr, sst), 3)
    adj_r_squared = round(calculate_adj_r_squared(nobs, ssr, sst, dfr, dfm), 3)
    print("Multiple R-squared: {}, Adjusted R-squared: {}".format(r_squared, adj_r_squared))

    mse_model = calculate_mse_model(sse, dfm)
    mse_resid = calculate_mse_resid(actual, predictions)

    f_value = calculate_f_value(mse_model, mse_resid)
    f_pvalue = calculate_fp_value(f_value, dfm, dfr)

    print("F-statistic: {} on {} and {} DF, p-value: {}".format(round(f_value, 3), dfm, dfr, f_pvalue))

    return residuals


# -------------- BEGIN STATISTICS HELPER FUNCTIONS ---------------------

def calculate_ssr(actual, predictions):
    ''' returns value residual sum of squared errors '''
    ssr = np.sum((actual.reshape(len(actual), 1) - predictions) ** 2)
    return ssr


def calculate_sse(actual, predictions):
    ''' returns value sum of squared errors '''
    ybar = np.mean(actual)
    sse = np.sum((predictions-ybar)**2)
    return sse


def calculate_sst(actual, predictions):
    ''' returns value total sum of squared errors '''
    ybar = np.mean(actual)
    sst = np.sum((actual - ybar)**2) 
    return sst


def AIC(n, sse, ssr, dfm, actual, predictions):
    ''' returns value for Akaike's information criteria '''
    ll = loglike(n, sse, ssr, dfm, actual, predictions)
    aic = 2*(dfm+1)-2*ll
    return aic


def loglike(n, sse, ssr, dfm, actual, predictions):
    ''' returns value for the log likelihood '''
    rse = sse / (n-dfm)

    nobs2 = n / 2.0
    llf = -math.log(ssr) * nobs2            # concentrated likelihood
    llf -= (1 + math.log(np.pi/nobs2))*nobs2  # with likelihood constant
    llf -= 1/(2*math.log(rse))

    return round(llf, 0)


def extractAIC(n, p, actual, predictions):
    ''' returns value for the extracted Akaike's information criteria '''
    sse = calculate_sse(actual, predictions)
    eaic = n*math.log(sse/n)+2*p
    return eaic


def calculate_r_squared(ssr, sst):
    ''' returns calculated value of r^2 '''
    r = 1 - (ssr/sst)
    return r


def calculate_adj_r_squared(n, ssr, sst, dfr, dfm):
    ''' returns calculated value of adjusted r^2 '''
    adjr = 1 - np.divide(n - 1, dfr) * (1 - calculate_r_squared(ssr, sst))
    return adjr


def calculate_mse_model(sse, dfm):
    ''' returns calculated value of the model mean squared error '''
    return sse/dfm

def calculate_mse_total(sst, dfr, dfm):
    ''' returns calculated value of the total mean squared error '''
    return sst / (dfr + dfm)


def calculate_mse_resid(actual, predictions):
    ''' returns calculated value of the residual mean squared error '''
    return mean_squared_error(actual, predictions)


def calculate_f_value(mse_model, dfr):
    ''' returns calculated value of the model's F statistic '''
    return mse_model / (dfr * 1.0)


def calculate_fp_value(fvalue, dfm, dfr):
    ''' returns calculated value of the model's p statistic '''
    fp = stats.f.sf(fvalue, dfm, dfr)
    fp = "%#6.3g" % fp
    return fp


# -------------- END STATISTICS HELPER FUNCTIONS ---------------------

if __name__ == "__main__":
    main()
