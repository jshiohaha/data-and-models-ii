import sys
import pydot
import numpy as np
import pandas as pd
import math as  math
import matplotlib.pyplot as plt
import operator

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
    # plot_feature_histograms(df, title="Unscaled Data Set")

    # for i in range(12):
    #     plot_feature_histograms(df.iloc[:,i], title="Feature {}".format(i+1))

    # describe_data_frame(df)

    # plot_feature_histograms(df)
    X, Y = create_matrix_and_vector_from_data_frame(df)
    X_train, X_test, Y_train, Y_test = create_train_test_set(X, Y)
    find_baseline(X_train, X_test, Y_train, Y_test)

    scaled_df = scale_dataset(df)
    scaled_X, scaled_Y = create_matrix_and_vector_from_data_frame(scaled_df)
    scaled_X_train, scaled_X_test, scaled_Y_train, scaled_Y_test = create_train_test_set(scaled_X, scaled_Y)
    # find_baseline(X_train, X_test, Y_train, Y_test)

    coefficients = create_linear_model(X_train, X_test, Y_train, Y_test)
    # compute_model_parameters(X, Y)
    gradient_descent_solver(scaled_X_train, scaled_X_test, scaled_Y_train, scaled_Y_test, coefficients)
    # actual, predictions, betas, nobs, p = create_custom_linear_model(X_train, X_test, Y_train, Y_test, coefficients)
    # sys.exit()
    # compute_custom_model_statistics(X_train, X_test, nobs, p, betas, actual, predictions)


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


def plot_feature_histograms(dataframe, title=None):
    # features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
    #             "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",
    #             "quality"]

    # TODO: might need to change this because of all the explanatory variables... it would
    # be simpler to break it out
    fig = plt.figure()

    if title is not None:
        fig.suptitle(title)

    dataframe.plot.hist(stacked=True)
    plt.show()


def scale_dataset(dataframe):
    print(">> Normalizing the data set\n")
    # http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    # looking at the histograms to choose between normal and uniform scaling

    plot = False

    # normal scaling
    scaled_data_array = Normalizer().fit_transform(dataframe)
    df_norm = pd.DataFrame(scaled_data_array, index=dataframe.index, columns=dataframe.columns)

    # describe_data_frame(df_norm)
    if plot:
        plot_feature_histograms(df_norm, title="Normal Scaled Data Set")

    # uniform scaling
    scaled_data_array = QuantileTransformer(output_distribution='uniform').fit_transform(dataframe)
    df_uni = pd.DataFrame(scaled_data_array, index=dataframe.index, columns=dataframe.columns)

    # describe_data_frame(df_uni)
    if plot:
        plot_feature_histograms(df_uni, title="Uniform Scaled Data Set")

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
    def loss_func(Y_test, X_test):
        diff = np.abs(ground_truth - predictions).max()
        return np.log(1 + diff)

    reg = DummyClassifier(strategy='stratified').fit(X_train, Y_train)
    
    baseline_score = reg.score(X_test, Y_test)
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
    print("Standard Errors: {}".format(regressionResult.bse))

    predictions = regressionResult.predict(X_train)

    nobs, p = X_train.shape
    eaic = extractAIC(nobs, p, Y_train, predictions)
    print("Extract AIC: {}".format(eaic))

    params = regressionResult.params

    # n, p = X_test.shape
    # X_test = add_constant(X_test)
    # predictions = X_test.dot(params).reshape(n,1)

    # num_matches = 0
    # for i in range(len(Y_test)):
    #     p = int(round(predictions[i][0], 0))
    #     is_match = (Y_test[i] == p)

    #     if is_match:
    #         num_matches += 1

    #     print("Actual: {}, Predictions: {}... Match: {}".format(Y_test[i], p, is_match))

    # print("Number of matches: {}, Total number of Instances: {}".format(num_matches, n))
    # print("Percent correct guesses: {}%".format(round((num_matches/n)*100, 3)))

    return params


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

    # ------------ BEGIN CUSTOM LINEAR MODEL PREDICTIONS ---------------

    predictions = X_train.dot(betas).reshape(n,1)

    num_matches = 0
    for i in range(len(Y_train)):
        p = int(round(predictions[i][0], 0))
        is_match = (Y_train[i] == p)

        if is_match:
            num_matches += 1

        print("Actual: {}, Predictions: {}... Match: {}".format(Y_train[i], p, is_match))

    print("Number of matches: {}, Total number of Instances: {}".format(num_matches, n))
    print("Percent correct guesses: {}%".format(round((num_matches/n)*100, 3)))

    return Y_train, predictions, betas, n, p


def compute_custom_model_statistics(X, X_test, nobs, p, betas, actual, predictions):
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

    create_summary_table(X, X_test, nobs, p, betas, actual, predictions)
    return


def compute_model_parameters(X, Y):
    X = add_constant(X)
    return


def gradient_descent_solver(X_train, X_test, Y_train, Y_test, coefficients):
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
    print("Libraray linear model coefficients: {}".format(coefficients))

    n_test, p_test = X_test.shape
    # Predict the wine quality based on the GD parameters
    predictions = add_constant(X_test).dot(B)

    num_matches = 0
    for i in range(n_test):
        y = round(Y_train[i][0], 3)
        p = round(predictions[i][0], 3)
        is_match = (y == p)

        if is_match:
            num_matches += 1

        print("Actual: {}, Predictions: {}... Match: {}".format(y, p, is_match))

    print("Number of matches: {}, Total number of Instances: {}".format(num_matches, n_test))
    print("Percent correct guesses: {}%".format(round((num_matches/n_test)*100, 3)))

    sys.exit()

    rmse = math.sqrt(mean_squared_error(Y_test, predictions))

    print("Gradient Descent Root Mean Squared Error: {}".format(rmse))

    for i in range(len(predictions)):
        print("Actual quality: {}, Predicted quality: {}".format(Y_test[i], predictions[i]))


def create_summary_table(X, X_test, nobs, p, betas, actual, predictions):
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

    # std_err_arr = compute_standard_error(X)
    std_err_arr = [2.07105460e+01, 2.38423359e-02, 1.30251968e-01, 1.11762701e-01,
                   8.43493292e-03, 6.48855536e-01, 9.85506087e-04, 4.39725828e-04,
                   2.10207985e+01, 1.21496023e-01, 1.16605896e-01, 2.67669971e-02]

    t_statistic = compute_t_statistic(std_err_arr, betas)
    p_statistic = compute_p_statistic(t_statistic, nobs)

    print("\nCoefficients:")
    print("\t\tEstimate\tStd. Error\tt-value\t\tPr")

    print("(Intercepts)\t{}\t\t{}\t\t{}\t\t{}".format(round(betas[0][0], 3), round(std_err_arr[0], 3), round(t_statistic[0], 3), round(p_statistic[0], 3)))
    for i in range(1, p):
        print("x{}\t\t{}\t\t{}\t\t{}\t\t{}".format(i, round(betas[i][0], 3), round(std_err_arr[i], 3), round(t_statistic[i], 3), round(p_statistic[i], 3)))

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
    print("\nAkaike's information criteria (AIC): {}".format(aic))

    rse = sse / dfr
    # print("\nResidual Standard Error {} on {} degrees of freedom.".format(round(rse, 3), dfr))

    r_squared = round(calculate_r_squared(ssr, sst), 3)
    adj_r_squared = round(calculate_adj_r_squared(nobs, ssr, sst, dfr, dfm), 3)
    print("Multiple R-squared: {}, Adjusted R-squared: {}".format(r_squared, adj_r_squared))

    mse_model = calculate_mse_model(sse, dfm)
    mse_resid = calculate_mse_resid(actual, predictions)

    f_value = calculate_f_value(mse_model, mse_resid)
    f_pvalue = calculate_fp_value(f_value, dfm, dfr)

    print("F-statistic: {} on {} and {} DF, p-value: {}\n".format(round(f_value, 3), dfm, dfr, f_pvalue))

    return residuals


# -------------- BEGIN STATISTICS HELPER FUNCTIONS ---------------------
# Useful link for calculating statistics on arrays and vectors in Python
# https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html

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


def compute_standard_error(X):
    ''' returns calculated value of standard errors of matrix X in an array '''
    cov_matrix = np.cov(X.T, rowvar=True)
    return np.sqrt(np.diag(cov_matrix))


def compute_t_statistic(std_err_arr, coefficients):
    ''' returns calculated value of t-statistics for each coefficent in an array '''
    t_stat_arr = []
    for i in range(len(std_err_arr)):
        t_stat = (coefficients[i] / std_err_arr[i])[0]
        t_stat_arr.append(t_stat)
    return t_stat_arr


def compute_p_statistic(t_statistic, n):
    ''' returns calculated value of p-statistics for each coefficent in an array '''
    p_stat_arr = []
    for i in range(len(t_statistic)):
        pval = stats.t.sf(np.abs(t_statistic[i]), n-1)*2
        p_stat_arr.append(pval)
    return p_stat_arr


# -------------- END STATISTICS HELPER FUNCTIONS ---------------------

if __name__ == "__main__":
    main()
