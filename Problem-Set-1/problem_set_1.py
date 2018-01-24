import pandas as panda
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# TODO: Question 5, 6, 7, 8
# local hw file: file:///Users/jacobshiohira/Desktop/ACADEMICS/JUNYA/SPRING/DATA-MODELS-II/HOMEWORK/HOMEWORK-1/Problem%20Set%201%20%20Assignment%20PDF.html

# intro on how to use sklearn for multi-layer perceptrons, compute precision scores, confusion matrices, etc.
# https://www.datacamp.com/community/tutorials/deep-learning-python

# boolean flag that helps us decide whether or not to plot the data
plot = False

def main():
    '''
        Solution for Problem Set 1 in Data & Models II (RAIK 370H)

        All of the functions pertaining to the set of questions are below
        and you should just be able to copy and paste the function name
        and universally include the dataframe as a parameter to see the
        results. Each function itself describes what it's doing along
        with the parts of the problem set question it addresses.
    '''

    # utilizing panda DataFrame, which mimics R DataFrame definition
    # reference found http://pandas.pydata.org/pandas-docs/stable/dsintro.html#dataframe
    problem_set_data = panda.read_csv("ProbSet1.csv")

    if plot:
        visualize_data_3d(problem_set_data)

    normalize_and_visualize_data(problem_set_data)


def build_model(df):
    return


def test_and_validate_model(df):
    return


def normalize_and_visualize_data(df):
    ''' Aimed at scaling each feature to lie in the range from 0 to 1
        and then making a 3d scatter plot of the scaled data.

        1. Print the mean, median, max, and min for the scaled data
        2. Make a 3D scatterplot of the scaled data
    '''
    scaler = MinMaxScaler()
    # scaler = StandardScaler().fit(df) # yields scaled features on range -1, 1
    scaled_data_array = scaler.fit_transform(df)
    df_norm = panda.DataFrame(scaled_data_array, index=df.index, columns=df.columns)
    describe_data_frame(df_norm)
    
    if plot:
        visualize_data_3d(df_norm)


def visualize_data_3d(df):
    ''' Aimed at visualizing the data in 3D space and then analyzing
        correlations between variables.

        1. Make a 3D scatterplot of the data. Identify observations as good or bad loans using color. Comment.
        2. Output the correlations between the explanatory variables (the features.) Comment.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # define paramtetric vectors for the scatter plot
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

    ax.legend() # matches `color` to `label`
    plt.show()


def check_nan(df):
    # TODO: what counts as NA in the data set -- a 0 in the ratio column?
    print(df.isnull().values.any())


def describe_data_frame(df):
    ''' Aimed at describing characteristics of the problem set 1 data set for the
        following parts:

        1. How many observations are in the data set?
        2. How many features are in the data set?
        3. What is the data type for each feature and the response?
        4. Print the mean, median, max, and min values for each feature. Suggest using summary()
    '''
    rows, columns = df.shape
    print("num rows (observations): " + str(rows))
    print("num cols (features): " + str(columns))
    print("\nData Set Data Types...\n")
    print(df.info())
    print("\nData Set Statistics...")
    print(df.describe())

if __name__ == "__main__":
    main()