import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from TFANN import ANNR
import csv
import os

import tensorflow as tf


def load_dates():
    dates = []
    # reads data from the file
    with open('tesla_data.csv', 'r') as stock_data:
        csvreader = csv.reader(stock_data)
        # skip 1st row
        next(csvreader)

        for row in csvreader:
            print(row[1].split('/')[0])
            dates.append(int(row[1].split('/')[0]))
    # scales the data to smaller values
    print(dates)
    return dates


def load_prices():
    prices = []

    with open('tesla_data.csv', 'r') as stock_data:
        csvreader = csv.reader(stock_data)
        # skip 1st row
        next(csvreader)

        for row in csvreader:
            # load in closing prices
            prices.append(float(row[5]))

    # scales the data to smaller values

    return prices


def show_model_predictions(x):
    print('show_model_Predictions runnning')
    dates = load_dates()
    prices = load_prices()
    # convert dates into a 1 column matrix for use with scikit.learn
    dates_fin = np.reshape(dates, (len(dates), 1))
    print('dates fin ' + str(dates_fin))

    # create 3 different support vector machines to preform regressions using different models.
    # Use the same error threshold on all models. This is to test which model is most accurate with the given data
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)  # degree of 2 because we have 2 params
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=.1)  # gamma is the amount of tolerance the model will have

    # train models with given data sets
    print('models about to be trained')
    svr_lin.fit(dates_fin, prices)
    print('linear trained')
    # svr_poly.fit(dates_fin, prices)
    # print('poly trainied')
    svr_rbf.fit(dates_fin, prices)
    print('rbf trained')

    print('show_model_Predictions runnning - model instantiated and dates/prices loaded ')
    # Plot the predictions of the three models
    plt.scatter(dates_fin, prices, color="black", label='Data')
    plt.plot(dates_fin, svr_rbf.predict(dates_fin), color='blue', label='RBF Model')
    # plt.plot(dates_fin,svr_poly.predict(dates_fin), color='red', label='Polynomial Model')
    plt.plot(dates_fin, svr_lin.predict(dates_fin), color='green', label='Linear Model')
    plt.xlabel('date(month prediction')
    plt.ylabel('Closing Price')

    plt.title('SVR')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0]  # ,svr_poly.predict(x)[0]


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print(str(show_model_predictions(3)))


if __name__ == "__main__":
    main()
