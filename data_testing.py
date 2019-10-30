"""
Module provides functions for manipulation and testing of data for both parts
of project:

1) Time series analysis on all monthly rentals
2) Time series analysis on broken down data between monthly member rentals and
   monthly casual rentals
"""
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pandas as pd
import plots as p
import functions as f


def test_stationarity(data, window, title):
    """
    Creates stationarity plot, returns results of Dickey-Fuller test
    """
    p.stationarity_plot(data, window, title)
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic',
                                'p-value',
                                '#Lags Used',
                                'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def stationarity_autocorrelation_test_original(data):
    """
    Tests stationarity and autocorrelation of time series
    """
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelation\n of the original time series.')
    test_stationarity(data['count'], window=12,
                      title="Original Time series")
    p.acf_pacf(data)


def stationarity_autocorrelation_test_first_diff(data):
    """
    Tests stationarity and autocorrelation of first difference
    """
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelation\n of the first difference.')
    first_diff = f.order_difference(data)
    test_stationarity(first_diff['count'], window=12,
                      title="First Order Difference")
    p.acf_pacf(first_diff)


def stationarity_autocorrelation_test_second_diff(data):
    """
    Tests stationarity and autocorrelation of second difference
    """
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelation\n of the second difference.')
    first_diff = f.order_difference(data)
    second_diff = f.order_difference(first_diff)
    test_stationarity(second_diff['count'], window=12,
                      title="Second Order Difference")
    p.acf_pacf(second_diff)


def stationarity_test_seasonal_diff(data):
    """
    Tests stationarity and autocorrelation of seasonal difference
    """
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelation\n of the seasonal difference.')
    season = f.seasonal_difference(data)
    test_stationarity(season['count'], window=12,
                      title="Seasonal Difference")
    p.acf_pacf(season)


def stationarity_test_seasonal_first_diff(data):
    """
    Tests stationarity and autocorrelation of first difference and first
    seasonal difference
    """
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelation\n of the seasonal difference of the first '
          'difference.')
    first_diff = f.order_difference(data)
    season_first = f.order_difference(first_diff)
    test_stationarity(season_first['count'], window=12,
                      title="Seasonal Difference of First Order Difference")
    p.acf_pacf(season_first)


def sarima(train_df, test_df):
    """
    Evaluates SARIMA models for all combinations of orders and seasonal orders
    to be tested. Plots model output on top of training and test data as visual
    aid, prints training and testing mean squared error for each model. This
    function is for part 1 of the project (all monthly rentals).
    """
    orders = [(2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3)]
    seasonal_orders = [(0, 1, 0, 12), (1, 1, 0, 12), (0, 1, 1, 12)]

    for o in orders:
        for s_o in seasonal_orders:
            model = sm.tsa.statespace.SARIMAX(train_df['count'],
                                              order=(o[0], o[1], o[2]),
                                              seasonal_order=(
                                                  s_o[0], s_o[1], s_o[2],
                                                  s_o[3])).fit()
            print(model.summary())
            _, _ = compare_mse(model, train_df, test_df)
            p.prediction_plot(model, train_df, test_df,
                                  o[0], o[1], o[2], s_o[0], s_o[1], s_o[2],
                                  s_o[3])


def sarima_breakdown(train_df, test_df):
    """
    Evaluates SARIMA models for all combinations of orders and seasonal orders
    to be tested for both monthly member rentals and monthly casual rentals.
    Plots model output on top of training and test data as visual
    aid, prints training and testing mean squared error for each model. This
    function is for part 2 of the project (broken down monthly rentals between
    member and casual).
    """
    orders = [(2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3)]
    seasonal_orders = [(0, 1, 0, 12), (1, 1, 0, 12), (0, 1, 1, 12)]
    for o_val in orders:
        for s_o in seasonal_orders:
            print(f'Orders: {o_val}. Seasonal Orders: {s_o}')
            model = sm.tsa.statespace.SARIMAX(train_df['member'],
                                              order=(o_val[0], o_val[1],
                                                     o_val[2]),
                                              seasonal_order=(s_o[0],
                                                              s_o[1],
                                                              s_o[2],
                                                              s_o[3])).fit()
            print(model.summary())
            _, _ = compare_mse_breakdown(model, train_df,
                                         test_df, 'member')
            p.prediction_plot_breakdown(model, train_df, test_df, 'member',
                                            o_val[0], o_val[1], o_val[2],
                                            s_o[0],
                                            s_o[1], s_o[2], s_o[3])

            model = sm.tsa.statespace.SARIMAX(train_df['casual'],
                                              order=(o_val[0], o_val[1],
                                                     o_val[2]),
                                              seasonal_order=(s_o[0], s_o[1],
                                                              s_o[2],
                                                              s_o[3])).fit()
            print(model.summary())
            _, _ = compare_mse_breakdown(model, train_df,
                                         test_df, 'casual')
            p.prediction_plot_breakdown(model, train_df, test_df, 'casual',
                                            o_val[0], o_val[1], o_val[2],
                                            s_o[0],
                                            s_o[1], s_o[2], s_o[3])


def compare_mse(sarima_model, training_set, testing_set):
    """
    Calculates mean squared errors (MSE) for both the training and testing data
    for a given model so that mse can be compared across models. This function
    is for part 1 of the project (all monthly rentals).
    """
    predict_train = sarima_model.predict(start=0, end=(len(training_set)))
    predict_test = sarima_model.predict(start=(len(training_set)),
                                        end=(len(training_set) +
                                             len(testing_set)))
    train_mse = mean_squared_error(training_set['count'], predict_train[:-1])
    test_mse = mean_squared_error(testing_set['count'], predict_test[:-1])
    print('Training MSE: ', '{:.2e}'.format(train_mse))
    print('Testing MSE: ', '{:.2e}'.format(test_mse))
    return train_mse, test_mse


def compare_mse_breakdown(sarima_model, training_set, testing_set, kind):
    """
    Calculates mean squared errors (MSE) for both the training and testing data
    for a given model so that mse can be compared across models. This function
    is for part 2 of the project (broken down monthly rentals between member
    and casual).
    """
    predict_train = sarima_model.predict(start=0, end=(len(training_set)))
    predict_test = sarima_model.predict(start=(len(training_set)),
                                        end=(len(training_set) +
                                             len(testing_set)))
    train_mse = mean_squared_error(training_set[kind], predict_train[:-1])
    test_mse = mean_squared_error(testing_set[kind], predict_test[:-1])
    print('Training MSE: ', '{:.2e}'.format(train_mse))
    print('Testing MSE: ', '{:.2e}'.format(test_mse))
    return train_mse, test_mse
