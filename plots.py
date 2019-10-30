"""
This module provides all plotting functions for both parts of the project:
1) Time series analysis on all monthly rentals
2) Time series analysis on broken down data between monthly member rentals and
   monthly casual rentals
"""


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm


def initial_plot(data):
    """
    Plots initial time series data prior to any analysis
    """
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(data.index.values, data['count'] / 100000,
            lw=2, color='mediumseagreen')

    ax.set_xlabel('Year', fontsize=20)
    ax.set_ylabel('Ride Count (100k)', fontsize=20)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title('Number of Bike Rentals per Month, Time Series', fontsize=30)
    plt.show()


def split_plot(train, test):
    """
    Plots both train and test data on the same graph
    """
    fig = plt.figure(figsize=(15, 8))

    plt.plot(train['count'] / 100000, lw=3)
    plt.plot(test['count'] / 100000, lw=3, color='navy')
    plt.legend(['Train', 'Test'], prop={'size': 24})

    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Ride Count (100k)', fontsize=20)

    plt.title('Number of Bike Rentals per Month, Time Series', fontsize=30)
    plt.show()


def decomposition(data):
    """
    Graphs seasonal decomposition
    """
    decomposition_data = seasonal_decompose(data['count'] / 100000, freq=12)
    fig = plt.figure()
    fig = decomposition_data.plot()
    fig.set_size_inches(15, 8)


def stationarity_plot(timeseries, window, title):
    """
    Calculates and plots rolling statistics
    """
    # Determining rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[window:] / 100000,
                    color='blue', label='Original')
    mean = plt.plot(rolmean / 100000, color='red', label='Rolling Mean')
    std = plt.plot(rolstd / 100000, color='black', label='Rolling Std')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Ride Count', fontsize=20)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


def acf_pacf(data):
    """
    Plots autocorrelation function (ACF) and partial autocorrelation function
    (PACF) outputs
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data['count'].iloc[1:], lags=13, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data['count'].iloc[1:], lags=13, ax=ax2)


def prediction_plot(model, training_set, testing_set, p, d, q, P, D, Q, m):
    """
    Plots model prediction on top of original data for both training and
    testing data for a given model. This is to act as a visual aid and not an
    indicator of which model is most optimal. This function is for part 1 of
    the project (all monthly rentals).
    """
    if m == 52:
        freq = 'Week'
        lw = 2
    elif m == 12:
        freq = 'Month'
        lw = 4
    else:
        lw = 2
        freq = '(test)'

    order = f'({p},{d},{q})'
    seasonal_order = f'({P},{D},{Q},{m})'

    predict_train = model.predict()
    predict_test = model.forecast(len(testing_set))

    fig = plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(training_set.index, training_set['count'],
             lw=lw, color='mediumturquoise')
    ax1.plot(testing_set.index, testing_set['count'], lw=lw, color='blue')
    ax1.plot(training_set.index, predict_train, lw=lw, color='magenta')

    ax2.plot(training_set.index, training_set['count'],
             lw=lw, color='mediumturquoise')
    ax2.plot(testing_set.index, testing_set['count'], lw=lw, color='blue')
    ax2.plot(testing_set.index, predict_test, lw=lw, color='orange')

    ax1.set_xlabel('Date', fontsize=20)
    ax1.set_ylabel('Count', fontsize=20)

    ax2.set_xlabel('Date', fontsize=20)
    ax2.set_ylabel('Count', fontsize=20)

    ax1.set_title(f'Number of Bike Rentals per {freq} Time Series, '
                  f'SARIMA {order}{seasonal_order}', fontsize=30)
    ax2.set_title(f'Number of Bike Rentals per {freq} Time Series, '
                  f'SARIMA {order}{seasonal_order}', fontsize=30)

    ax1.legend(['Train', 'Test',
                'Model Prediction on Training Set'], prop={'size': 24})
    ax2.legend(['Train', 'Test',
                'Model Prediction on Testing Set'], prop={'size': 24})
    plt.show()


def forecast_plot(model, master, n_forecast=12, predict_int_alpha=.2):
    """
    Plots predicted forecast and prediction interval alongside original data.
    This function is for part 1 of the project (all monthly rentals).
    """
    fig = plt.figure(figsize=(18, 9))
    plt.plot(master.index, master['count'] / 100000,
             lw=4, color='mediumorchid')
    forecast = model.forecast(n_forecast)
    predict_int = model.get_forecast(n_forecast
                                     ).conf_int(alpha=predict_int_alpha)
    forecast_dates = pd.date_range(master.index[-1], freq='m',
                                   periods=n_forecast + 1).tolist()[1:]
    plt.plot(forecast_dates, forecast / 100000, lw=6, color='indigo')
    plt.plot(forecast.index, predict_int['lower count'] / 100000,
             color='darkgray', lw=4)
    plt.plot(forecast.index, predict_int['upper count'] / 100000,
             color='darkgray', lw=4)
    plt.fill_between(forecast.index, predict_int['lower count'] / 100000,
                     predict_int['upper count'] / 100000, color='darkgray')
    plt.xlim([master.index[0], forecast.index[-1]])
    plt.ylim(0)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Ride Count (100k)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Data', 'Forecast', '80% Prediction Interval'],
               prop={'size': 24})
    plt.title(f'{n_forecast}-Month Forecast on Bike Rentals per Month',
              fontsize=30)
    plt.show()

    model.resid.hist()
    plt.title("Residual Histogram", fontsize=20)
    plt.show()

    qqplot(model.resid, line='s')
    plt.title('Q-Q Plot', fontsize=20)
    plt.show()


def initial_breakdown_graph(data):
    """
    Plots initial time series data broken down by member and casual rentals on
    the same graph prior to any analysis (Part 2)
    """
    fig = plt.figure(figsize=(18, 8))
    plt.plot(data.casual / 100000, lw=5, color='mediumorchid')
    plt.plot(data.member / 100000, lw=5, color='green')
    plt.xlabel('Year', fontsize=25)
    plt.ylabel('Ride Count (100k)', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Casual', 'Member'], prop={'size': 24})
    plt.xlim([data.index[0], data.index[-1]])
    plt.ylim(0)
    plt.title('Number of Casual and Member Rides per Month', fontsize=30)
    plt.show()


def prediction_plot_breakdown(model, training_set, testing_set, kind, p, d, q,
                              P, D, Q, m):
    """
    Plots model prediction on top of original data for both training and
    testing data for a given model. This is to act as a visual aid and not an
    indicator of which model is most optimal. This function is for part 2 of
    the project (broken down monthly rentals between member and casual).
    """
    if m == 52:
        freq = 'Week'
        lw = 2
    elif m == 12:
        freq = 'Month'
        lw = 4
    else:
        lw = 2
        freq = '(test)'

    order = f'({p},{d},{q})'
    seasonal_order = f'({P},{D},{Q},{m})'

    predict_train = model.predict()
    predict_test = model.forecast(len(testing_set))

    fig = plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(training_set.index,
             training_set[kind], lw=lw, color='mediumturquoise')
    ax1.plot(testing_set.index, testing_set[kind], lw=lw, color='blue')
    ax1.plot(training_set.index, predict_train, lw=lw, color='magenta')

    ax2.plot(training_set.index,
             training_set[kind], lw=lw, color='mediumturquoise')
    ax2.plot(testing_set.index, testing_set[kind], lw=lw, color='blue')
    ax2.plot(testing_set.index, predict_test, lw=lw, color='orange')

    ax1.set_xlabel('Date', fontsize=20)
    ax1.set_ylabel('Count', fontsize=20)

    ax2.set_xlabel('Date', fontsize=20)
    ax2.set_ylabel('Count', fontsize=20)

    ax1.set_title(
        f'Number of {kind} Bike Rentals per {freq} Time Series, SARIMA '
        f'{order}{seasonal_order}', fontsize=30)
    ax2.set_title(
        f'Number of {kind} Bike Rentals per {freq} Time Series, SARIMA'
        f' {order}{seasonal_order}',
        fontsize=30)

    ax1.legend(['Train', 'Test', 'Model Prediction on Training Set'],
               prop={'size': 24})
    ax2.legend(['Train', 'Test', 'Model Prediction on Testing Set'],
               prop={'size': 24})
    plt.show()


def model_plot_breakdown(model, master, kind):
    """
    Plots initial data alongside prediction (dotted) for any given model. This
    is to act as a visual aid and not an indicator of which model is most
    optimal. This function is for part 2 of the project (broken down monthly
    rentals between member and casual). This differs from
    prediction_plot_breakdown in that original data is not split between train
    and test and only one graph is output for this function.
    """
    fig = plt.figure(figsize=(18, 9))
    plt.plot(master[kind.lower()] / 100000, lw=4, color='royalblue')
    plt.plot(master.index, model.predict() / 100000, lw=5, color='black',
             ls=':')
    plt.title(
        f'Number of {kind} Bike Rentals per Month: Model of Historical Data',
        fontsize=30)
    plt.xlabel('Year', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Ride Count (100k)', fontsize=25)
    plt.xlim([master.index[0], master.index[-1]])
    plt.ylim(0)
    plt.legend(['Data', 'Model'], prop={'size': 24})
    plt.show()


def model_plot_breakdown_both(model_member, model_casual, master):
    """
    Plots initial data alongside prediction (dotted) for any given model and
    plots member and casual data/predictions on the same graph. This is to act
    as a visual aid and not an indicator of which model is most optimal. This
    function is for part 2 of the project (broken down monthly rentals between
    member and casual).
    """
    fig = plt.figure(figsize=(18, 9))
    plt.plot(master.member / 100000, lw=4, color='mediumseagreen')
    plt.plot(master.casual / 100000, lw=4, color='mediumorchid')
    plt.plot(master.index, model_member.predict() /
             100000, lw=5, color='black', ls=':')
    plt.plot(master.index, model_casual.predict() /
             100000, lw=5, color='black', ls=':')
    plt.title(
        f'Number of Member and Casual Bike Rentals per Month: Model of '
        f'Historical Data',
        fontsize=30)
    plt.xlabel('Year', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Ride Count (100k)', fontsize=25)
    plt.xlim([master.index[0], master.index[-1]])
    plt.ylim(0)
    plt.legend(['Member Data', 'Casual Data', 'Models'], prop={'size': 24})
    plt.show()


def forecast_plot_breakdown_both(model_member, model_casual, master,
                                 n_forecast, predict_int_alpha=.2):
    """
    Plots predicted forecast and prediction interval alongside original data.
    Plots both the member forecasts and casual forecasts on the same graph.
    This function is for part 2 of the project (broken down monthly rentals
    between member and casual).
    """
    fig = plt.figure(figsize=(18, 9))
    plt.plot(master.index, master['member'] /
             100000, lw=4, color='mediumseagreen')
    plt.plot(master.index, master['casual'] / 100000, lw=4,
             color='mediumorchid')

    forecast_member = model_member.forecast(n_forecast)
    forecast_casual = model_casual.forecast(n_forecast)
    predict_int_member = model_member.get_forecast(
        n_forecast).conf_int(alpha=predict_int_alpha)
    predict_int_casual = model_casual.get_forecast(
        n_forecast).conf_int(alpha=predict_int_alpha)
    forecast_dates = pd.date_range(
        master.index[-1], freq='m', periods=n_forecast + 1).tolist()[1:]

    plt.plot(forecast_dates, forecast_member / 100000, lw=6, color='darkgreen')
    plt.plot(forecast_dates, forecast_casual / 100000, lw=6, color='indigo')

    plt.plot(forecast_casual.index,
             predict_int_casual['lower casual'] / 100000, color='thistle',
             lw=4)
    plt.plot(forecast_casual.index,
             predict_int_casual['upper casual'] / 100000, color='thistle',
             lw=4)

    plt.plot(forecast_member.index,
             predict_int_member['lower member'] / 100000, color='powderblue',
             lw=4)
    plt.plot(forecast_member.index,
             predict_int_member['upper member'] / 100000, color='powderblue',
             lw=4)

    plt.fill_between(forecast_casual.index,
                     predict_int_casual['lower casual'] /
                     100000, predict_int_casual['upper casual'] / 100000,
                     color='thistle')
    plt.fill_between(forecast_member.index,
                     predict_int_member['lower member'] /
                     100000, predict_int_member['upper member'] / 100000,
                     color='powderblue')

    plt.xlim([master.index[0], forecast_member.index[-1]])
    plt.ylim(0)
    plt.xlabel('Year', fontsize=25)
    plt.ylabel('Ride Count (100k)', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Member Rides', 'Casual Rides', 'Forecast on Member Rides',
                'Forecast on Casual Rides'], prop={'size': 24})
    plt.title(
        f'{n_forecast}-Month Forecast on Bike Rentals per Month', fontsize=30)
