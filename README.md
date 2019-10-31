# Capital Bikeshare Time Series

## Project Members:

* __Marissa Eppes__
* __Kyle Hayes__

## Goal: 

__Develop and deploy a Seasonal ARIMA time series model capable of forecasting Capital Bikeshare bicycle rentals over the next 12 months__
#### Part 1: Model all Capital Bikeshare rentals by sampling frequency of rentals since 2010 monthly
#### Part 2: Break data down into Member and Casual rentals and model each separately

## Files in Repository:

* __technical_notebook.ipynb__ - step-by-step walk through of the modeling/optimization process with rationale explaining decisions made. This notebook starts by exploring stationarity, ACFs, and PACFs for the data. From here, many models are tested and compared against one another, and a 12-month forecast is made using the best model. The notebook then repeats the process for data broken down according to member vs. casual rentals.

* __data_prep.py__ - gathers and prepares data for analysis for both parts of project

* __data_testing.py__ - provides functions for manipulation and testing of data for both parts of project

* __functions.py__ - provides general functions used in other modules and technical notebook

* __plots.py__ - provides all plotting functions for both parts of the project, including seasonality plots, ACF/PCF, as well as presentation visual aids

* __cleaned_for_testing.csv__ - clean, concatenated data ready for modeling in Part 1

* __master_breakdown.csv__ - clean, concatenated data, broken down according to member vs. casual rentals, ready for modeling in Part 2

* __bikeshare_presentation.pdf__ - final presentation slides
