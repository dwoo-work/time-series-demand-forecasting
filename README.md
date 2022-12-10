# Time Series Forecasting

This will demonstrate to you how to perform time series forecasting using Python.

The purpose of this is to analyse time series data with the use of statistics, and create a model that performs prediction of the future.

In general, there are two commonly used forecasting models:

- Auto Regressive Integrated Moving Average (ARIMA): This takes past observations, to gain information for the next observation.
- Exponential Smoothing: This takes the average of the past forecasts, to gain information for the next observation.

After running a model, there is a need to ascertain the accuracy of it. This can be done by four different methods:

- Mean Square Error (MSE): This magnifies the difference, between the actual and the forecast.
- Mean Absolute Error (MAE): This identifies the absolute difference, between the actual and the forecast.
- Mean Error (ME): This identifies the biasness of the forecast, to see if there is a pattern of overshooting or undershooting.
- Root Mean Square Error (RMSE): This identifies the standard deviation of the residuals.

## ARIMA model

For this, there are four different variations for the ARIMA model, which are:

- Moving Average (MA):  This incorporates past errors of the series.
- Auto Regressive (AR): This incorporates past values of the series.
- Auto Regressive Moving Average (ARMA): This incorporates both the past errors and values of the series.
- Auto Regressive Integrated Moving Average (ARIMA): This incorporates both the past errors and values of the series, with an extra 'differencing' component that acts as a lag.

## Exponential Smoothing model

For this, there are three different variations for the Exponential Smoothing model, which are:

- Single Exponential Smoothing: This incorporates smoothing factor for the level.
- Double Exponential Smoothing (Holt's Method): This incorporates smoothing factor for the level, and the trend.
- Triple Exponential Smoothing (Holt Winter's Method): This incorporates smoothing factor for the level, the trend, and the seasonality.

For the smoothing factor for trend, there are two different types, which are:

- Additive: Seasonal variation remains constant throughout the series.
- Multiplicative: Seasonal variation changes throughout the series.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install:

- pandas: to perform data administration by pulling in .csv or .xlsx files.
- numpy: to perform data manipulation by establishing arrays.
- statsmodels: to provide clases and functions for estimation of different statistical models.
- matplotlib.pyplot: to create static, animated, and interactive visualisations.

```bash
pip install pandas
pip install numpy
pip install statsmodels
pip install matplotlib
```

## Sample Dataset

For this, you can download the sales_data_sample_utf8.csv file from the source folder, which is located [here](https://github.com/dwoo-work/MultiCriteriaABC.Analysis/blob/main/src).

Ensure that the file is in CSV UTF-8 format, to avoid UnicodeDecodeError later on.

## Code Explanation

### Part 1 - Lorem Ipsum

Lines X-X:  
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
```python   
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
```

## Credit

Sales Data Sample (https://www.kaggle.com/datasets/kyanyoga/sample-sales-data)

## License

[MIT](https://choosealicense.com/licenses/mit/)