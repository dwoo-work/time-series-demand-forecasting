## TIME SERIES FORECASTING

#------------------------------------------------------------------------------------------
# PART 1 - PREPARATION

# Import the required libraries.

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels as sms

import warnings
import itertools

# Import the specific function from Pylab abd Statsmodels.

from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Adjust the style and size of the plots later on.

plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 16, 8

# Import and clean the CSV dataframe.

sales = pd.read_csv('sales_data_sample_utf8.csv')

sales = sales.drop_duplicates()

# Create a new CSV dataframe (cleaned data).

sales_clean = sales.copy()
sales_clean.info()

# Create a new columnn for Revenue within the sales_clean dataframe.

sales_clean['Revenue'] = sales['PRICEEACH'] * sales['QUANTITYORDERED']

# Change ORDERDATE from object to datetime.

sales_clean['ORDERDATE'] = pd.to_datetime(sales_clean['ORDERDATE'])

# Create a date column within the sales_clean dataframe.

sales_clean['date'] = sales_clean['ORDERDATE'].dt.strftime("%Y-%m-%d")
sales_clean['date'] = pd.to_datetime(sales_clean['date'])

# Create different columns for week, month, and year within the sales_clean dataframe.

sales_clean['month'] = sales_clean.date.dt.month
sales_clean['year'] = sales_clean.date.dt.year
sales_clean['week'] = sales_clean.date.dt.week

# Create a variable for time series, and plot a line chart using it.

time_series = sales_clean.groupby(['week', 'month', 'year']).agg(date = ('date', 'first'), total_revenue = ('Revenue', np.sum)).reset_index().sort_values('date')

# Index the date in time_series dataframe.

time_series.info()
time_series['date'] = pd.to_datetime(time_series['date'])
time_series = time_series.set_index('date')

# Create a variable for monthly total revenue, and create a line chart for it.

monthly_series = time_series.total_revenue.resample('M').sum()
monthly_series.plot()

#------------------------------------------------------------------------------------------
# PART 2 - DISSECT MONTHLY SERIES DATA INTO SEASONALITY, TREND, AND REMAINDER

# Use the decomposition function to dissect seasonality and trend from the time series data.

components = sm.tsa.seasonal_decompose(monthly_series)
components.plot()

seasonality = components.seasonal
trend = components.trend
remainder = components.resid

#------------------------------------------------------------------------------------------
# PART 3 - PERFORM STATIONALITY TEST FOR MONTHLY SERIES DATA

# Plot the monthly series chart with the actual data, mean, and the standard deviation.

monthly_series.plot(label = 'actual')
monthly_series.rolling(window = 12).mean().plot(label = 'mean')
monthly_series.rolling(window = 12).std().plot(label = 's.d')
plt.legend(loc = 'upper left')

# Run the Augmented Dickey-Fuller (ADF) test to confirm stationality.
# The P-Value is 0.004768. Therefore, reject null hypothesis, and confirm that the data is stationary. Therefore, use ARIMA model to compute.

ad_fuller_test = sm.tsa.stattools.adfuller(monthly_series, autolag = 'AIC')
ad_fuller_test

#------------------------------------------------------------------------------------------
# PART 4 - [ARIMA MODEL] IDENTIFY WHICH TIME SERIES MODEL (MA, AR, ARMA, and ARIMA) IS MOST SUITABLE

plot_acf(monthly_series)
plot_pacf(monthly_series, lags = 13)

# Prepare all 4 types of ARIMA models (MA, AR, ARMA, and ARIMA).

model_MA = sm.tsa.statespace.SARIMAX(monthly_series, order = (0, 0, 1))
model_AR = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 0, 0))
model_ARMA = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 0, 1))
model_ARIMA = sm.tsa.statespace.SARIMAX(monthly_series, order = (1, 1, 1))

# Fit all 4 types of ARIMA models (MA, AR, ARMA, and ARIMA).

result_MA = model_MA.fit()
result_AR = model_AR.fit()
result_ARMA = model_ARMA.fit()
result_ARIMA = model_ARIMA.fit()

# Perform Akaike Information Criterion (AIC) Analysis on all 4 types of ARIMA models (MA, AR, ARMA, and ARIMA).
# The ARIMA model has the lowest AIC value of 765.804. Therefore, it shall be used for later's analysis.

result_MA.aic
result_AR.aic
result_ARMA.aic
result_ARIMA.aic

# Run diagnostics for the ARIMA model.

result_ARIMA.plot_diagnostics(figsize = [20, 16])

#------------------------------------------------------------------------------------------
# PART 5 - [ARIMA MODEL] PERFORM GRID SEARCH TO IDENTIFY THE BEST POSSIBLE COMBINATION

# Set a pre-determined range of values for p, d, q, P, D, and Q.

p = d = q = P = D = Q = range(0, 3)
S = 12

# Create a variable to store all the possible combinations.

combinations = list(itertools.product(p, d, q, P, D, Q))
len(combinations)

# Identify all possible non-seasonal and seasonal portion orders.

arima_orders = [(x[0], x[1], x[2]) for x in combinations]
seasonal_orders = [(x[3], x[4], x[5], S) for x in combinations]

# Save the output of the models in a dataframe.

results_data = pd.DataFrame(columns = ['p', 'd', 'q', 'P', 'D', 'Q', 'AIC'])

# Create a function to automatically compute all the combination's AIC, and create an error handling mechanism.

for i in range(len(combinations)):
    try:
        model = sm.tsa.statespace.SARIMAX(monthly_series, order = arima_orders[i], seasonal_order = seasonal_orders[i])
        result= model.fit()
        results_data.loc[i,'p'] = arima_orders[i][0]
        results_data.loc[i,'d'] = arima_orders[i][1]
        results_data.loc[i,'q'] = arima_orders[i][2]
        results_data.loc[i,'P'] = seasonal_orders[i][0]
        results_data.loc[i,'D'] = seasonal_orders[i][1]
        results_data.loc[i,'Q'] = seasonal_orders[i][2]
        results_data.loc[i,'AIC'] = result.aic
    except:
        continue

# Identify the combinations with the lowest AIC.
# There is one best combination: 
#      p  d  q  P  D  Q  AIC
# 180  0  2  0  2  0  0  6.0

results_data[results_data.AIC == min(results_data.AIC)]

#------------------------------------------------------------------------------------------
# PART 6 - [ARIMA MODEL] RUN THE AMIRA MODEL TO PERFORM THE FORECASTING

# Use the best combination (no. 87) value, to create and fit the best forecasting model.

best_model = sm.tsa.statespace.SARIMAX(monthly_series, order = (0, 2, 0), seasonal_order = (2, 0, 0, 12))

results = best_model.fit()

# Define fitting model's timeframe (from when the monthly series begin), and identify its fitting value.

monthly_series
fitting = results.get_prediction(start = '2003-01-31')
fitting_mean = fitting.predicted_mean

# Define forecast model's extended prediction (12 months), and identify its forecast value.

forecast = results.get_forecast(steps = 12)
forecast_mean = forecast.predicted_mean

# Create a plot with fitting line, forecast line, and actual line.

fitting_mean.plot(label = 'fitting')
forecast_mean.plot(label = 'forecast')
monthly_series.plot(label = 'actual')
plt.legend(loc = 'upper left')

# Measure the accuracy of the model using the Mean Absolute Error.

mean_absolute_error = abs(monthly_series - fitting_mean).mean()

#------------------------------------------------------------------------------------------
# PART 7 - [EXPONENTIAL SMOOTHING MODEL] RUN THE EXPONENTIAL SMOOTHING MODEL TO PERFORM THE FORECASTING

# Prepare all 4 types of Holt Winter's Exponential Smoothing models.

model_expo1 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'add', seasonal = 'add', seasonal_periods = 12)
model_expo2 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'mul', seasonal = 'add', seasonal_periods = 12)
model_expo3 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'add', seasonal = 'mul', seasonal_periods = 12)
model_expo4 = sms.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend = 'mul', seasonal = 'mul', seasonal_periods = 12)

# Fit all 4 types of Holt Winter's Exponential Smoothing models.

results_1 = model_expo1.fit()
results_2 = model_expo2.fit()
results_3 = model_expo3.fit()
results_4 = model_expo4.fit()

fit1 = model_expo1.fit().predict(0, len(monthly_series))
fit2 = model_expo2.fit().predict(0, len(monthly_series))
fit3 = model_expo3.fit().predict(0, len(monthly_series))
fit4 = model_expo4.fit().predict(0, len(monthly_series))

# Measure the accuracy of the 4 exponential smoothing models. The lowest MAE is the best.

# mae1: 36216.80487215917 (lowest)
# mae2: 36207.94643409891
# mae3: 2730333.661587535
# mae4: 37960.96106850218

mae1 = abs(monthly_series - fit1).mean()
mae2 = abs(monthly_series - fit2).mean()
mae3 = abs(monthly_series - fit3).mean()
mae4 = abs(monthly_series - fit4).mean()

# Use the exponential smoothing model with the lowest MAE to perform the forecasting.

forecast = model_expo1.fit().predict(0, len(monthly_series) + 12)

# Create a plot with forecast line, and actual line.

monthly_series.plot(label = 'actual')
forecast.plot(label = 'forecast')
plt.legend(loc = 'upper left')

