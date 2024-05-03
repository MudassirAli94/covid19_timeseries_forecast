import pandas as pd
pd.set_option('display.max_columns', None)
import json
from gcp_functions import read_csv_from_gcs
import matplotlib.pyplot as plt

import statsmodels.api as sm
import numpy as np

## get GCP configurations
with open('config.json') as config_file:
    config = json.load(config_file)

YOUR_BUCKET_NAME = config["bucket_name"]
PROJECT_ID = config["project_id"]

## begin data cleaning and transformation
c19_raw = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name='2024-04-29/covid_19_timeseries_raw_20240429150748.csv')
c19_raw["total_results_reported"] = c19_raw["total_results_reported"].apply(lambda i: np.log(i+1))

positive_df = c19_raw[c19_raw["overall_outcome"] == "Positive"][["date","total_results_reported"]]
positive_df["date"] = pd.to_datetime(positive_df["date"])
positive_df = positive_df.groupby("date").sum().reset_index()
positive_df = positive_df.rename(columns = {"total_results_reported":"positive_cases"})

## get yearly
positive_yearly_df = positive_df.copy()
positive_yearly_df["year"] = positive_yearly_df["date"].dt.year
positive_yearly_df["year"] = positive_yearly_df["year"].astype(int)
positive_yearly_df = positive_yearly_df[["year","positive_cases"]]
positive_yearly_df = positive_yearly_df.groupby(["year"]).sum().reset_index()


## graph the data

plt.plot(positive_yearly_df["year"], positive_yearly_df["positive_cases"])
plt.title("Positive Cases")
plt.xlabel("Year")
plt.ylabel("# of Cases")
plt.xticks(positive_yearly_df["year"].unique(), labels=[int(year) for year in positive_yearly_df["year"].unique()])
plt.show()



## negative cases

negative_df = c19_raw[c19_raw["overall_outcome"] == "Negative"][["date","total_results_reported"]]
negative_df["date"] = pd.to_datetime(negative_df["date"])
negative_df = negative_df.groupby("date").sum().reset_index()
negative_df = negative_df.rename(columns = {"total_results_reported":"negative_cases"})

## get yearly

negative_yearly_df = negative_df.copy()
negative_yearly_df["year"] = negative_yearly_df["date"].dt.year
negative_yearly_df["year"] = negative_yearly_df["year"].astype(int)
negative_yearly_df = negative_yearly_df[["year","negative_cases"]]
negative_yearly_df = negative_yearly_df.groupby(["year"]).sum().reset_index()

plt.plot(negative_yearly_df["year"], negative_yearly_df["negative_cases"])
plt.title("Negative Cases")
plt.xlabel("Year")
plt.ylabel("# of Cases")
plt.xticks(negative_yearly_df["year"].unique(), labels=[int(year) for year in negative_yearly_df["year"].unique()])
plt.show()

## run a regression model on positive cases to see if there is a trend with residuals
positive_df['date_ordinal'] = pd.to_datetime(positive_df['date']).apply(lambda date: date.toordinal())

# Set up the regression model
X = sm.add_constant(positive_df['date_ordinal'])  # adding a constant
y = positive_df['positive_cases']

model = sm.OLS(y, X).fit()  # Fit the model

### Step 3: Plotting the Residuals
residuals = model.resid

# Plotting the residuals
plt.figure(figsize=(10, 5))
plt.scatter(positive_df['date'], residuals, color='blue')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals from Linear Regression of positive Cases')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()

# Print out the summary of the regression
print(model.summary())

## detrend the data with differencing
positive_df['positive_cases_diff'] = positive_df['positive_cases'].diff()
positive_df = positive_df.dropna()

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(positive_df['date'], positive_df['positive_cases'], label='Original')
plt.title('Original Positive Cases')
plt.xlabel('Date')
plt.ylabel('Positive Cases')

plt.subplot(1, 2, 2)
plt.plot(positive_df['date'], positive_df['positive_cases_diff'], label='Differenced', color='orange')
plt.title('Differenced Positive Cases')
plt.xlabel('Date')
plt.ylabel('Differences in Cases')

plt.tight_layout()
plt.show()

plt.clf()
plt.close('all')

from statsmodels.tsa.stattools import adfuller

## my dataframe to series

result = adfuller(positive_df["positive_cases_diff"], autolag='AIC')  # Automatically select the lag length based on information criterion

# Print the results
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# Apply a second differencing to the data
positive_df['positive_cases_diff2'] = positive_df['positive_cases_diff'].diff().dropna()
positive_df.dropna(inplace=True)

import matplotlib.pyplot as plt

# Plot the original, first differenced, and possibly second differenced series
plt.figure(figsize=(12, 9))
plt.subplot(311)
plt.plot(positive_df['positive_cases'], label='Original')
plt.title('Original Series')
plt.legend()

plt.subplot(312)
plt.plot(positive_df['positive_cases_diff'], label='1st Differencing')
plt.title('1st Differenced Series')
plt.legend()

plt.subplot(313)
plt.plot(positive_df['positive_cases_diff2'], label='2nd Differencing')
plt.title('2nd Differenced Series')
plt.legend()

plt.tight_layout()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

# Perform seasonal decomposition
decomposition = seasonal_decompose(positive_df['positive_cases'], model='additive', period=12)  # change period according to your data

# Access the detrended component
detrended = decomposition.resid

# Plot and analyze
detrended.dropna(inplace=True)  # Drop NA values that result from decomposition
result = adfuller(detrended, autolag='AIC')

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


positive_df.dropna(inplace=True)

positive_df['detrended'] = decomposition.resid

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

# Plot original data
axes[0].plot(positive_df['date'], positive_df['positive_cases'], label='Original')
axes[0].set_title('Original Positive Cases')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Positive Cases')
axes[0].legend()

# Plot detrended data
axes[1].plot(positive_df['date'], positive_df['detrended'], label='Detrended', color='orange')
axes[1].set_title('Detrended Positive Cases')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Detrended Cases')
axes[1].legend()

plt.tight_layout()
plt.show()

## Now begin ARIMA model

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(15, 4))
plot_acf(positive_df['detrended'].dropna(), ax=axes[0])
plot_pacf(positive_df['detrended'].dropna(), ax=axes[1])
plt.show()


model = ARIMA(positive_df['positive_cases'], order=(1, 1, 1))
results = model.fit()
print(results.summary())
results.plot_diagnostics(figsize=(12, 8))
plt.show()

## calculate mean absolute error

from sklearn.metrics import mean_absolute_error

# Forecast the next 10 steps
forecast = results.forecast(steps=10)
print(forecast)



