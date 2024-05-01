# Covid-19 Timeseries Analysis/Forecasting

## Overview:

healthdata.gov provides daily data on covid-19 positive and negative cases since March 2020. The following data can be found here: https://healthdata.gov/dataset/COVID-19-Diagnostic-Laboratory-Testing-PCR-Testing/j8mb-icvb/about_data .
The scope of this project is to run a forecasting model and perform analysis on the following:

1) Overall forecasting on covid-19 since March 2020.
2) First Pfizer-BioNTech booster dose - August 12, 2021.
3) Second Pfizer-BioNTech booster dose - October 14, 2021.
4) HHS Secretary declaring end of covid-19 public health emergency - May 11, 2023.

The aforementioned dates in time will help us understand the variability in positive and negative cases on covid-19 cases.

## Initial Analysis

Below are the log transformations of the number of positive cases of covid-19 from 2020 to 2024

![image](https://github.com/MudassirAli94/covid19_timeseries_forecast/assets/38592433/69d238c5-045a-4d89-8eeb-57a1f5e19e7a)

There is a positive trend from 2020 to 2022 which is to be expected since that was at the peak of the pandemic, and then a rapid decline after 2022 which is likely because of the vaccines and society adapting to the virus.

However, for our forecasting we need to remove any and all trends and seasonality because it will affect the final forecasting. By looking at the graph we can see the trends but to confirm numerically we can run an Augmented Dickey-Fuller (ADF) test. ADF test is a method for testing the stationarity of a time series data, if there is a trend or seasonlity spike in the data then it is not stationary and our forecasting model will not be a reliable model.

![image](https://github.com/MudassirAli94/covid19_timeseries_forecast/assets/38592433/53eb42a1-d3ae-46ce-8596-1573336e9631)

After running an ADF test we can conclude that our data is NOT stationary and thus we need to perform data transformations to make it stationary.

## Non-Stationary to Stationary methods

To make time series data stationary, we can do the following:

- Differencing
- Decomposition

### Differencing

Differencing is the process of subtracting our values from our lag data. For example, if we have a lag of 1, we would subtract the value of the series at time t-1 (the previous point_ from the value at time t (the current point).

In this project differencing did NOT work.

### Decomposition

Decmoposition is the process of taking the seasonality, trend and residuals of the data and either adding or multiplying them. For example, 

- yt is the observed time series,
- Tt os the trend component
- St is the seasonal component
- Rt is the residual component

An additive model is:

yt = Tt + St + Rt

A multiplicative model is:

yt = Tt x St x Rt

Once the trend and seasonality is known we can remove them from the data.

For this project, decomposition produces an ADF p value of 0 thus we went with this method.

![image](https://github.com/MudassirAli94/covid19_timeseries_forecast/assets/38592433/0c8717c9-4ae4-47a9-b65e-3de952597511)

The graph on the right shows most of the data hovers around 0 which suggests that the trend c omponent has been removed. There are some volatility and spikes at the end but that can be due to the fact of vaccines and not a component of trend or seasonality. 

We can begin modeling for our forecasting.

## Modeling on data since March 2020
