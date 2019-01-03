
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/deKeijzer/Multivariate-time-series-models-in-Keras/master)  
Click the above button to launch this repository as a notebook in your browser.  
`Last updated of the README: 03-01-2019`  
This repository is currently being created, it is not yet finished.  
The notebooks in the repository look the best when using Jupyter.

# Multivariate Time Series Models in Keras

# Introduction
This repository contains a throughout explanation on how to create different deep learning models in Keras for multivariate (tabular) time-series prediction. The data being used in this repository is from the [KB-74 OPSCHALER](https://github.com/deKeijzer/KB-74-OPSCHALER) project. The goal of this project is to do gas consumption prediction of houses on an hourly resolution, for the minor Applied Data Science at The Hague University of Applied Sciences.

# Jargon
The jargon used in this repository. 
- Dwelling: An individual house
- EDA: Exploratory Data Annalysis  
- MVLR: Multivariate Linear Regression
- DNN: Deep Neural Network
- CNN: Convolutional Neural Network
- RNN: Recurrent Neural Network
- LSTM: Long Short-Term Memory
- GRU: Gated Recurrent Unit

# The data

## Available data
The original data is as follows.  

### Smart meter data
<table class="tableizer-table">
<thead><tr class="tableizer-firstrow"><th>Parameter</th><th>Unit</th><th>Sample rate</th><th>Description</th></tr></thead><tbody>
 <tr><td>Timestamp</td><td>-</td><td>10 s</td><td>Timestamp of data telegram (set by smart meter) in local time </td></tr>
 <tr><td>eMeter</td><td>kWh</td><td>10 s</td><td>Meter reading electricity delivered to client, normal tariff </td></tr>
 <tr><td>eMeterReturn</td><td>kWh</td><td>10 s</td><td>Meter reading electricity delivered by client, normal tariff </td></tr>
 <tr><td>eMeterLow</td><td>kWh</td><td>10 s</td><td>Meter reading electricity delivered to client, low tariff </td></tr>
 <tr><td>eMeterLowReturn</td><td>kWh</td><td>10 s</td><td>Meter reading electricity delivered by client, low tariff </td></tr>
 <tr><td>ePower</td><td>kWh</td><td>10 s</td><td>Actual electricity power delivered to client </td></tr>
 <tr><td>ePowerReturn</td><td>kWh</td><td>10 s</td><td>Actual electricity power delivered by client </td></tr>
 <tr><td>gasTimestamp</td><td>-</td><td>1 h</td><td>Timestamp of the gasMeter reading (set by smart meter) in local time </td></tr>
 <tr><td>gasMeter</td><td>m3</td><td>1 h</td><td>Last hourly value (temperature converted0, gas delivered to client </td></tr>
</tbody></table>

### Weather data
This is weather data from the KNMI weather station in Rotterdam with a sample rate of 15 minutes.  
A representative from OPSCHALER says that this weather station is the most nearby most of the dwellings, the exact dwelling locations however are unknown.  
The dwelling furthest away from the weather station is 103 km north east.  
When this weather data is used to make predictions on the validation and test dataset (which is future data for the model), this weather data is assumed to be the 'predictions' for the weather at given timestamp.   
In reality the weather predictions made by climate models should be used.

<table class="tableizer-table">
<thead><tr class="tableizer-firstrow"><th>Parameter</th><th>Unit</th><th>Description</th></tr></thead><tbody>
 <tr><td>DD</td><td>degrees</td><td>Wind direction</td></tr>
 <tr><td>DR</td><td>s</td><td>Precipitation time</td></tr>
 <tr><td>FX</td><td>m/s</td><td>Maximum gust of wind at 10 m</td></tr>
 <tr><td>FF</td><td>m/s</td><td>Windspeed at 10 m</td></tr>
 <tr><td>N</td><td>okta</td><td>Cloud coverage</td></tr>
 <tr><td>P</td><td>hPa</td><td>Outside pressure</td></tr>
 <tr><td>Q</td><td>W/m2</td><td>Global radiation</td></tr>
 <tr><td>RG</td><td>mm/h</td><td>Rain intensity</td></tr>
 <tr><td>SQ</td><td>m</td><td> Sunshine duration (in minutes)</td></tr>
 <tr><td>T</td><td>deg C</td><td>Temperature at 1,5 m (1 minute mean)</td></tr>
 <tr><td>T10</td><td>deg C</td><td>Minimum temperature at 10 cm</td></tr>
 <tr><td>TD</td><td>deg C</td><td>Dew point temperature</td></tr>
 <tr><td>U</td><td>%</td><td>Relative humidity at 1,5 m</td></tr>
 <tr><td>VV</td><td>m</td><td>Horizontal sight</td></tr>
 <tr><td>WW</td><td>-</td><td>Weather- and station-code</td></tr>
</tbody></table>

## Used data
The original data has been resampled to an hour, this is the data available in this repository.  

Features: 
- Electrical power consumption (ePower)
- Wind speed (FF)
- Rain intensity (RG)
- Temperature (T)
- Timestamp YYYY:MM:DD HH:MM:SS (datetime)

Target:
- Gas consumption (gasPower)

# About the notebooks
The notebooks are written in order.  
Due to this reason certain information that has been put in notebook 1 might for example not appear in notebook 2 and so on.  
The `hyperas MODEL.py` files contains the Python scripts that use [Hyperas](https://github.com/maxpumperla/hyperas) for the hyperparameter optimazation.  
The `MODEL.py` files contains the `.py` versions of the notebooks, these train  quicker than training from within Jupyter (e.g. 50 epochs/s instead of 2 epochs/s, for DNN).  
  
## Notebooks in order

- [1. EDA & Feature engineering](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/1.%20EDA%20%26%20Feature%20engineering.ipynb)  
- [2. MVLR](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/2.%20MVLR%20.ipynb)  
- [3. DNN](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/3.%20DNN.ipynb)  
- [4.1 CNN & RNN Data Preperation](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/4.1%20CNN%20%26%20RNN%20Data%20Preperation.ipynb)  
- [4.2 LSTM](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/4.2%20LSTM.ipynb)  
- [4.3 GRU](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/4.3%20GRU.ipynb)  
- [5.1 Multivariate time-series to images](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/5.1%20Multivariate%20time-series%20to%20images.ipynb)  
- [5.2 CNN](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/5.2%20CNN.ipynb)   
- [6. TimeDistributed(CNN)+RNN+DNN](https://github.com/deKeijzer/Multivariate-time-series-models-in-Keras/blob/master/notebooks/6.%20TimeDistributed(CNN)%2BRNN%2BDNN.ipynb)  

