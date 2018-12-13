`Last updated: 13-12-2018`  
This repository is currently being created, it is not yet finished.  
If you have got any suggestions for the already finished notebooks, feel free to open an issue.

# Multivariate Time Series Models in Keras

# Introduction
This repository contains a throughout explanation on how to create different deep learning models in Keras for multivariate (tabular) time-series prediction. The data being used in this repository is from the [KB-74 Opschaler](https://github.com/deKeijzer/KB-74-OPSCHALER) project. The goal of this project is to do gas consumption prediction of houses on an hourly resolution, for the minor Applied Data Science at The Hague University of Applied Sciences.

# Jargon
The jargon used in this repository. 
- Dwelling: An individual house

# Data used
The data has a samplerate of one hour.  
Features: 
- Electrical power consumption (ePower)
- Wind speed (FF)
- Rain intensity (RG)
- Temperature (T)
- Timestamp YYYY:MM:DD HH:MM:SS (datetime)

Target:
- Gas consumption (gasPower)

# Models used
- Deep neural network (DNN)
- Recurrent neural networks: LSTM & GRU (RNN)
- Convolutional neural network (CNN)
- Timedistributed(CNN) -> RNN -> DNN
...

# About the notebooks
The notebooks are written in order.  
Due to this reason certain information that has been put in notebook 2 might for example not appear in notebook 4 and so on.  
