import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from tqdm import tqdm
import dask.dataframe as dd

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Conv1D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam, SGD, Nadam
from time import time
from livelossplot import PlotLossesKeras
from keras.layers.advanced_activations import LeakyReLU, PReLU
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from livelossplot import PlotLossesKeras
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe


from keijzer import *

# Setup (multi) GPU usage with scalable VRAM
num_gpu = setup_multi_gpus()


def data():
    # Loading the data
    df = pd.read_csv("F:\\Jupyterlab\\Multivariate-time-series-models-in-Keras\\data\\house_data_processed.csv", delimiter='\t', parse_dates=['datetime'])
    df = df.set_index(['datetime']) 

    magnitude = 1 # Take this from the 1. EDA & Feauture engineering notebook. It's the factor where gasPower has been scaled with to the power 10.
    
    # Preprocessing
    data = df.copy()
    
    columns_to_category = ['hour', 'dayofweek', 'season']
    data[columns_to_category] = data[columns_to_category].astype('category') # change datetypes to category
    
    # One hot encoding the dummy variables
    data = pd.get_dummies(data, columns=columns_to_category) # One hot encoding the categories
    
    # Create train and test set
    
    X = data.drop(['gasPower'], axis=1)
    #print('X columns: %s' % list(X.columns))

    y = data['gasPower']

    #X = np.array(X).reshape(-1,len(X.columns)) # Reshape to required dimensions for sklearn
    #y = np.array(y).reshape(-1,1)

    train_size = 0.7

    split_index = int(data.shape[0]*train_size) # the index at which to split df into train and test

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]
    
    # Scaling the features
    scalerX = StandardScaler(with_mean=True, with_std=True).fit(X_train)

    X_train = scalerX.transform(X_train)
    X_test = scalerX.transform(X_test)
    
    return X_train, y_train, X_test, y_test
    
def create_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense({{choice([8, 16, 32, 64, 128, 256, 512])}}, input_shape=(X_train.shape[1],), kernel_initializer='TruncatedNormal'))
    model.add(LeakyReLU())
    model.add(Dropout({{uniform(0, 1)}}))
    
    #model.add(Dense({{choice([8, 16, 32, 64, 125, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal'))
    #model.add(LeakyReLU())
    #model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    #model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    for _ in range({{choice([1, 2, 4, 8, 16])}}):
        model.add(Dense({{choice([8, 16, 32, 64, 128, 256, 512])}}, kernel_initializer='TruncatedNormal'))
        # We can also choose between complete sets of layers
        model.add(LeakyReLU())
        model.add(Dropout({{uniform(0, 1)}}))
        
    model.add(Dense({{choice([8, 16, 32, 64, 128, 256, 512])}}, kernel_initializer='TruncatedNormal'))
    model.add(LeakyReLU())
    model.add(Dropout({{uniform(0, 1)}}))
        
    model.add(Dense(1))

    adam = Adam(lr={{uniform(1e-5, 1e-1)}})
    nadam = Nadam(lr={{uniform(1e-5, 1e-1)}})
    
    
    model.compile(loss='mse', metrics=['mape'],
                  optimizer={{choice(['adam', 'nadam'])}})

    result = model.fit(X_train, y_train,
              batch_size={{choice([1000000000, 10000000000, 100000000000, 1000000000000])}},
              epochs=500,
              verbose=2,
              validation_split=0.3)
    
    #get the highest validation accuracy of the training epochs
    validation_loss = np.amin(result.history['val_loss']) 
    print('Best validation acc of epoch:', validation_loss)
    return {'loss': -validation_loss, 'status': STATUS_OK, 'model': model}


    
if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model, 
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)