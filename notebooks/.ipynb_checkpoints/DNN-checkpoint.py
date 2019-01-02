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

from keras.layers.normalization import BatchNormalization

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
    #val_size= 0.1 # The validation size of the train set

    # Split train & test
    #split_index_val = int(data.shape[0]*(train_size-val_size)) # the index at which to split df into train and test
    split_index_test = int(data.shape[0]*train_size) # the index at which to split df into train and test

    X_train = X[:split_index_test]
    #X_val = X[split_index_val:split_index_test]
    X_test = X[split_index_test:]

    y_train = y[:split_index_test]
    #y_val = y[split_index_val:split_index_test]
    y_test = y[split_index_test:]
    
    # Scaling the features
    scalerX = StandardScaler(with_mean=True, with_std=True).fit(X_train)

    X_train = scalerX.transform(X_train)
    #X_val = scalerX.transform(X_val)
    X_test = scalerX.transform(X_test)
    
    return X_train, y_train, X_test, y_test
    
def create_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), kernel_initializer='TruncatedNormal', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.083))
    
    # 1
    for _ in range(0):
        model.add(Dense(32, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.156))   
    
    # 2
    for _ in range(0):
        model.add(Dense(32, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.899))
    
    #3
    for _ in range(2):
        model.add(Dense(256, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.144))
    #4
    for _ in range(3):
        model.add(Dense(256, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.080))
    
    #5
    for _ in range(2):
        model.add(Dense(128, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.453))
    
    #6
    model.add(Dense(64, kernel_initializer='TruncatedNormal', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.556))
        
    model.add(Dense(1))
    
    model.compile(loss='mse', metrics=['mape'],
                  optimizer='nadam')
    
    early_stopping_monitor = EarlyStopping(patience=50000) # Not using earlystopping monitor for now, that's why patience is high
    bs = 2**13
    epoch_size = 1
    schedule = SGDRScheduler(min_lr=7.7e-6, #1e-5
                                     max_lr=2.9e-2, # 1e-2
                                     steps_per_epoch=np.ceil(epoch_size/bs),
                                     lr_decay=0.9,
                                     cycle_length=25, # 5
                                     mult_factor=1.5)
    
    checkpoint1 = ModelCheckpoint("models\\DNN.val_loss.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint("models\\DNN.val_mape.hdf5", monitor='val_mape', verbose=1, save_best_only=True, mode='min')

    checkpoint4 = ModelCheckpoint("models\\DNN.train_loss.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint5 = ModelCheckpoint("models\\DNN.train_mape.hdf5", monitor='mape', verbose=1, save_best_only=True, mode='min')

    result = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=35*10**3, # this should take ~ one hour
              verbose=2,
              validation_split=0.2,
                       callbacks=[schedule, checkpoint1, checkpoint2])
    
    pd.DataFrame(result.history).to_csv('models\\DNN_fit_history.csv')
    #get the highest validation accuracy of the training epochs
    validation_loss = np.amin(result.history['val_loss']) 
    print('validation loss of epoch:', validation_loss)
    return model


    
if __name__ == '__main__':
    X_train, y_train, X_test, y_test = data()
    
    """
    GTX 960m and GTX 970 support FP32
    """

    from keras import backend as K

    float_type ='float32' # Change this to float16 to use FP16
    K.set_floatx(float_type)
    K.set_epsilon(1e-4) #default is 1e-7

    X_train = X_train.astype(float_type)
    y_train = y_train.astype(float_type)
    X_test = X_test.astype(float_type)
    y_test = y_test.astype(float_type)
    
    model = create_model(X_train, y_train, X_test, y_test)
    
    #print("Evalutation of best performing model:")
    #print(model.evaluate(X_test, y_test))