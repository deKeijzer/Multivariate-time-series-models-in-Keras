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
from keras.layers.normalization import BatchNormalization
from time import time
from livelossplot import PlotLossesKeras
from keras.layers.advanced_activations import LeakyReLU, PReLU
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras import backend as K

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

    train_size = 0.9
    val_size= 0.1 # The validation size of the train set

    # Split train & test
    split_index_val = int(data.shape[0]*(train_size-val_size)) # the index at which to split df into train and test
    split_index_test = int(data.shape[0]*train_size) # the index at which to split df into train and test

    X_train = X[:split_index_val]
    X_val = X[split_index_val:split_index_test]
    X_test = X[split_index_test:]

    y_train = y[:split_index_val]
    y_val = y[split_index_val:split_index_test]
    y_test = y[split_index_test:]
    
    # Scaling the features
    scalerX = StandardScaler(with_mean=True, with_std=True).fit(X_train)

    X_train = scalerX.transform(X_train)
    X_val = scalerX.transform(X_val)
    X_test = scalerX.transform(X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    
    
def create_model(X_train, y_train, X_test, y_test):
    
    def hyper_activation():
        activation = {{choice(['relu', 'leakyrelu', 'sigmoid'])}}
        if  activation == 'leakyrelu':
            model.add(LeakyReLU())
        elif activation == 'relu':
            model.add(Activation('relu'))
        else:
            model.add(Activation('sigmoid'))
    
    model = Sequential()
    model.add(Dense({{choice([32, 64, 128, 256])}}, input_shape=(X_train.shape[1],), kernel_initializer='TruncatedNormal', use_bias=False))
    model.add(BatchNormalization())
    hyper_activation()
    model.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        hyper_activation()
        model.add(Dropout({{uniform(0, 1)}}))   
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        hyper_activation()
        model.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        hyper_activation()
        model.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        hyper_activation()
        model.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        hyper_activation()
        model.add(Dropout({{uniform(0, 1)}}))
  
    model.add(Dense({{choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048])}}, kernel_initializer='TruncatedNormal', use_bias=False))
    model.add(BatchNormalization())
    hyper_activation()
    model.add(Dropout({{uniform(0, 1)}}))
        
    model.add(Dense(1))

    #adam = Adam(lr={{uniform(1e-5, 1e-1)}})
    #nadam = Nadam(lr={{uniform(1e-5, 1e-1)}})
    
    
    model.compile(loss='mse', metrics=['mape'],
                  optimizer={{choice(['adadelta', 'adagrad', 'adam', 'nadam'])}})
    
    early_stopping_monitor = EarlyStopping(patience=100) # Not using earlystopping monitor for now, that's why patience is high
    bs = 2**13
    epoch_size = 1
    schedule = SGDRScheduler(min_lr={{uniform(1e-8, 1e-5)}}, #1e-5
                                     max_lr={{uniform(1e-3, 1e-1)}}, # 1e-2
                                     steps_per_epoch=np.ceil(epoch_size/bs),
                                     lr_decay=0.9,
                                     cycle_length=5, # 5
                                     mult_factor=1.5)

    result = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=1000,
              verbose=2,
              validation_split=0.2,
                       callbacks=[early_stopping_monitor, schedule])
    
    #get the highest validation accuracy of the training epochs
    val_loss = np.amin(result.history['val_loss'])
    print('Best validation loss of epoch:', val_loss)
    
    
    K.clear_session() # Clear the tensorflow session (Free up RAM)
    
    return {'loss': val_loss, 'status': STATUS_OK} # Not returning model to save RAM


    
if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model, 
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1000,
                                          trials=Trials(),
                                          eval_space=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test = data()
    print("Evalutation of best performing model:")
    #print(best_model.evaluate(X_val, y_val))
    
    print("Crossvalidation of best performing model:")
    #print(best_model.evaluate(X_test, y_test))
    
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    
    #best_model.save('models//DNN.h5')
    
    #from hyperas.utils import eval_hyperopt_space
    #real_param_values = eval_hyperopt_space(space, best_run)
    #print('----------')
    #print(real_param_values)