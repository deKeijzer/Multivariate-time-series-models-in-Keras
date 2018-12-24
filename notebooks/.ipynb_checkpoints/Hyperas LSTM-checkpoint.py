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
from keras.layers import Dense, Conv1D, MaxPool2D, Flatten, Dropout, CuDNNLSTM
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
    
    look_back = 5*24 # D -> 5, H -> 5*24
    num_features = data.shape[1] - 1
    output_dim = 1
    train_size = 0.7

    X_train, y_train, X_test, y_test = df_to_cnn_rnn_format(df=data, train_size=train_size, look_back=look_back, target_column='gasPower', scale_X=True)

    val_size= 0.2 # The validation size of the train set

    # Split train & test
    split_index_val = int(data.shape[0]*(train_size-val_size)) # the index at which to split df into train and test
    split_index_test = int(data.shape[0]*train_size) # the index at which to split df into train and test
    
    X_val = X_train[split_index_val:] # TODO: only fit scaler on the train data
    X_train = X_train[:split_index_val]

    y_val = y_train[split_index_val:]
    y_train = y_train[:split_index_val]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
    
def create_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(CuDNNLSTM({{choice([4, 8, 16, 32])}}, input_shape=(look_back, num_features), return_sequences=True, kernel_initializer='TruncatedNormal'))
    model.add(LeakyReLU())
    model.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(CuDNNLSTM({{choice([4, 8, 16, 32])}}, kernel_initializer='TruncatedNormal', return_sequences=True))
        model.add(LeakyReLU())
        model.add(Dropout({{uniform(0, 1)}}))   
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(CuDNNLSTM({{choice([4, 8, 16, 32])}}, kernel_initializer='TruncatedNormal', return_sequences=True))
        model.add(LeakyReLU())
        model.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(CuDNNLSTM({{choice([4, 8, 16, 32])}}, kernel_initializer='TruncatedNormal', return_sequences=True))
        model.add(LeakyReLU())
        model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(CuDNNLSTM({{choice([4, 8, 16, 32])}}, kernel_initializer='TruncatedNormal', return_sequences=False))
    model.add(LeakyReLU())
    model.add(Dropout({{uniform(0, 1)}}))
    
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal'))
        model.add(LeakyReLU())
        model.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16])}}):
        model.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal'))
        model.add(LeakyReLU())
        model.add(Dropout({{uniform(0, 1)}}))
  
    model.add(Dense({{choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048])}}, kernel_initializer='TruncatedNormal'))
    model.add(LeakyReLU())
    model.add(Dropout({{uniform(0, 1)}}))
        
    model.add(Dense(1))

    adam = Adam(lr={{uniform(1e-5, 1e-1)}})
    nadam = Nadam(lr={{uniform(1e-5, 1e-1)}})
    
    
    model.compile(loss='mse', metrics=['mape'],
                  optimizer={{choice(['nadam', 'adam'])}})
    
    early_stopping_monitor = EarlyStopping(patience=50) # Not using earlystopping monitor for now, that's why patience is high
    bs = 1024
    epoch_size = 3
    schedule = SGDRScheduler(min_lr=1e-5, #1e-5
                                     max_lr={{choice([1, 0.1, 0.3, 0.6, 0.01, 0.03, 0.06])}}, # 1e-2
                                     steps_per_epoch=np.ceil(epoch_size/bs),
                                     lr_decay=0.9,
                                     cycle_length=5, # 5
                                     mult_factor=1.5)

    result = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=500,
              verbose=2,
              validation_split=0.1,
                       callbacks=[early_stopping_monitor, schedule])
    
    #get the highest validation accuracy of the training epochs
    validation_loss = np.amin(result.history['val_loss']) 
    print('Best validation loss of epoch:', validation_loss)
    return {'loss': -validation_loss, 'status': STATUS_OK, 'model': model}


    
if __name__ == '__main__':
    best_run, best_model, space = optim.minimize(model=create_model, 
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials(),
                                          eval_space=True,
                                          return_space=True)
    
    X_train, y_train, X_val, y_val, X_test, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_val, y_val))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    
    #from hyperas.utils import eval_hyperopt_space
    #real_param_values = eval_hyperopt_space(space, best_run)
    #print('----------')
    #print(real_param_values)