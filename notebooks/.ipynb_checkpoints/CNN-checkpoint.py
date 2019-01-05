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
from keras.layers import Dense, Conv1D, MaxPool2D, Flatten, Dropout, CuDNNLSTM, CuDNNGRU, Conv2D, MaxPooling2D
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
    df = pd.read_csv("F:\\Jupyterlab\\Multivariate-time-series-models-in-Keras\\data\\house_data_processed.csv", delimiter='\t', parse_dates=['datetime'])
    df = df.set_index(['datetime']) 

    magnitude = 1

    data = df.copy()
    
    columns_to_category = ['hour', 'dayofweek', 'season']
    data[columns_to_category] = data[columns_to_category].astype('category') 
    data = pd.get_dummies(data, columns=columns_to_category) 
    
    look_back = 5*24
    num_features = data.shape[1] - 1
    output_dim = 1
    train_size = 0.7

    X_train, y_train, X_test, y_test = df_to_cnn_rnn_format(df=data, train_size=train_size, look_back=look_back, target_column='gasPower', scale_X=True)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    
    return X_train, y_train, X_test, y_test
    
def create_model(X_train, y_train, X_test, y_test):
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    model = Sequential()
    
    ks1_first = 8
    ks1_second = 4
    
    ks2_first = 10
    ks2_second = 8
    
    model.add(Conv2D(filters=(5), 
                     kernel_size=(ks1_first, ks1_second),
                     input_shape=input_shape, 
                     padding='same',
                     kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.612))
    
    for _ in range(1):
        model.add(Conv2D(filters=(8), 
                     kernel_size= (ks2_first, ks2_second), 
                         padding='same',
                     kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.250))  
    
    model.add(Flatten())
    
    for _ in range(2):
        model.add(Dense(64 , kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.447))
    
    for _ in range(2):
        model.add(Dense(128 , kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.030))
  
    model.add(Dense(256 , kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.546))
        
    model.add(Dense(1))
    
    model.compile(loss='mse', metrics=['mape'],
                  optimizer='nadam')
    
    early_stopping_monitor = EarlyStopping(patience=50000) # Not using earlystopping monitor for now, that's why patience is high
    bs = 32
    epoch_size = 109
    schedule = SGDRScheduler(min_lr= 5.6e-6 ,
                                     max_lr= 1.9e-2 ,
                                     steps_per_epoch=np.ceil(epoch_size/bs),
                                     lr_decay=0.9,
                                     cycle_length=5, # 5
                                     mult_factor=1.5)
    
    checkpoint1 = ModelCheckpoint("models\\CNN.val_loss.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint("models\\CNN.val_mape.hdf5", monitor='val_mape', verbose=1, save_best_only=True, mode='min')

    checkpoint4 = ModelCheckpoint("models\\CNN.train_loss.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint5 = ModelCheckpoint("models\\CNN.train_mape.hdf5", monitor='mape', verbose=1, save_best_only=True, mode='min')

    result = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=8*10**3, # should take 24 hours
              verbose=1,
              validation_split=0.2,
                       callbacks=[schedule, checkpoint1, checkpoint2])
    
    pd.DataFrame(result.history).to_csv('models\\CNN_fit_history.csv')
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