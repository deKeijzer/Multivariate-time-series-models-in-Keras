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
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Conv1D, MaxPool2D, Flatten, Dropout, GRU, CuDNNLSTM, CuDNNGRU
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, Nadam
from time import time
from livelossplot import PlotLossesKeras
from keras.layers.advanced_activations import LeakyReLU, PReLU
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib
from sklearn.preprocessing import StandardScaler

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
    
    return X_train, y_train, X_test, y_test, look_back, num_features
    
def create_model(X_train, y_train, X_test, y_test, look_back, num_features):
    model = Sequential()
    model.add(CuDNNGRU(16, input_shape=(look_back, num_features), return_sequences=True, kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.031))
    #1
    for _ in range(0):
        model.add(CuDNNGRU(16, kernel_initializer='TruncatedNormal', return_sequences=True))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.358))   
    #2
    for _ in range(1):
        model.add(CuDNNGRU(8, kernel_initializer='TruncatedNormal', return_sequences=True))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.573))
    #3
    for _ in range(0):
        model.add(CuDNNGRU(8, kernel_initializer='TruncatedNormal', return_sequences=True))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.827))
    #4
    model.add(CuDNNGRU(4, kernel_initializer='TruncatedNormal', return_sequences=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.183))
    
    #5
    for _ in range(0):
        model.add(Dense(512, kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.476))
    #6
    for _ in range(4):
        model.add(Dense(8, kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.041))
    #7
    model.add(Dense(8, kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.688))
        
    model.add(Dense(1))
    
    model.compile(loss='mse', metrics=['mape'],
                  optimizer='nadam')
    
    early_stopping_monitor = EarlyStopping(patience=50000) # Not using earlystopping monitor for now, that's why patience is high
    bs = 64
    epoch_size = 55
    schedule = SGDRScheduler(min_lr=7.8e-6, #1e-5
                                     max_lr=7e-3, # 1e-2
                                     steps_per_epoch=np.ceil(epoch_size/bs),
                                     lr_decay=0.9,
                                     cycle_length=5, # 5
                                     mult_factor=1.5)
    
    checkpoint1 = ModelCheckpoint("models\\GRU.val_loss.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint("models\\GRU.val_mape.hdf5", monitor='val_mape', verbose=1, save_best_only=True, mode='min')

    checkpoint4 = ModelCheckpoint("models\\GRU.train_loss.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoint5 = ModelCheckpoint("models\\GRU.train_mape.hdf5", monitor='mape', verbose=1, save_best_only=True, mode='min')

    result = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=4*10**3,
              verbose=1,
              validation_split=0.2,
                       callbacks=[schedule, checkpoint1, checkpoint2])
    
    pd.DataFrame(result.history).to_csv('models\\GRU_fit_history.csv')
    #get the highest validation accuracy of the training epochs
    validation_loss = np.amin(result.history['val_loss']) 
    print('validation loss of epoch:', validation_loss)
    return model


    
if __name__ == '__main__':
    X_train, y_train, X_test, y_test, look_back, num_features = data()
    
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
    
    model = create_model(X_train, y_train, X_test, y_test, look_back, num_features)
    
    #print("Evalutation of best performing model:")
    #print(model.evaluate(X_test, y_test))