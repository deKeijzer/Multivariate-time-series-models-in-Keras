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
from keras.layers import TimeDistributed, Input
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Conv1D, MaxPool2D, Flatten, Dropout, CuDNNGRU, CuDNNLSTM, Conv2D, MaxPooling2D, Reshape
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, Nadam
from time import time
from livelossplot import PlotLossesKeras
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe


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
    
    return X_train, y_train, X_test, y_test
    
def create_model(X_train, y_train, X_test, y_test):
    
    # CNN Model
    cnn = Sequential()
    
    ks1_first = {{choice([2, 3, 4, 5, 8, 10])}}
    ks1_second = {{choice([2, 3, 4, 5, 8, 10])}}
    
    ks2_first = {{choice([2, 3, 4, 5, 8, 10])}}
    ks2_second = {{choice([2, 3, 4, 5, 8, 10])}}
    
    cnn.add(Conv2D(filters=( {{choice([1, 2, 3, 4, 5, 8])}} ), 
                     kernel_size=(ks1_first, ks1_second),
                     padding='same',
                     kernel_initializer='TruncatedNormal'))
    cnn.add(BatchNormalization())
    cnn.add(LeakyReLU())
    cnn.add(Dropout( {{uniform(0, 1)}} ))
    
    for _ in range( {{choice([0, 1, 2, 3])}} ):
        cnn.add(Conv2D(filters=( {{choice([4, 8])}} ), 
                     kernel_size= (ks2_first, ks2_second), 
                         padding='same',
                     kernel_initializer='TruncatedNormal'))
        cnn.add(BatchNormalization())
        cnn.add(LeakyReLU())
        cnn.add(Dropout( {{uniform(0, 1)}} ))  
    
    cnn.add(Flatten())
    
    # RNN Model
    rnn = Sequential()
    rnn.add(CuDNNLSTM({{choice([1, 2, 3, 4, 5, 6, 7, 8])}}, return_sequences=True, kernel_initializer='TruncatedNormal'))
    rnn.add(BatchNormalization())
    rnn.add(LeakyReLU())
    rnn.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8])}}):
        rnn.add(CuDNNLSTM({{choice([4, 8])}}, kernel_initializer='TruncatedNormal', return_sequences=True))
        rnn.add(BatchNormalization())
        rnn.add(LeakyReLU())
        rnn.add(Dropout({{uniform(0, 1)}}))   
    
    rnn.add(CuDNNLSTM({{choice([1, 2, 3, 4, 5, 6, 7, 8])}}, kernel_initializer='TruncatedNormal', return_sequences=False))
    rnn.add(BatchNormalization())
    rnn.add(LeakyReLU())
    rnn.add(Dropout({{uniform(0, 1)}}))
    
    # DNN Model
    
    dnn = Sequential()
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16, 32])}}):
        dnn.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512])}}, kernel_initializer='TruncatedNormal'))
        dnn.add(BatchNormalization())
        dnn.add(LeakyReLU())
        dnn.add(Dropout({{uniform(0, 1)}}))
    
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16, 32])}}):
        dnn.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512])}}, kernel_initializer='TruncatedNormal'))
        dnn.add(BatchNormalization())
        dnn.add(LeakyReLU())
        dnn.add(Dropout({{uniform(0, 1)}}))
        
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16, 32])}}):
        dnn.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512])}}, kernel_initializer='TruncatedNormal'))
        dnn.add(BatchNormalization())
        dnn.add(LeakyReLU())
        dnn.add(Dropout({{uniform(0, 1)}}))
                 
    for _ in range({{choice([0, 1, 2, 3, 4, 8, 16, 32])}}):
        dnn.add(Dense({{choice([4, 8, 16, 32, 64, 128, 256, 512])}}, kernel_initializer='TruncatedNormal'))
        dnn.add(BatchNormalization())
        dnn.add(LeakyReLU())
        dnn.add(Dropout({{uniform(0, 1)}}))
  
    dnn.add(Dense({{choice([8, 16, 32, 64, 128, 256, 512, 1024])}}, kernel_initializer='TruncatedNormal'))
    dnn.add(BatchNormalization())
    dnn.add(LeakyReLU())
    dnn.add(Dropout({{uniform(0, 1)}}))
    
    dnn.add(Dense(1))
    
    # Putting it all together
    
    main_input = Input(shape=(X_train.shape[1], X_train.shape[2])) # Data has been reshaped to (800, 5, 120, 60, 1)
    reshaped_to_smaller_images = Reshape(target_shape=(24, 5, X_train.shape[2], 1))(main_input)

    model = TimeDistributed(cnn)(reshaped_to_smaller_images) # this should make the cnn 'run' 5 times?
    model = rnn(model) # combine timedistributed cnn with rnn
    model = dnn(model) # add dense
    
    # create the model, specify in and output
    model = Model(inputs=main_input, outputs=model)
    
    model.compile(loss='mse', metrics=['mape'],
                  optimizer='nadam')
    
    early_stopping_monitor = EarlyStopping(patience=15) # Not using earlystopping monitor for now, that's why patience is high
    
    bs = {{choice([32, 64, 128, 256])}}
    
    epoch_size = 1
    
    if bs == 32:
        epoch_size = 109
    elif bs == 64:
        epoch_size = 56
    elif bs == 128:
        epoch_size = 28
    elif bs == 256:
        epoch_size = 14
    
    #bs = 256
    #epoch_size = 14
    
    schedule = SGDRScheduler(min_lr={{uniform(1e-8, 1e-5)}}, #1e-5
                                     max_lr={{uniform(1e-3, 1e-1)}}, # 1e-2
                                     steps_per_epoch=np.ceil(epoch_size/bs),
                                     lr_decay=0.9,
                                     cycle_length=5, # 5
                                     mult_factor=1.5)

    result = model.fit(X_train, y_train,
              batch_size=bs,
              epochs=100,
              verbose=1,
              validation_split=0.2,
                       callbacks=[early_stopping_monitor, schedule])
    
    #get the highest validation accuracy of the training epochs
    
    hist = pd.DataFrame(result.history['val_loss']).fillna(9999) #replace nan with 9999
    val_loss = np.amin(hist.values)
    
    print('Best validation loss of epoch:', val_loss)
    
    
    K.clear_session() # Clear the tensorflow session (Free up RAM)
    
    return {'loss': val_loss, 'status': STATUS_OK} # Not returning model to save RAM


    
if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model, 
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials(),
                                          eval_space=True)
    
    X_train, y_train, X_test, y_test = data()
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