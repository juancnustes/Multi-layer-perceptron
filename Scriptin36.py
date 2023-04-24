import wget
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, BatchNormalization, Flatten, Dropout
from keras.optimizers import Adam
from keras import Input
import h5py

#tf.keras.backend.set_floatx('float64')


data = loadmat('C:/Users/Juan Camilo/Desktop/Papers for tesis/Electric Car Batteries/Motor Control/MATLAB_MODEL/MotorDataSet.mat')
array_data = data['Motordata']

print('Original vector length: ',len(array_data[:,0]), 'samples')
len_i = 1440000
len_f = 2000000

# data shape:    [Wref Wmeas Idref Iqref Idmeas Iqmeas Vd Vq] 
Wref = array_data[len_i:len_f,0]
Wmeas = array_data[len_i:len_f,1]
Idref = array_data[len_i:len_f,2]
Iqref = array_data[len_i:len_f,3]
Idmeas = array_data[len_i:len_f,4]
Iqmeas = array_data[len_i:len_f,5]
Vd = array_data[len_i:len_f,6]
Vq = array_data[len_i:len_f,7]

# Error signals (input of the PID)
Speed_error = np.subtract(Wref, Wmeas) 

# Organize input and output of NN
in1 = [Speed_error]
in1 = np.asarray(in1)
in1 = np.transpose(in1)

out1 = [Iqref]
out1 = np.asarray(out1)
out1 = np.transpose(out1)    

x = in1
y = out1

print('Input shape:', x.shape)
print('Output shape:', y.shape)

from sklearn.model_selection import train_test_split

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle= True)

model = Sequential()

model.add(Dense(16, input_dim = 1, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1,  activation = 'linear'))

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.summary()

history = model.fit(x_train, y_train, validation_split= 0.2, epochs = 100)

score = model.evaluate(x_test, y_test)

model.save('PID_SPEED_init.h5')
