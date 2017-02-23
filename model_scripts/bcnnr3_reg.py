import os
import sys
import re
import glob
import h5py
import numpy as np
#import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Activation,Input, Dense, Dropout, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge
sys.path.append('/scratch/daint/jmahapat/scripts/')
from Generators import *
from callback_mods import *
from nn_packages import *


def train_model():
    ds = RegGen(1000)
    vs = RegGen(1000)
    filename='/scratch/daint/jmahapat/rRegression/bcnnr3_reg'
    model = loadmodel(filename)
    model.compile(loss=['mse'],optimizer='sgd')
    check = Checkpoint_Reg(filepath=filename, verbose=0)
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    print('%s has started training'%filename)
    hist = model.fit_generator(ds.train(modeltype=3), samples_per_epoch=50000, nb_epoch=1000, validation_data= vs.validation(modeltype=3), nb_val_samples=50000, verbose=0, callbacks=[check,early])
    savelosses(hist,name=filename)
    savemodel(model,name=filename)
    print('%s has finished training'%filename)

if __name__ == "__main__":
    train_model()