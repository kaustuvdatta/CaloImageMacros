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

def loadmodel(name, weights = False):
    json_file = open('%s.json'%name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model
    if weights==True:
        model.load_weights('%s.h5'%name)
    #print (model.summary())
    print("Loaded model from disk")
    return model

def savemodel(model,name="neural network"):

    model_name = name
    #model.summary()
    model.save_weights('%s_w.h5'%model_name, overwrite=True)
    model_json = model.to_json()
    with open("%s_m.json"%model_name, "w") as json_file:
        json_file.write(model_json)
        
def savelosses(hist, name="neural network"):    
    loss = np.array(hist.history['loss'])
    valoss = np.array(hist.history['val_loss'])
    f = h5py.File("%s_h.h5"%name,"w")
    f.create_dataset('loss',data=loss)
    f.create_dataset('val_loss',data=valoss)
    f.close()