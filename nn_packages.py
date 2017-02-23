#Kaustuv Datta, Niki Howe, Jayesh Mahapatra June 2016 
#one of several small classes for streamlining code

import numpy as np
import sys
import ast
import h5py
from sklearn.cross_validation import train_test_split
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, Convolution3D, Flatten, MaxPooling2D, MaxPooling3D, Merge
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as p
import os
import ROOT as rt
import root_numpy as rnp
from numpy import random
#matplotlib inline
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot

def draw_model(model, label):
    dot = plot(model, to_file='/home/kaustuv1993/Notebooks/Models_and_Weights/'+str(label)+'.png', show_shapes=True,
               show_layer_names=False)

def loadmodel(name, weights = False):
    json_file = open('%s.json'%name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights into new model
    if weights==True:
        model.load_weights('%s_w.h5'%name)
    print (model.summary())
    print("Loaded model from disk")
    return model

def savemodel(model,name="neural network"):

    model_name = name
    model.summary()
    model.save_weights('%s_w.h5'%model_name, overwrite=True)
    model_json = model.to_json()
    with open("%s.json"%model_name, "w") as json_file:
        json_file.write(model_json)
        
def savelosses(hist, name="neural network"):    
    loss = np.array(hist.history['loss'])
    valoss = np.array(hist.history['val_loss'])
    
    f = h5py.File("%s_h.h5"%name,"w")
    f.create_dataset('loss',data=loss)
    f.create_dataset('val_loss',data=valoss)
    f.close()
        
def savelosses_regcls(hist, name="neural network"):    
    loss = np.array(hist.history['loss'])
    valoss = np.array(hist.history['val_loss'])
    enloss = np.array(hist.history['energy_loss'])
    envaloss = np.array(hist.history['val_energy_loss'])
    particleloss = np.array(hist.history['particle label_loss'])
    particlevaloss = np.array(hist.history['val_particle label_loss'])

    f = h5py.File("%s_h.h5"%name,"w")
    f.create_dataset('loss',data=loss)
    f.create_dataset('val_loss',data=valoss)
    f.create_dataset('energy_loss',data=enloss)
    f.create_dataset('val_energy_loss',data=envaloss)
    f.create_dataset('plabel_loss',data=particleloss)
    f.create_dataset('val_plabelloss',data=particlevaloss)
    f.close()

def training_error(loss, valoss, fname):
    plt.figure(figsize=(10,10))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    colors=[]
    do_acc=False
    color = tuple(np.random.random(3))
    colors.append(color)
    plt.plot(loss, label='loss', color=color)
    plt.plot(valoss, lw=2, ls='dashed', label='val_loss', color=color)
    plt.legend()
    plt.yscale('log')
    plt.savefig('%s.pdf'%fname)
    plt.show()
    if not do_acc: 
	return
        
def show_losses( histories,fname ):
    plt.figure(figsize=(10,10))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    
    colors=[]
    do_acc=False
    for label,loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label+" validation"
        if 'acc' in loss.history:
            l+=' (acc %2.4f)'% (loss.history['acc'][-1])
            do_acc = True
        if 'val_acc' in loss.history:
            vl+=' (acc %2.4f)'% (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)
            
            
def show_losses_regcls( histories,fname ):
    plt.figure(figsize=(10,10))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    
    colors=[]
    do_acc=False
    for label,loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label+" validation"
        if 'acc' in loss.history:
            l+=' (acc %2.4f)'% (loss.history['acc'][-1])
            do_acc = True
        if 'val_acc' in loss.history:
            vl+=' (acc %2.4f)'% (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)            


    plt.legend()
    plt.yscale('log')
    plt.savefig('%s.pdf'%fname)
    plt.savefig('%s.png'%fname)
    plt.show()
    if not do_acc: 
	return

def convert(filename):
    f = h5py.File(filename,'r')
    HCAL = np.array(f['HCAL'])
    ECAL = np.array(f['ECAL'])
    target=np.array(f['target'])
    f.close()
    np.savez_compressed(filename.replace('.h5','.npz'), HCAL=HCAL,ECAL=ECAL,target=target)
    #plt.figure(figsize=(10,10))
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #for i,(label,loss) in enumerate(histories):
    #    color = colors[i]
    #    if 'acc' in loss.history:
    #        plt.plot(loss.history['acc'], lw=2, label=label+" accuracy", color=color)
    #    if 'val_acc' in loss.history:
    #        plt.plot(loss.history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
    #plt.legend(loc='lower right')
    #plt.savefig('%s.png'%fname)
   
    #plt.show()
