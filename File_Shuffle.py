#Kaustuv Datta and Jayesh Mahapatra, July 2016

import numpy as np
import h5py
from io_functions import *

if __name__ == "__main__":
    import sys
    read(sys.argv[1])

'''

Given concatenated gamma and pion event files, opens two files in the dataset and 
uses the same random seed to shuffle image and target arrays in both files before 
returning two shuffled files. This goes on iteratively for as many file shuffling
operation is deemed to be required, mixing pion and gamma events each at different
energies, prior to data feeding for neural network training or inference. 

'''

def shuffle(N) 
#N=max number of times for shuffling operations to be carried out, recommended N >= 200
    for i in xrange(0,N): 
        
        #random seed is created  between 1 and 100, the range of file numbers
        perm=np.random.permutation(np.arange(1, 101))
        
        print (i, perm[0], perm[25]) #perm[0]and [25] are arbitrarily chosen elements WTLOG
        
        if perm[0]==perm[25]:
            perm=np.random.permutation(np.arange(1, 101))
        
        #open two files for shuffling events between
        fname0 = ('/data/shared/LCD/New_Data_Shuffled/GammaPi0_shuffled_%d.h5'%perm[0])
        fname1 = ('/data/shared/LCD/New_Data_Shuffled/GammaPi0_shuffled_%d.h5'%perm[25])
        
        f0=h5py.File(fname0,'r+')
        f1=h5py.File(fname1,'r+')
       
        #reads h5 event files

        np_ECAL0 = np.array(f0.get('ECAL'))
        np_HCAL0 = np.array(f0.get('HCAL'))
        np_target0 = np.array(f0.get('target'))
        print (np_ECAL0.shape, np_HCAL0.shape, np_target0.shape)    
        
        np_ECAL1 = np.array(f1.get('ECAL'))
        np_HCAL1 = np.array(f1.get('HCAL'))
        np_target1 = np.array(f1.get('target'))
        print (np_ECAL1.shape, np_HCAL1.shape, np_target1.shape)    
        
        f0.close()
        f1.close()
            
        np_ECAL = np.concatenate((np_ECAL0,np_ECAL1),axis=0)
        np_HCAL = np.concatenate((np_HCAL0,np_HCAL1),axis=0)
        np_target = np.concatenate((np_target0,np_target1),axis=0)
        
        rperm = np.random.permutation(np_ECAL.shape[0])
        
        lim = int(np.floor(np_ECAL.shape[0]/2.0))
        
        print (lim)
        
        #file shuffling is carried out with the same random seed on ECAL, HCAL image and target arrays

        np_ECAL = np_ECAL[rperm]
        np_HCAL = np_HCAL[rperm]
        np_target = np_target[rperm]
        
        npECAL0 = np_ECAL[:lim]
        npECAL1 = np_ECAL[lim:]
        npHCAL0 = np_HCAL[:lim]
        npHCAL1 = np_HCAL[lim:]
        nptarget0 = np_target[:lim]
        nptarget1 = np_target[lim:]  
            
        #two new shuffled files are overwritten on previously opened files
         
        with h5py.File(fname0,'w') as hf:
            hf.create_dataset('ECAL', data=npECAL0)
            hf.create_dataset('HCAL', data=npHCAL0)
            hf.create_dataset('target', data=nptarget0)
            hf.close()
        
        with h5py.File(fname1,'w') as hf:
            hf.create_dataset('ECAL', data=npECAL1)
            hf.create_dataset('HCAL', data=npHCAL1)
            hf.create_dataset('target', data=nptarget1)
            hf.close()