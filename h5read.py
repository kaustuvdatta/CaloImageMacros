import numpy as np
import h5py

if __name__ == "__main__":
    import sys
    read(sys.argv[1])

def read(filename):
    f = h5py.File(filename,'r')
    train_HCAL = np.array(f['HCAL'])
    train_ECAL = np.array(f['ECAL'])
    train_target=np.array(f['target'])
    print(train_HCAL.shape,train_ECAL.shape,train_target.shape)