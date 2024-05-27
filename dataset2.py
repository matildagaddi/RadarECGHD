import os
import os.path
import torch
from torch.utils import data
#import pandas as pd
import numpy as np

import scipy
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd
from sklearn.preprocessing import StandardScaler

# radar_path = "/Users/matildagaddi/Documents/SEElab/DATASET/trainVal/radar"
# ecg_path = "/Users/matildagaddi/Documents/SEElab/DATASET/trainVal/ecg"

class MyDataset(data.Dataset):
    def __init__(self, radar_path, ecg_path, window_size, device):
        self.radar_path = radar_path
        self.ecg_path = ecg_path
        self.window_size = int(window_size)
        self.window_samples = int(window_samples)
        self.device = device
        self.load_data()

    def __len__(self): #this method indicates how many times you have to loop when iterating over the dataset (line 58)
        return self.data.size(0) 

    def __getitem__(self, index): #given an index (from 0, to __len__ output), return sample and label
        sample = self.data[index] 
        target = self.target[index]
        return sample, target

    def load_data(self):
        myData = None
        myTarget = None
        #for filename in os.listdir(self.radar_path): #just do one file at a time
        #f = os.path.join(self.radar_path, filename)
        fr = self.radar_path
        if os.path.isfile(fr):
            mat = loadmat(fr)
            if(len(mat["radar_l"][0]) != 1024):
                print("WRONG: ", len(mat["radar_l"][0]) )
                exit()

            #if myData is None:
            myData = mat["radar_l"][0]
            # else:
            #     myData = np.concatenate((myData, mat["radar_l"][0]))

        #for filename in os.listdir(self.ecg_path):
        #f = os.path.join(self.ecg_path, filename)
        fe = self.ecg_path
        if os.path.isfile(fe):
            mat = loadmat(fe)
            if(len(mat["ecg_l"][0]) != 1024):
                print("WRONG: ", len(mat["ecg_l"][0]) )
                exit()

            # if myTarget is None:
            #     #print(mat["ecg_l"][0])
            myTarget = mat["ecg_l"][0]
            # else:
            #     # print(mat["ecg_l"][0])
            #     # print(myTarget)
            #     myTarget = np.concatenate((myTarget, mat["ecg_l"][0]))



        myData = torch.FloatTensor( myData )
        myTarget = torch.FloatTensor( myTarget )

        self.data = MyData
        self.Target = myTarget
        
       #  # myData = torch.div( torch.sub(myData, torch.mean(myData)), torch.std(myData) )

       #  # myData = torch.FloatTensor( butter_lowpass_filter(myData, 10, 2000, 2).flatten() )
       #  # myData = torch.FloatTensor( butter_bandpass_filter(myData, 1, 5, 200, order=6) )
    
       #  len_data = len(myData)
       #  target_data_shape = (int(len_data/self.window_samples), int(self.window_samples))
       #  required_elements = target_data_shape[0] * target_data_shape[1]

       #  self.data = torch.reshape(myData[0:required_elements], target_data_shape)
       #  self.target = torch.reshape(myTarget[0:required_elements], target_data_shape)
       # # myTarget = torch.reshape(myTarget[0:required_elements], target_data_shape)
