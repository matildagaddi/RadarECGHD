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
    def __init__(self, path, radar_files, ecg_files, window_size, device):
        self.path = path
        self.radar_files = radar_files
        self.ecg_files = ecg_files
        self.window_size = int(window_size)
        self.device = device
        self.load_data()
        self.normalize()

    def __len__(self): #this method indicates how many times you have to loop when iterating over the dataset (line 58)
        return self.data.size(0) - self.window_size

    def __getitem__(self, index):
        sample = self.data[index:index+self.window_size] 
        target = self.target[index+self.window_size]
        return sample, target

    def load_data(self):
        myData = None
        myTarget = None
        for file in self.radar_files:
            fr = self.path+file
            if os.path.isfile(fr):
                mat = loadmat(fr)
                if(len(mat["radar_l"][0]) != 1024):
                    print("WRONG: ", len(mat["radar_l"][0]) )
                    exit()
                if myData is None:
                    myData = torch.from_numpy(mat["radar_l"][0])
                else:
                    myData = torch.cat((myData, torch.from_numpy(mat["radar_l"][0])), 0)

        for file in self.ecg_files:
            fe = self.path+file
            if os.path.isfile(fe):
                mat = loadmat(fe)
                if(len(mat["ecg_l"][0]) != 1024):
                    print("WRONG: ", len(mat["ecg_l"][0]) )
                    exit()
                if myTarget is None:
                    myTarget = torch.from_numpy(mat["ecg_l"][0])
                else:
                    myTarget = torch.cat((myTarget, torch.from_numpy(mat["ecg_l"][0])), 0)

        self.data = myData.float() 
        self.target = myTarget.float()


    def normalize(self):
        self.DATA_STD = self.data.std(0)
        self.DATA_MEAN = self.data.mean(0)
        self.TARGET_STD = self.target.std(0)
        self.TARGET_MEAN = self.target.mean(0)
        self.data = (self.data - self.DATA_MEAN) / self.DATA_STD
        self.target = (self.target - self.TARGET_MEAN) / self.TARGET_STD

    def get_params(self):
        return self.DATA_MEAN, self.DATA_STD, self.TARGET_MEAN, self.TARGET_STD