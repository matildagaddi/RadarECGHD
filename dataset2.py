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
from modwt import *

class MyDataset(data.Dataset):
    def __init__(self, path, radar_files, ecg_files, window_size, device):
        self.path = path
        self.radar_files = radar_files
        self.ecg_files = ecg_files
        self.window_size = int(window_size)
        self.device = device
        self.load_data()
        self.normalize()

    def __len__(self):
        return self.data.shape[1] - self.window_size

    def __getitem__(self, index):
        sample = self.data[:, index:index+self.window_size] 
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
                if(len(mat["ecg_l"][0]) != 1024): # file already comes subsampled from OG 2000 pts per second, now 200 pts per second
                    print("WRONG: ", len(mat["ecg_l"][0]) )
                    exit()
                if myTarget is None:
                    myTarget = torch.from_numpy(mat["ecg_l"][0])
                else:
                    myTarget = torch.cat((myTarget, torch.from_numpy(mat["ecg_l"][0])), 0)
        self.data = myData.float() 
        self.target = myTarget.float()

    def normalize(self):
        self.DATA_MAX = self.data.max()
        self.DATA_MIN = self.data.min()
        self.TARGET_MAX = self.target.max()
        self.TARGET_MIN = self.target.min()
        self.data = (self.data - self.DATA_MIN) / (self.DATA_MAX - self.DATA_MIN)
        self.target = (self.target - self.TARGET_MIN) / (self.TARGET_MAX - self.TARGET_MIN)

        wt = modwt(self.data, 'sym2', 5)
        self.data = modwtmra(wt, 'sym2')
        self.data = self.data[3:5, :]        
        
        
    def get_params(self):
        return self.DATA_MIN, self.DATA_MAX, self.TARGET_MIN, self.TARGET_MAX, self.data.shape[0]