##dataset2multi

import os
import os.path
import torch
from torch.utils import data
import numpy as np

import scipy
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
#import matplotlib.pyplot as plt
from datetime import datetime, date, time

class MyDataset(data.Dataset):
    def __init__(self, path, radar_files, ecg_files, window_size, device):
        self.path = path
        self.radar_files = radar_files
        self.ecg_files = ecg_files
        self.window_size = int(window_size)
        self.device = device
        self.load_data()

    def __len__(self): #this method indicates how many times you have to loop when iterating over the dataset (line 58)
        return self.data.size(0) 

    def __getitem__(self, index): #given an index (from 0, to __len__ output), return sample and label
        sample = self.data[index] 
        target = self.target[index]
        return sample, target

    def load_data(self):
        myData = [] ### do with tensors instead for speed? #######
        myTarget = []

        for file in self.radar_files:
	        fr = self.path+file
	        if os.path.isfile(fr):
	            mat = loadmat(fr)
	            if(len(mat["radar_l"][0]) != 1024):
	                print("WRONG: ", len(mat["radar_l"][0]) )
	                exit()

	            myData.append(mat["radar_l"][0]) 

        for file in self.ecg_files:
	        fe = self.path+file
	        if os.path.isfile(fe):
	            mat = loadmat(fe)
	            if(len(mat["ecg_l"][0]) != 1024):
	                print("WRONG: ", len(mat["ecg_l"][0]) )
	                exit()

	            myTarget.append(mat["ecg_l"][0])


        myData = torch.FloatTensor( myData ) ## shape == (num files, 1024 points)
        myTarget = torch.FloatTensor( myTarget )

        self.data = myData 
        self.target = myTarget
        
       #  # myData = torch.FloatTensor( butter_lowpass_filter(myData, 10, 2000, 2).flatten() )
       #  # myData = torch.FloatTensor( butter_bandpass_filter(myData, 1, 5, 200, order=6) ) ### Eric mentioned using bandpass filter
    
       #  len_data = len(myData)
       #  target_data_shape = (int(len_data/self.window_samples), int(self.window_samples))
       #  required_elements = target_data_shape[0] * target_data_shape[1]

       #  self.data = torch.reshape(myData[0:required_elements], target_data_shape)
       #  self.target = torch.reshape(myTarget[0:required_elements], target_data_shape)
       # # myTarget = torch.reshape(myTarget[0:required_elements], target_data_shape)
