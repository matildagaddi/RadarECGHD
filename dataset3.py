import os
import os.path
import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from modwt import *


#dataset3 edanami_dataset
#The first 22 lines comprised a header. The variable names were displayed on line 23 in each column. 
#After #line 24, the signals were displayed. The information stored in each column was as follows: 
#column 1: time; 
#column 2: 24 GHz radar I-channel; 
#column 3: 24 GHz radar Q-channel; 
#column 4: 10 GHz radar I-channel; 
#column 5: respiratory band signal; and 
#column 6: ECG signal.
#sampling rate was 1000 Hz, 10 minutes collected for each subject

class MyDataset(data.Dataset):
    def __init__(self, path, subjects_sections, window_size, step, device):
        self.path = path
        self.subjects_sections = subjects_sections
        self.window_size = int(window_size)
        self.step = step
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
        myDataI = None
        #myDataQ = None
        myTarget = None
        for i in range(len(self.subjects_sections)):
            curSubj = self.subjects_sections[i]
            file = self.path+'subject'+str(curSubj[0])+'.csv'
            if os.path.isfile(file):
                curData, curTarget = np.loadtxt(file, dtype='float,float', delimiter=',', usecols=(1, 5), skiprows=23, unpack=True)
                if myDataI is None:
                    myDataI = torch.from_numpy(curData[curSubj[1]:curSubj[2]:self.step])
                else:
                    myDataI = torch.cat((myDataI, torch.from_numpy(curData[curSubj[1]:curSubj[2]:self.step])), 0)
                if myTarget is None:
                    myTarget = torch.from_numpy(curTarget[curSubj[1]:curSubj[2]:self.step])
                else:
                    myTarget = torch.cat((myTarget, torch.from_numpy(curTarget[curSubj[1]:curSubj[2]:self.step])), 0)
                plt.plot(myDataI, color = 'blue')
                plt.plot(myTarget, color = 'orange')
                plt.show()
        
        self.data = myDataI.float() 
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