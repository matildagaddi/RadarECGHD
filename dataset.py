import os
import os.path
import torch
from torch.utils import data
#import pandas as pd
import numpy as np

import numpy as np
import scipy
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd
from sklearn.preprocessing import StandardScaler

from modwt import modwt

#standardize center of data / levels. get mean (or median) value for all, subract individual means and add overall mean

class MyDataset(data.Dataset):

    def __init__(self, root, name, train, window_size, window_samples, device, seed, transform=None, target_transform=None,):
        self.path = os.path.join(root, name) # building the path
        self.train = train #train or test data? Here I assume data is split in two separate files
        self.window_size = int(window_size) #size of the window (i.e., number of samples in this case). 
        self.window_samples = int(window_samples) #number of evenly spaced points sampled from window to not take all points (6000 for 3 seconds)
        self.transform = transform #if you want to make some transformation to your data. Check PyTorch data transform
        self.target_transform = target_transform #if you want to make some transformation to your data. Check PyTorch data transform
        self.device = device #cpu or cuda?
        self.seed = seed
        self.load_data()
        

    def __len__(self): #this method indicates how many times you have to loop when iterating over the dataset (line 58)
        return self.data_i.size(0) 

    # def shape(self):
    #     return self.data_i.size()

    def __getitem__(self, index): #given an index (from 0, to __len__ output), return sample and label
        sample_i = self.data_i[index] 
        sample_q = self.data_q[index] 
        label = self.data_ecg[index]
        return sample_i, sample_q, label

    # def extract_samples(self, radar_data): 
    #     return radar_data

    
    def load_data(self):
        # need to do 5 times, randomly assigning train and test, take average of accuracies

        # all_files = [
        # 'GDN0004/GDN0004_3_Apnea.mat','GDN0005/GDN0005_3_Apnea.mat', 
        # 'GDN0006/GDN0006_3_Apnea.mat', 'GDN0007/GDN0007_3_Apnea.mat',
        # 'GDN0008/GDN0008_3_Apnea.mat', 'GDN0009/GDN0009_3_Apnea.mat',

        # 'GDN0011/GDN0011_3_Apnea.mat', 'GDN0012/GDN0012_3_Apnea.mat',
        # 'GDN0013/GDN0013_3_Apnea.mat', 'GDN0014/GDN0014_3_Apnea.mat',
        # 'GDN0016/GDN0016_3_Apnea.mat', 'GDN0017/GDN0017_3_Apnea.mat',
        # 'GDN0018/GDN0018_3_Apnea.mat', 

        # 'GDN0021/GDN0021_3_Apnea.mat', 'GDN0022/GDN0022_3_Apnea.mat', 
        # 'GDN0023/GDN0023_3_Apnea.mat', 'GDN0025/GDN0025_3_Apnea.mat', 
        # 'GDN0027/GDN0027_3_Apnea.mat', 'GDN0028/GDN0028_3_Apnea.mat',

        # 'GDN0010/GDN0010_3_Apnea.mat', 
        # 'GDN0019/GDN0019_3_Apnea.mat', 'GDN0020/GDN0020_3_Apnea.mat', #20 has weird jumps
        # 'GDN0029/GDN0029_3_Apnea.mat','GDN0030/GDN0030_3_Apnea.mat'
        # ]

        # np.random.seed(seed = self.seed) 
        # percent80 = int(np.floor(len(all_files)*.8))
        # mix_files = np.random.permutation(all_files)
        # rand_train_files = mix_files[:percent80]
        # rand_test_files = mix_files[percent80:]

        
        # train_files = [
        # 'GDN0004/GDN0004_3_Apnea.mat','GDN0005/GDN0005_3_Apnea.mat', 
        # 'GDN0006/GDN0006_3_Apnea.mat', 'GDN0007/GDN0007_3_Apnea.mat',
        # 'GDN0008/GDN0008_3_Apnea.mat', 'GDN0009/GDN0009_3_Apnea.mat',

        # 'GDN0011/GDN0011_3_Apnea.mat', 'GDN0012/GDN0012_3_Apnea.mat',
        # 'GDN0013/GDN0013_3_Apnea.mat', 'GDN0014/GDN0014_3_Apnea.mat',
        # 'GDN0016/GDN0016_3_Apnea.mat', 'GDN0017/GDN0017_3_Apnea.mat',
        # 'GDN0018/GDN0018_3_Apnea.mat', 

        # 'GDN0021/GDN0021_3_Apnea.mat', 'GDN0022/GDN0022_3_Apnea.mat', 
        # 'GDN0023/GDN0023_3_Apnea.mat', 'GDN0025/GDN0025_3_Apnea.mat', 
        # 'GDN0027/GDN0027_3_Apnea.mat', 'GDN0028/GDN0028_3_Apnea.mat'

        # ]
        # test_files = [
        # 'GDN0010/GDN0010_3_Apnea.mat', 
        # 'GDN0019/GDN0019_3_Apnea.mat', 'GDN0020/GDN0020_3_Apnea.mat', 
        # 'GDN0029/GDN0029_3_Apnea.mat','GDN0030/GDN0030_3_Apnea.mat'
        # ]

        path = '/Users/matildagaddi/Documents/SEElab/apnea_dataset/'
        step_for_samples = int(self.window_size//self.window_samples)
        radar_data = {'radar_i': torch.ShortTensor(), 'radar_q': torch.ShortTensor(), 'tfm_ecg1': torch.FloatTensor()}

        def process_data(file, radar_data): 
            file_dict = loadmat(path+file) #loads dictionary 
            cur_radar_data = {'radar_i': file_dict['radar_i'], 'radar_q': file_dict['radar_q'], 
            'tfm_ecg1': pd.Series(file_dict['tfm_ecg1'].reshape(len(file_dict['tfm_ecg1']))).rolling(40).mean()}
            #ecg not smoothed, need to take mean of nearest points (40: phase length before sampling), ecg2 is an amplified version?
            for k in cur_radar_data.keys():
                cur_radar_data[k] = cur_radar_data[k][:((len(cur_radar_data[k])//self.window_size)*self.window_size):step_for_samples] #cut off extra data #take only sample of window

            #standardize center of values 
            stdscaleri = StandardScaler()
            stdscaleri.fit(cur_radar_data['radar_i'])
            cur_radar_data['radar_i'] = torch.Tensor(stdscaleri.transform(cur_radar_data['radar_i']))
            stdscalerq = StandardScaler()
            stdscalerq.fit(cur_radar_data['radar_q'])
            cur_radar_data['radar_q'] = torch.Tensor(stdscaleri.transform(cur_radar_data['radar_q']))

            radar_data['radar_i'] = torch.cat((radar_data['radar_i'], cur_radar_data['radar_i']))
            radar_data['radar_q'] = torch.cat((radar_data['radar_q'], cur_radar_data['radar_q']))
            radar_data['tfm_ecg1'] = torch.cat((radar_data['tfm_ecg1'], torch.Tensor(cur_radar_data['tfm_ecg1'])))
                                                                        # pd.Series(cur_radar_data['tfm_ecg1'] #*10
                                                                        # .reshape(len(cur_radar_data['tfm_ecg1'])))
                                                                        # .rolling(4).mean()))) #if smoothing after sampling (but this leaves more noise)
            
            


        if self.train:
            files = ['GDN0006/GDN0006_3_Apnea.mat']#rand_train_files
            ##print('loading training data')
        else:
            files = rand_test_files
            ##print('loading testing data')


        for file in files:
            process_data(file, radar_data)


        #radar_data = self.extract_samples(radar_data)
        

        data_i = radar_data['radar_i']
        data_q = radar_data['radar_q']
        data_ecg = radar_data['tfm_ecg1']
        
        # plt.plot(data_i[6144:7168]/5, color = 'orange') #1024 for 5 seconds = 205 points per second (original was 2000 per second)
        # plt.plot(data_q[6144:7168]/5, color = 'purple') #they might just be using data_q?
        # # plt.plot(data_i[6144:7168]+data_q[6144:7168]/5, color = 'pink')
        # # plt.plot(data_i[6144:7168]*data_q[6144:7168]/5, color = 'green')
        # # plt.plot(data_i[6144:7168]+data_q[6144:7168]/5*data_i[6144:7168]*data_q[6144:7168], color = 'blue')
        # # plt.plot(data_i[6144:7168]*data_q[6144:7168]/5*data_i[6144:7168]+data_q[6144:7168], color = 'red')
        # plt.plot(data_ecg[6144:7168], color = 'black') #7 7th window of 5 seconds 0-32000 1024*6 = 6144
        # #plt.plot(modwt(data_q, 'sym4', 1)/5, color = 'yellow')
        # plt.show()  

        # print(torch.mean(torch.FloatTensor(data_i)))

        len_data = (len(data_i),)
        ##print(f'Number of files = {len(files)}. Data length = {len_data}.')
        self.data_i = torch.reshape(data_i, len_data)
        self.data_q = torch.reshape(data_q, len_data)
        self.data_ecg = torch.reshape(data_ecg, len_data)
