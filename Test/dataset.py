import os
import torch
from torch.utils import data
import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MyDataset(data.Dataset):

    def __init__(self, root, name, train, window_size, window_samples, device, seed, transform=None, target_transform=None,):
        self.path = os.path.join(root, name)
        self.train = train
        self.window_size = int(window_size)
        self.window_samples = int(window_samples)
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.seed = seed
        self.load_data()

    def __len__(self):
        return self.data_i.size(0)

    def __getitem__(self, index):
        sample_i = self.data_i[index]
        sample_q = self.data_q[index]
        label = self.data_ecg[index]
        return sample_i, sample_q, label

    def load_data(self):
        path = '/Users/matildagaddi/Documents/SEElab/apnea_dataset/'
        step_for_samples = int(self.window_size // self.window_samples)
        radar_data = {'radar_i': torch.ShortTensor(), 'radar_q': torch.ShortTensor(), 'tfm_ecg1': torch.FloatTensor()}

        def process_data(file, radar_data): 
            file_dict = loadmat(path + file)
            cur_radar_data = {'radar_i': file_dict['radar_i'], 'radar_q': file_dict['radar_q'], 
                              'tfm_ecg1': pd.Series(file_dict['tfm_ecg1'].reshape(len(file_dict['tfm_ecg1']))).rolling(40).mean()}
            for k in cur_radar_data.keys():
                cur_radar_data[k] = cur_radar_data[k][:((len(cur_radar_data[k]) // self.window_size) * self.window_size):step_for_samples]
            stdscaleri = StandardScaler()
            stdscaleri.fit(cur_radar_data['radar_i'])
            cur_radar_data['radar_i'] = torch.Tensor(stdscaleri.transform(cur_radar_data['radar_i']))
            stdscalerq = StandardScaler()
            stdscalerq.fit(cur_radar_data['radar_q'])
            cur_radar_data['radar_q'] = torch.Tensor(stdscaleri.transform(cur_radar_data['radar_q']))

            radar_data['radar_i'] = torch.cat((radar_data['radar_i'], cur_radar_data['radar_i']))
            radar_data['radar_q'] = torch.cat((radar_data['radar_q'], cur_radar_data['radar_q']))
            radar_data['tfm_ecg1'] = torch.cat((radar_data['tfm_ecg1'], torch.Tensor(cur_radar_data['tfm_ecg1'])))

        if self.train:
            files = ['GDN0006/GDN0006_3_Apnea.mat']  # Replace with actual train files
        else:
            files = ['GDN0010/GDN0010_3_Apnea.mat']  # Replace with actual test files

        for file in files:
            process_data(file, radar_data)

        data_i = radar_data['radar_i']
        data_q = radar_data['radar_q']
        data_ecg = radar_data['tfm_ecg1']

        len_data = (len(data_i),)
        self.data_i = torch.reshape(data_i, len_data)
        self.data_q = torch.reshape(data_q, len_data)
        self.data_ecg = torch.reshape(data_ecg, len_data)