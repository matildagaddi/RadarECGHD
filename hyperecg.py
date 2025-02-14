import math
import torch
import torch.nn as nn
import torch.utils.data as data

from tqdm import tqdm

from dataset2 import MyDataset
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from scipy.signal import find_peaks
import datetime

from timeit import default_timer as timer


DIMENSIONS = [10000]  
WINDOW_SIZE = [200] # Flavios: 256
# 200 pts = 1 sec, 1 pt = 0.005 sec, 1 pt = 5 milliseconds
NUM_FEATURES = WINDOW_SIZE 
BATCH_SIZE = 1
LEARN_RATE = [1e-6]
SAMPLE_RATE = [230]
FS = [600] #has to be bigger than cutoff*2 (to make between 0 and 1)
CUTOFF = 30
PEAK_THRESH = 0.3
TRAIN_ITERS = 50
FINETUNE_ITERS = 10

### FOR FINAL EXPERIMENTS: ADD AS MANY FILES AS POSSIBLE while remaining consistent in number of files across subjects
train_files_r= ['trainVal/radar/GDN0004_Resting_radar_1.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_2.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_3.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_4.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_5.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_6.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_7.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_8.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_9.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_10.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_11.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_12.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_13.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_14.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_15.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_16.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_17.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_18.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_19.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_20.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_21.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_22.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_23.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_24.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_25.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_26.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_27.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_28.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_29.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_30.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_31.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_32.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_33.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_34.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_35.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_36.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_37.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_38.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_39.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_40.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_41.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_42.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_43.mat',
                # 'trainVal/radar/GDN0004_Resting_radar_44.mat',
               ]

train_files_e= ['trainVal/ecg/GDN0004_Resting_ecg_1.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_2.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_3.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_4.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_5.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_6.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_7.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_8.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_9.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_10.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_11.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_12.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_13.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_14.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_15.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_16.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_17.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_18.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_19.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_20.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_21.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_22.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_23.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_24.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_25.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_26.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_27.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_28.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_29.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_30.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_31.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_32.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_33.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_34.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_35.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_36.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_37.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_38.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_39.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_40.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_41.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_42.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_43.mat',
                # 'trainVal/ecg/GDN0004_Resting_ecg_44.mat'
               ]

test_files_r= ['trainVal/radar/GDN0004_Resting_radar_45.mat',
               # 'trainVal/radar/GDN0004_Resting_radar_46.mat',
               # 'trainVal/radar/GDN0004_Resting_radar_47.mat',

              ]
test_files_e= ['trainVal/ecg/GDN0004_Resting_ecg_45.mat',
               # 'trainVal/ecg/GDN0004_Resting_ecg_46.mat',
               # 'trainVal/ecg/GDN0004_Resting_ecg_47.mat',

              ]
path_to_DS = '../DATASET/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))


# THIS IS FOR THE BARPLOT #############################################
# subjects = ("Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6")
# values = {
#     'LOO':      (0.45, 0.19, 0.52, 0.22, 0.28, 0.58),
#     'Finetune': (0.81, 0.84, 0.87, 0.90, 0.82, 0.84),
# }
# x = np.arange(len(subjects))  # the label locations
# width = 0.35  # the width of the bars
# multiplier = 0
# color = ["slategray", "limegreen"]

# fig, ax = plt.subplots(figsize=(10, 5), dpi=3600)

# for i, (attribute, measurement) in enumerate(values.items()):
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute, color=color[i])
#     ax.bar_label(rects, padding=3, fontsize=10)
#     multiplier += 1

# # ax.set_ylabel('Correlation')
# # ax.set_xticks(x + width, subjects)
# ax.legend(loc='upper left', ncol=2, fontsize=12)
# ax.set_ylim(0, 1)
# # plt.figure(figsize=(7, 5), dpi=3600)
# plt.savefig(f'plot.png') #need date so it doesn't rewrite previous file

# exit()
#########################################################################






pAbsErrs, qAbsErrs, rAbsErrs, sAbsErrs, tAbsErrs = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
allCorr = np.array([])


# train_files_r= ['trainVal/radar/GDN0002_Resting_radar_1.mat',
#                 'trainVal/radar/GDN0002_Resting_radar_2.mat',
#                 'trainVal/radar/GDN0002_Resting_radar_3.mat',
#                 'trainVal/radar/GDN0002_Resting_radar_4.mat',
#                 'trainVal/radar/GDN0002_Resting_radar_5.mat',
#                 'trainVal/radar/GDN0002_Resting_radar_6.mat',
#                 'trainVal/radar/GDN0002_Resting_radar_7.mat',
#                 'trainVal/radar/GDN0002_Resting_radar_8.mat',
#                 'trainVal/radar/GDN0002_Resting_radar_9.mat',
                
#                 'trainVal/radar/GDN0003_Resting_radar_1.mat',
#                 'trainVal/radar/GDN0003_Resting_radar_2.mat',
#                 'trainVal/radar/GDN0003_Resting_radar_3.mat',
#                 'trainVal/radar/GDN0003_Resting_radar_4.mat',
#                 'trainVal/radar/GDN0003_Resting_radar_5.mat',
#                 'trainVal/radar/GDN0003_Resting_radar_6.mat',
#                 'trainVal/radar/GDN0003_Resting_radar_7.mat',
#                 'trainVal/radar/GDN0003_Resting_radar_8.mat',
#                 'trainVal/radar/GDN0003_Resting_radar_9.mat',

#                 'trainVal/radar/GDN0004_Resting_radar_1.mat',
#                 'trainVal/radar/GDN0004_Resting_radar_2.mat',
#                 'trainVal/radar/GDN0004_Resting_radar_3.mat',
#                 'trainVal/radar/GDN0004_Resting_radar_4.mat',
#                 'trainVal/radar/GDN0004_Resting_radar_5.mat',
#                 'trainVal/radar/GDN0004_Resting_radar_6.mat',
#                 'trainVal/radar/GDN0004_Resting_radar_7.mat',
#                 'trainVal/radar/GDN0004_Resting_radar_8.mat',
#                 'trainVal/radar/GDN0004_Resting_radar_9.mat',

#                 'trainVal/radar/GDN0005_Resting_radar_1.mat',
#                 'trainVal/radar/GDN0005_Resting_radar_2.mat',
#                 'trainVal/radar/GDN0005_Resting_radar_3.mat',
#                 'trainVal/radar/GDN0005_Resting_radar_4.mat',
#                 'trainVal/radar/GDN0005_Resting_radar_5.mat',
#                 'trainVal/radar/GDN0005_Resting_radar_6.mat',
#                 'trainVal/radar/GDN0005_Resting_radar_7.mat',
#                 'trainVal/radar/GDN0005_Resting_radar_8.mat',
#                 'trainVal/radar/GDN0005_Resting_radar_9.mat',
# ]


# train_files_e= ['trainVal/ecg/GDN0002_Resting_ecg_1.mat',
#                 'trainVal/ecg/GDN0002_Resting_ecg_2.mat',
#                 'trainVal/ecg/GDN0002_Resting_ecg_3.mat',
#                 'trainVal/ecg/GDN0002_Resting_ecg_4.mat',
#                 'trainVal/ecg/GDN0002_Resting_ecg_5.mat',
#                 'trainVal/ecg/GDN0002_Resting_ecg_6.mat',
#                 'trainVal/ecg/GDN0002_Resting_ecg_7.mat',
#                 'trainVal/ecg/GDN0002_Resting_ecg_8.mat',
#                 'trainVal/ecg/GDN0002_Resting_ecg_9.mat',

#                 'trainVal/ecg/GDN0003_Resting_ecg_1.mat',
#                 'trainVal/ecg/GDN0003_Resting_ecg_2.mat',
#                 'trainVal/ecg/GDN0003_Resting_ecg_3.mat',
#                 'trainVal/ecg/GDN0003_Resting_ecg_4.mat',
#                 'trainVal/ecg/GDN0003_Resting_ecg_5.mat',
#                 'trainVal/ecg/GDN0003_Resting_ecg_6.mat',
#                 'trainVal/ecg/GDN0003_Resting_ecg_7.mat',
#                 'trainVal/ecg/GDN0003_Resting_ecg_8.mat',
#                 'trainVal/ecg/GDN0003_Resting_ecg_9.mat',

#                 'trainVal/ecg/GDN0004_Resting_ecg_1.mat',
#                 'trainVal/ecg/GDN0004_Resting_ecg_2.mat',
#                 'trainVal/ecg/GDN0004_Resting_ecg_3.mat',
#                 'trainVal/ecg/GDN0004_Resting_ecg_4.mat',
#                 'trainVal/ecg/GDN0004_Resting_ecg_5.mat',
#                 'trainVal/ecg/GDN0004_Resting_ecg_6.mat',
#                 'trainVal/ecg/GDN0004_Resting_ecg_7.mat',
#                 'trainVal/ecg/GDN0004_Resting_ecg_8.mat',
#                 'trainVal/ecg/GDN0004_Resting_ecg_9.mat',

#                 'trainVal/ecg/GDN0005_Resting_ecg_1.mat',
#                 'trainVal/ecg/GDN0005_Resting_ecg_2.mat',
#                 'trainVal/ecg/GDN0005_Resting_ecg_3.mat',
#                 'trainVal/ecg/GDN0005_Resting_ecg_4.mat',
#                 'trainVal/ecg/GDN0005_Resting_ecg_5.mat',
#                 'trainVal/ecg/GDN0005_Resting_ecg_6.mat',
#                 'trainVal/ecg/GDN0005_Resting_ecg_7.mat',
#                 'trainVal/ecg/GDN0005_Resting_ecg_8.mat',
#                 'trainVal/ecg/GDN0005_Resting_ecg_9.mat',
#                ]

# test_files_r= ['trainVal/radar/GDN0001_Resting_radar_13.mat',
#                'trainVal/radar/GDN0001_Resting_radar_14.mat',
#                'trainVal/radar/GDN0001_Resting_radar_15.mat',

#               ]
# test_files_e= ['trainVal/ecg/GDN0001_Resting_ecg_13.mat',
#                'trainVal/ecg/GDN0001_Resting_ecg_14.mat',
#                'trainVal/ecg/GDN0001_Resting_ecg_15.mat',
# ]




for subject in ['01', '02', '03', '04']: #subject 06 has different indexing bc it's in the test folder
    train_files_r[:][20:22] = subject
    train_files_e[:][18:20] = subject
    test_files_r[:][20:22] = subject
    test_files_r[:][18:20] = subject

    ## HERE I LEARN THE MATRIX ######################################################
    for f in NUM_FEATURES:
      train_ds = MyDataset(path=path_to_DS,radar_files=train_files_r,ecg_files=train_files_e, window_size=f,device=device)
      test_ds = MyDataset(path=path_to_DS,radar_files=test_files_r,ecg_files=test_files_e, window_size=f,device=device)
      DATA_MIN, DATA_MAX, TARGET_MIN, TARGET_MAX, channels = test_ds.get_params()
      train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE) 
      test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE)
      for d in DIMENSIONS:
        for lr in LEARN_RATE:
          CNN = FlavioNet(f*channels, d, device).to(device)
          criterion = torch.nn.CrossEntropyLoss()
          criterion.to(device)
          optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-5, weight_decay=1e-4)
          scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=.1)

          train(train_dl, CNN, criterion, optimizer, scheduler, 100, device)
    #################################################################################


    for f in NUM_FEATURES:
        for d in DIMENSIONS:
            for lr in LEARN_RATE: 
                for fs in FS:
                    for sr in SAMPLE_RATE:
                        train_ds = MyDataset(path=path_to_DS,radar_files=train_files_r,ecg_files=train_files_e, window_size=f,device=device)
                        test_ds = MyDataset(path=path_to_DS,radar_files=test_files_r,ecg_files=test_files_e, window_size=f,device=device)
                        DATA_MIN, DATA_MAX, TARGET_MIN, TARGET_MAX, channels = test_ds.get_params()
                        train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE) 
                        test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE)
                        
                        model = HDradarECG(f*channels, d, lr, device).to(device)
                        model.project2.weight = CNN.proj.weight #copy the learned matrix to the new HDC model
                        
                        #Just baselines to measure performance
                        # CNN = Baseline8().to(device)
                        # criterion = torch.nn.CrossEntropyLoss()
                        # criterion.to(device)
                        # optimizer = torch.optim.Adam(CNN.parameters(), lr=1e-5, weight_decay=1e-4)
                        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=.1)
                        # train(train_dl, CNN, criterion, optimizer, scheduler, 100, device)

                        with torch.no_grad():
                            for _ in range(TRAIN_ITERS):
                                for samples, label in tqdm(train_dl, desc="Iteration {}".format(_ + 1)):
                                    samples = samples.to(device)
                                    label = label.to(device)
                                    # start = timer()
                                    model.model_update(samples, label)
                                    # end = timer()
                                    # print(end - start)
                                    # input()

                        #this is for personalizarion
                        for subject in ['05']: #subject 06 has different indexing bc it's in the test folder
                            train_files_r[:][20:22] = subject
                            train_files_e[:][18:20] = subject
                            test_files_r[:][20:22] = subject
                            test_files_r[:][18:20] = subject
                        train_ds = MyDataset(path=path_to_DS,radar_files=train_files_r,ecg_files=train_files_e, window_size=f,device=device)
                        test_ds = MyDataset(path=path_to_DS,radar_files=test_files_r,ecg_files=test_files_e, window_size=f,device=device)
                        DATA_MIN, DATA_MAX, TARGET_MIN, TARGET_MAX, channels = test_ds.get_params()
                        train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE) 
                        test_dl = data.DataLoader(test_ds, batch_size=BATCH_SIZE)
                        with torch.no_grad():
                            for _ in range(FINETUNE_ITERS):
                                for samples, label in tqdm(train_dl, desc="Iteration {}".format(_ + 1)):
                                    samples = samples.to(device)
                                    label = label.to(device)
                                    model.model_update(samples, label)
                        ##########################################
                  
                        mse, labelsArr, predictionsArr = test(test_dl, model, TARGET_MIN, TARGET_MAX, True, device)
                        
                        #preserve peaks
                        peaks, _ = find_peaks(predictionsArr, height=PEAK_THRESH)
                        predictionsArrFiltered = butter_lowpass_filter(predictionsArr, CUTOFF, fs, 5)
                        AAEs, medAEs, pAbsErr, qAbsErr, rAbsErr, sAbsErr, tAbsErr = get_AAEs_medAEs(labelsArr, predictionsArrFiltered, sr)
                        print(f'----- Subject {subject}, num train files: {len(train_files_r)}, num test files: {len(test_files_r)} -----')
                        print(f'AAEs (p,q,r,s,t):{AAEs}') 
                        print(f'MedAEs (p,q,r,s,t):{medAEs}') 
                        #replace filtered peaks with original peaks for higher correlation and better R peak height
                        predictionsArrFiltered[peaks] = predictionsArr[peaks]
                        corr = torch.corrcoef(torch.tensor(np.array([labelsArr.flatten(),predictionsArrFiltered])))[0][1]
                        print(f"Testing correlation of {corr}")
                        print(f"Testing mean squared error of {(mse.compute().item()):.20f}")

                        plt.figure(figsize=(7, 5), dpi=3600)
                        # plt.plot(np.arange(len(labelsArr.flatten())), labelsArr.flatten(), label='Actual', color='blue')
                        # plt.plot(np.arange(len(predictionsArrFiltered)), predictionsArrFiltered, label='Predicted', color='green')
                        plt.plot(np.arange(1250), labelsArr.flatten()[250:1500], label='Actual', color='blue')
                        plt.plot(np.arange(1250), predictionsArrFiltered[250:1500], label='Predicted', color='green')
                        # plt.title(f'Predicted ECG- iters:{TRAIN_ITERS}, SR:{sr}, window:{WINDOW_SIZE}')
                        # plt.annotate(f'Subject {subject}, num train files: {len(train_files_r)}, num test files: {len(test_files_r)} \n \
                            # AAEs (p,q,r,s,t):{AAEs} \n MedAEs (p,q,r,s,t):{medAEs} \n correlation: {corr:.5f}, MSE:{(mse.compute().item()):.5f}',
                            #         xy = (1.0, -0.2),
                            #         xycoords='axes fraction',
                            #         ha='right',
                            #         va="center",
                            #         fontsize=10,
                            #         annotation_clip=False)
                        plt.tight_layout(h_pad=4)
                        plt.legend()
                        plt.savefig(f'FINAL_{(mse.compute().item()):.4f}_{corr:.4f}_{d}_{f}_{lr}_{fs}_{PEAK_THRESH}.png') #need date so it doesn't rewrite previous file

                plt.clf()

    pAbsErrs = np.append(pAbsErrs, pAbsErr)
    qAbsErrs = np.append(qAbsErrs, qAbsErr)
    rAbsErrs = np.append(rAbsErrs, rAbsErr)
    sAbsErrs = np.append(sAbsErrs, sAbsErr)
    tAbsErrs = np.append(tAbsErrs, tAbsErr)
    allCorr = np.append(allCorr, corr)

overallMedAE = np.array([
    np.median(pAbsErrs),
    np.median(qAbsErrs),
    np.median(rAbsErrs),
    np.median(sAbsErrs),
    np.median(tAbsErrs)
])

overallMeanAE = np.array([
    np.mean(pAbsErrs),
    np.mean(qAbsErrs),
    np.mean(rAbsErrs),
    np.mean(sAbsErrs),
    np.mean(tAbsErrs)
])
print('-------------------------')
print(f'overallMedAE: {overallMedAE}')
print(f'median corr: {np.median(allCorr)}')
print(f'overallMeanAE: {overallMeanAE}')
print(f'mean corr: {np.mean(allCorr)}')