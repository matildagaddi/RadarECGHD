# RadarECGHD

## Files
\* = most important/revelant files

### Dataloaders:
'dataset1' loads the raw radar and ecg data. (from apnea_dataset folder)
'dataset1_1' also loads the raw data, more flexible/compatible with regHD and sliding windows.
'dataset2' loads the processed combined radar data and ecg signal in 5 second segment files used in the matlab example. (from DATASET folder)
\*'dataset2multi' loads the processed data for multiple files at once.

### Our methods:
'regHD1' uses dataset1_1 and uses sliding window to try to infer ecg signal with regHD.
'regHD2' uses dataset2 and uses sliding window to try to infer ecg signal with regHD.
\* 'regHD2multi' uses dataset2multi and uses sliding window to try to infer ecg signal with regHD by training and on multiple files (testing on multiple files not checked/implemented yet but possible in this file).

### Baselines:
'trainBase1' uses MatildaNet (from the FatemehNet file) with dataset1.
'trainBase2' uses MatildaNet with dataset2 to infer with a neural network based on the one in the matlab example. Not in working condition.

'test' folder has Fatemeh's experiments.

## Instructions:
- With regHD2multi, modify the 'path_to_DS' variable for your directory to each dataset
- If you want to change the default files, designate which files you want to train and test on with the train_files_r, train_files_e, test_files_r, test_files_e variables.
- There is the option to change hyperparameters: hypervector dimensions, learning rate, train iterations, window size

in terminal, use command:
- python3 regHD2multifile.py
