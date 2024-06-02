# RadarECGHD

'dataset1' loads the raw radar and ecg data. (from apnea_dataset folder)
'dataset1_1' also loads the raw data, more flexible/compatible with regHD and sliding windows
'dataset2' loads the processed combined radar data and ecg signal in 5 second segment files used in the matlab example. (from DATASET folder)

'regHD1' uses dataset1_1 and uses sliding window to try to infer ecg signal
'regHD2' uses dataset2 and uses sliding window to try to infer ecg signal

'trainBase1' uses MatildaNet (from the FatemehNet file) with dataset1
'trainBase2' uses MatildaNet with dataset2 to infer with a neural network based on the one in the matlab example.

'test' folder has Fatemeh's experiments

'results' folder has plots (not sure if the images work) and data output by all the models.