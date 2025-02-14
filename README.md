# RadarECGHD

#### Data loader:
- 'dataset2' loads the processed combined radar data and ecg signal in 5 second segment files. (from DATASET folder)

#### Our methods:
- 'hyperecg' uses dataset2 and joint training of the regressor model hypervector and projection matrix. Various metrics are caclulated including accuracy Pearson correlation coefficient, average absolute error of PQRST peaks, and median absolute error PQRST peaks.
- 'regHD2' uses dataset2 and uses sliding window to try to infer ecg signal with regHD.

## Instructions:
- With 'hyperecg.py', make sure the 'path_to_DS' variable is set to your path to the dataset, downloadable from https://ssd.mathworks.com/supportfiles/SPT/data/SynchronizedRadarECGData.zip
- There is the option to change which files of radar and ECG data are used, hyperparameters: hypervector dimensions, learning rate, train iterations, window size

In terminal, use command:
- python3 hyperecg.py

Some packages that may need installation:
```
pip install PyWavelets
pip install torchmetrics
pip install neurokit2
```

## Citation
```
@INPROCEEDINGS{10820450,
  author={Gaddi, Matilda and Ponzina, Flavio and Asgarinejad, Fatemeh and Aksanli, Baris and Rosing, Tajana},
  booktitle={2024 IEEE 24th International Conference on Bioinformatics and Bioengineering (BIBE)}, 
  title={HyperECG: ECG Signal Inference From Radar With Hyperdimensional Computing}, 
  year={2024},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Runtime;Accuracy;Computational modeling;Estimation;Radar;Medical services;Electrocardiography;Encoding;Monitoring;Hyperdimensional Computing;ECG monitoring;wireless sensing;model personalization},
  doi={10.1109/BIBE63649.2024.10820450}}
  ```
