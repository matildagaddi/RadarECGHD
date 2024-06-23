# #finding different peaks
# import neurokit2 as nk
# import numpy as np
# import pandas as pd

# # Retrieve ECG data from data folder
# ecg_signal = nk.data(dataset="ecg_1000hz")
# # Extract R-peaks locations
# _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=1000)
# plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
# plt.show() #AttributeError: 'NoneType' object has no attribute 'show'

import neurokit2 as nk
import matplotlib.pyplot as plt

# Retrieve ECG data from data folder
ecg_signal = nk.data(dataset="ecg_1000hz")
# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=1000)

# Plot the events using the events_plot function
nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)

# Save the plot to a file
plt.savefig('ecg_events_plot.png')

# Display the plot
plt.show()