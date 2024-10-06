import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy import signal
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import os
from pathlib import Path
from scipy.signal import butter, filtfilt
from sklearn.metrics.pairwise import pairwise_distances
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Load seismic data from mseed file
script_dir = Path(__file__).parent
mseed_file = script_dir / 'XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed'
if not os.path.exists(mseed_file):
    raise FileNotFoundError(f"File not found: {mseed_file}")

st = read(mseed_file)
tr = st[0]  # Take the first trace

# Parameters for STA/LTA algorithm
sta_window = 5  # Short-term average window in seconds
lta_window = 60  # Long-term average window in seconds
threshold_on = 2.5  # Trigger threshold for event detection
threshold_off = 0.9  # Threshold for event ending

# Convert the STA/LTA window to number of samples
sampling_rate = tr.stats.sampling_rate
sta_samples = int(sta_window * sampling_rate)
lta_samples = int(lta_window * sampling_rate)

# Filter the trace to bring out particular frequencies (bandpass filter)
minfreq = 0.01
maxfreq = 0.5
tr_filtered = tr.copy()
tr_filtered.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)

# Butterworth filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Use of Butterworth filter
tr_filtered = tr.copy()
tr_filtered.data = butter_bandpass_filter(tr_filtered.data, minfreq, maxfreq, sampling_rate)

# Apply the STA/LTA algorithm
cft = classic_sta_lta(tr_filtered.data, sta_samples, lta_samples)

# Find the onset and end times of events
onsets = trigger_onset(cft, threshold_on, threshold_off)

# Define a custom dataset class for CNN
data = tr_filtered.data
labels = np.zeros(len(data))
for onset in onsets:
    labels[onset[0]:onset[1]] = 1  # Label events as 1

class SeismicDataset(Dataset):
    def __init__(self, data, labels, window_size=512):
        self.data = data
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.window_size]
        y = self.labels[idx:idx+self.window_size]
        y = 1 if y.max() > 0 else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

window_size = 512
dataset = SeismicDataset(data, labels, window_size)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define CNN model for seismic data
class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        fc_input_size = (window_size // 8) * 64
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Training the CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SeismicCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Inference
event_times = []
model.eval()
with torch.no_grad():
    for i in range(len(dataset)):
        input_data, _ = dataset[i]
        input_data = input_data.to(device)
        input_data = input_data.unsqueeze(0)  # Add batch dimension
        output = model(input_data)
        _, predicted = torch.max(output.data, 1)
        if predicted.item() == 1:
            event_times.append(tr.stats.starttime + timedelta(seconds=(i * tr.stats.delta)))

# Print detected events
for event_time in event_times:
    print(f"Detected event at {event_time}")

# Save the model
torch.save(model.state_dict(), 'seismic_cnn.pth')

# Load the model
model = SeismicCNN()
model.load_state_dict(torch.load('seismic_cnn.pth'))
model.to(device)

# Plot the results
fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

# Original Seismic Signal
axs[0].plot(tr.times(), tr.data, 'k')
axs[0].set_title('Original Seismic Signal')
axs[0].set_ylabel('Amplitude')

# Filtered Seismic Signal
axs[1].plot(tr_filtered.times(), tr_filtered.data, 'b')
axs[1].set_title('Filtered Seismic Signal (Bandpass)')
axs[1].set_ylabel('Amplitude')

# STA/LTA Characteristic Function
axs[2].plot(tr.times(), cft, 'b')
axs[2].hlines([threshold_on, threshold_off], tr.times()[0], tr.times()[-1], colors=['r', 'g'], linestyles='--')
axs[2].set_title('STA/LTA Characteristic Function')
axs[2].set_ylabel('STA/LTA Ratio')

# Spectrogram
f, t, Sxx = signal.spectrogram(tr_filtered.data, tr_filtered.stats.sampling_rate, nperseg=256, noverlap=128)
im = axs[3].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')
axs[3].set_title('Spectrogram of Filtered Seismic Signal')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Frequency (Hz)')
cbar = plt.colorbar(im, ax=axs[3], orientation='vertical')
cbar.set_label('Power (dB)')

plt.tight_layout()
plt.show()