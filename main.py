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

# Set the file paths
script_dir = Path(__file__).parent
mseed_file = script_dir / 'XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed'

if not os.path.exists(mseed_file):
    raise FileNotFoundError(f"File not found: {mseed_file}")

# Load the seismic data
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

# Butterworth filter functions
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

# Apply Butterworth filter
tr_filtered = tr.copy()
tr_filtered.data = butter_bandpass_filter(tr_filtered.data, minfreq, maxfreq, sampling_rate)

# Apply the STA/LTA algorithm
cft = classic_sta_lta(tr_filtered.data, sta_samples, lta_samples)

# Find the onset and end times of events
onsets = trigger_onset(cft, threshold_on, threshold_off)

# Label the data based on detected events
data = tr_filtered.data
labels = np.zeros(len(data))
for onset in onsets:
    labels[onset[0]:onset[1]] = 1  # Label events as 1

# Define a custom dataset class for CNN
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

# Check if the model already exists
model_path = script_dir / 'seismic_cnn.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SeismicCNN().to(device)

if model_path.exists():
    # Load the saved model
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from disk.")
else:
    # Train the CNN model
    print("Model not found. Starting training.")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels_batch in dataloader:
            inputs = inputs.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), model_path)
    print("Model saved to disk.")

# Inference
event_windows = []
model.eval()
with torch.no_grad():
    for idx in range(len(dataset)):
        input_data, _ = dataset[idx]
        input_data = input_data.to(device)
        input_data = input_data.unsqueeze(0)  # Add batch dimension
        output = model(input_data)
        _, predicted = torch.max(output.data, 1)
        if predicted.item() == 1:
            # Convert window index to sample index
            sample_idx = idx
            event_windows.append(sample_idx)

# Post-process the detected event windows to merge contiguous windows
def merge_event_windows(event_windows, window_size):
    if not event_windows:
        return []
    events = []
    current_event = [event_windows[0], event_windows[0] + window_size]
    for idx in event_windows[1:]:
        if idx <= current_event[1]:
            # Extend the current event
            current_event[1] = idx + window_size
        else:
            # Save the current event and start a new one
            events.append(current_event)
            current_event = [idx, idx + window_size]
    events.append(current_event)
    return events

merged_events = merge_event_windows(event_windows, window_size)

# Collect event information and save to CSV
detections = []
for event in merged_events:
    event_start_idx = event[0]
    event_end_idx = event[1]
    event_start_time = tr.stats.starttime + event_start_idx * tr.stats.delta
    event_end_time = tr.stats.starttime + event_end_idx * tr.stats.delta
    duration = event_end_time - event_start_time  # Removed .total_seconds()
    amplitude = np.max(np.abs(tr_filtered.data[event_start_idx:event_end_idx]))
    detection = {
        'filename': mseed_file.name,
        'start_time': event_start_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
        'end_time': event_end_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
        'duration': duration,  # Now duration is in seconds
        'amplitude': amplitude
    }
    detections.append(detection)

# Save detections to CSV
output_csv = script_dir / 'detected_events.csv'
detections_df = pd.DataFrame(detections)
detections_df.to_csv(output_csv, index=False)
print(f"Detections saved to {output_csv}")

# Optional: Print the detections
for detection in detections:
    print(f"Detected event from {detection['start_time']} to {detection['end_time']}, "
          f"Duration: {detection['duration']:.2f}s, Amplitude: {detection['amplitude']:.2e}")

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

# Mark detected events on the plots
for event in merged_events:
    event_start_time = tr.stats.starttime + event[0] * tr.stats.delta
    event_end_time = tr.stats.starttime + event[1] * tr.stats.delta
    axs[0].axvspan(event_start_time - tr.stats.starttime, event_end_time - tr.stats.starttime, color='red', alpha=0.3)
    axs[1].axvspan(event_start_time - tr.stats.starttime, event_end_time - tr.stats.starttime, color='red', alpha=0.3)
    axs[2].axvspan(event_start_time - tr.stats.starttime, event_end_time - tr.stats.starttime, color='red', alpha=0.3)

plt.tight_layout()
plt.show()
