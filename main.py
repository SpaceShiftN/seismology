import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import butter, filtfilt
from datetime import timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from collections import Counter

# Definition of the Dataset class
class SeismicDataset(Dataset):
    def _init_(self, data_info_list, window_size=512):
        """
        data_info_list: list of dictionaries with keys 'data', 'labels', 'filename'
        """
        self.data_info_list = data_info_list
        self.window_size = window_size
        self.indices = []
        for idx, info in enumerate(self.data_info_list):
            data_length = len(info['data']) - self.window_size
            for i in range(data_length):
                self.indices.append((idx, i))
    
    def _len_(self):
        return len(self.indices)
    
    def _getitem_(self, idx):
        data_idx, offset = self.indices[idx]
        info = self.data_info_list[data_idx]
        data = info['data']
        labels = info['labels']
        x = data[offset:offset+self.window_size]
        y = labels[offset:offset+self.window_size]
        y = 1 if y.max() > 0 else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Definition of the model
class SeismicCNN(nn.Module):
    def _init_(self, window_size=512):
        super(SeismicCNN, self)._init_()
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
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)
        return x

# Functions for filtering and processing data
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.999
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    if highcut >= fs / 2:
        highcut = fs / 2 - 0.1
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def classic_sta_lta_labels(tr_filtered, sta_samples, lta_samples, threshold_on, threshold_off):
    cft = classic_sta_lta(tr_filtered.data, sta_samples, lta_samples)
    onsets = trigger_onset(cft, threshold_on, threshold_off)
    labels = np.zeros(len(tr_filtered.data))
    for onset in onsets:
        labels[onset[0]:onset[1]] = 1
    return labels

# Function to load data from a file or directory
def load_data(data_path, window_size, minfreq, maxfreq, target_sampling_rate=None):
    data_info_list = []
    sampling_rates = []
    
    if os.path.isfile(data_path):
        mseed_files = [data_path]
    else:
        mseed_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.mseed'):
                    mseed_files.append(os.path.join(root, file))
    
    # First, collect all sampling rates
    for mseed_file in mseed_files:
        st = read(mseed_file)
        tr = st[0]
        sampling_rate = tr.stats.sampling_rate
        sampling_rates.append(sampling_rate)
    
    # Determine the target sampling rate
    sampling_rate_counter = Counter(sampling_rates)
    if target_sampling_rate is None:
        target_sampling_rate = sampling_rate_counter.most_common(1)[0][0]
        print(f"Determined target sampling rate: {target_sampling_rate} Hz")
    else:
        print(f"Using specified target sampling rate: {target_sampling_rate} Hz")
    
    for mseed_file in mseed_files:
        st = read(mseed_file)
        tr = st[0]
        original_sampling_rate = tr.stats.sampling_rate
        
        # Resample if necessary
        if original_sampling_rate != target_sampling_rate:
            tr.resample(target_sampling_rate)
            print(f"File {mseed_file} resampled from {original_sampling_rate} Hz to {target_sampling_rate} Hz")
        
        sampling_rate = tr.stats.sampling_rate
        
        tr_filtered = tr.copy()
        tr_filtered.data = butter_bandpass_filter(tr_filtered.data, minfreq, maxfreq, sampling_rate)
        
        # STA/LTA parameters
        sta_window = 5  # seconds
        lta_window = 60  # seconds
        sta_samples = int(sta_window * sampling_rate)
        lta_samples = int(lta_window * sampling_rate)
        threshold_on = 2.5
        threshold_off = 0.9

        # Calculate STA/LTA characteristic function
        cft = classic_sta_lta(tr_filtered.data, sta_samples, lta_samples)
        
        labels = classic_sta_lta_labels(tr_filtered, sta_samples, lta_samples, threshold_on, threshold_off)
        data = tr_filtered.data
        
        data_info = {
            'data': data,
            'labels': labels,
            'filename': mseed_file,
            'tr': tr,  # Save the original trace
            'tr_filtered': tr_filtered,  # Save the filtered trace
            'cft': cft,  # Save the STA/LTA characteristic function
            'sampling_rate': sampling_rate
        }
        
        data_info_list.append(data_info)
    
    dataset = SeismicDataset(data_info_list, window_size)
    return dataset, target_sampling_rate

# Function to merge event windows
def merge_event_windows(event_windows, window_size):
    if not event_windows:
        return []
    events = []
    current_event = [event_windows[0], event_windows[0] + window_size]
    for idx in event_windows[1:]:
        if idx <= current_event[1]:
            current_event[1] = idx + window_size
        else:
            events.append(current_event)
            current_event = [idx, idx + window_size]
    events.append(current_event)
    return events

# Main function
def main():
    parser = argparse.ArgumentParser(description='Seismic Data Processing')
    parser.add_argument('--mode', type=str, choices=['train', 'retrain', 'infer'], required=True, help='Mode of operation: train, retrain, or infer')
    parser.add_argument('--data', type=str, required=True, help='Path to the data file or directory')
    parser.add_argument('--model', type=str, default='seismic_model.pth', help='Path to the model file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--window_size', type=int, default=512, help='Window size')
    parser.add_argument('--minfreq', type=float, default=0.01, help='Minimum filter frequency')
    parser.add_argument('--maxfreq', type=float, default=0.5, help='Maximum filter frequency')
    parser.add_argument('--output', type=str, default='detected_events.csv', help='Path to the output CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for event detection')
    parser.add_argument('--target_sampling_rate', type=float, help='Target sampling rate for resampling data')
    parser.add_argument('--save_plots', action='store_true', help='Save plots as PNG files')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = SeismicCNN(window_size=args.window_size).to(device)
    
    if args.mode == 'train':
        # Load data
        dataset, sampling_rate = load_data(args.data, args.window_size, args.minfreq, args.maxfreq, args.target_sampling_rate)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(args.epochs):
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
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")
        
        # Save the model
        torch.save(model.state_dict(), args.model)
        print(f"Model saved to {args.model}")
    
    elif args.mode == 'infer':
        # Load existing model
        if not os.path.exists(args.model):
            print("Error: Model file not found.")
            sys.exit(1)
        model.load_state_dict(torch.load(args.model))
        model.eval()
        
        # Load data
        dataset, sampling_rate = load_data(args.data, args.window_size, args.minfreq, args.maxfreq, args.target_sampling_rate)
        inference_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        
        # Inference
        event_indices = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(inference_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                predicted = (probabilities >= args.threshold).cpu().numpy()
                batch_start_idx = batch_idx * inference_loader.batch_size
                for i, pred in enumerate(predicted):
                    if pred == 1:
                        idx = batch_start_idx + i
                        event_indices.append(idx)
        
        # Post-processing and saving results
        merged_events = merge_event_windows(event_indices, args.window_size)
        detections = []
        for event in merged_events:
            event_start_idx = event[0]
            event_end_idx = event[1]
            # Find the corresponding file and trace
            cumulative_length = 0
            for idx, info in enumerate(dataset.data_info_list):
                data_length = len(info['data']) - args.window_size
                if event_start_idx < cumulative_length + data_length:
                    data_idx = idx
                    local_start_idx = event_start_idx - cumulative_length
                    local_end_idx = event_end_idx - cumulative_length
                    tr = info['tr']
                    tr_filtered = info['tr_filtered']
                    cft = info['cft']
                    mseed_file = info['filename']
                    sampling_rate = info['sampling_rate']
                    break
                cumulative_length += data_length
            else:
                continue  # If corresponding file not found, skip
            
            event_start_time = tr.stats.starttime + local_start_idx * tr.stats.delta
            event_end_time = tr.stats.starttime + local_end_idx * tr.stats.delta
            duration = event_end_time - event_start_time
            amplitude = np.max(np.abs(tr.data[local_start_idx:local_end_idx]))
            detection = {
                'filename': os.path.basename(mseed_file),
                'start_time': event_start_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                'end_time': event_end_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                'duration': duration,
                'amplitude': amplitude
            }
            detections.append(detection)
        
        # Save results
        detections_df = pd.DataFrame(detections)
        detections_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
        
        # Save plots
        if args.save_plots:
            os.makedirs(args.plots_dir, exist_ok=True)
            for idx, info in enumerate(dataset.data_info_list):
                tr = info['tr']
                tr_filtered = info['tr_filtered']
                cft = info['cft']
                mseed_file = info['filename']
                filename = os.path.basename(mseed_file)
                plot_file = os.path.join(args.plots_dir, f"{filename}.png")
                
                # Plotting
                fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
                
                # Original seismic signal
                axs[0].plot(tr.times(), tr.data, 'k')
                axs[0].set_title('Original Seismic Signal')
                axs[0].set_ylabel('Amplitude')
                
                # Filtered seismic signal
                axs[1].plot(tr_filtered.times(), tr_filtered.data, 'b')
                axs[1].set_title('Filtered Seismic Signal (Bandpass)')
                axs[1].set_ylabel('Amplitude')
                
                # STA/LTA characteristic function
                axs[2].plot(tr_filtered.times(), cft, 'b')
                axs[2].hlines([2.5, 0.9], tr_filtered.times()[0], tr_filtered.times()[-1], colors=['r', 'g'], linestyles='--')
                axs[2].set_title('STA/LTA Characteristic Function')
                axs[2].set_ylabel('STA/LTA Ratio')
                
                # Spectrogram
                f, t_spec, Sxx = signal.spectrogram(tr_filtered.data, tr_filtered.stats.sampling_rate, nperseg=256, noverlap=128)
                im = axs[3].pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')
                axs[3].set_title('Spectrogram of Filtered Seismic Signal')
                axs[3].set_xlabel('Time (s)')
                axs[3].set_ylabel('Frequency (Hz)')
                cbar = plt.colorbar(im, ax=axs[3], orientation='vertical')
                cbar.set_label('Power (dB)')
                
                # Highlight detected events on the first plot
                for detection in detections:
                    if detection['filename'] == filename:
                        event_start_time = UTCDateTime(detection['start_time'])
                        event_end_time = UTCDateTime(detection['end_time'])
                        start = event_start_time - tr.stats.starttime
                        end = event_end_time - tr.stats.starttime
                        axs[0].axvspan(start, end, color='red', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plot_file)
                plt.close(fig)
                print(f"Plot saved to {plot_file}")

    else:
        print("Invalid mode. Use 'train', 'retrain', or 'infer'.")

if _name_ == '_main_':
    main()
