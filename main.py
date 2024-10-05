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

# Load seismic data from mseed file
mseed_file = 'D:/SpaceShift/XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed'
if not os.path.exists(mseed_file):
    raise FileNotFoundError(f"File not found: {mseed_file}")

st = read(mseed_file)
tr = st[0]  # Take the first trace

# Parameters for STA/LTA algorithm
sta_window = 5  # Short-term average window in seconds
lta_window = 50  # Long-term average window in seconds
threshold_on = 2.5  # Trigger threshold for event detection
threshold_off = 1.0  # Threshold for event ending

# Convert the STA/LTA window to number of samples
sampling_rate = tr.stats.sampling_rate
sta_samples = int(sta_window * sampling_rate)
lta_samples = int(lta_window * sampling_rate)

# Filter the trace to bring out particular frequencies (bandpass filter)
minfreq = 0.01
maxfreq = 0.5
tr_filtered = tr.copy()
tr_filtered.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)

# Apply the STA/LTA algorithm
cft = classic_sta_lta(tr_filtered.data, sta_samples, lta_samples)

# Find the onset and end times of events
onsets = trigger_onset(cft, threshold_on, threshold_off)

# Extract features for AI-based filtering
features = []
for onset in onsets:
    event_start_idx = onset[0]
    event_end_idx = onset[1]
    duration = (event_end_idx - event_start_idx) / sampling_rate
    amplitude = max(abs(tr_filtered.data[event_start_idx:event_end_idx]))
    energy = np.sum(tr_filtered.data[event_start_idx:event_end_idx] ** 2)
    features.append([duration, amplitude, energy])

# Check if features list is empty
if features:
    # Scale features for One-Class SVM
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train One-Class SVM to identify significant events
    svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    svm.fit(features_scaled)
    labels = svm.predict(features_scaled)

    # Determine which events are significant enough to send back to Earth
    significant_events = []
    for i, onset in enumerate(onsets):
        if labels[i] == 1:  # Label 1 indicates an inlier (significant event)
            event_start = tr.stats.starttime + timedelta(seconds=onset[0] / sampling_rate)
            event_end = tr.stats.starttime + timedelta(seconds=onset[1] / sampling_rate)
            duration = (onset[1] - onset[0]) / sampling_rate
            amplitude = max(abs(tr_filtered.data[onset[0]:onset[1]]))
            significant_events.append((event_start, event_end, duration, amplitude))

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

    # Print the significant events
    detections = []
    for event in significant_events:
        event_start, event_end, duration, amplitude = event
        print(f"Significant event detected from {event_start} to {event_end} with duration {duration:.2f} seconds and amplitude {amplitude:.2e}")
        detection = {
            'filename': os.path.basename(mseed_file),
            'time_abs(%Y-%m-%dT%H:%M:%S.%f)': event_start.strftime('%Y-%m-%dT%H:%M:%S.%f'),
            'time_rel(sec)': (event_start - tr.stats.starttime).total_seconds(),
            'duration(sec)': duration,
            'amplitude': amplitude
        }
        detections.append(detection)

    # Create a DataFrame and save to CSV
    output_csv = 'D:/SpaceShift/detected_events_catalog.csv'
    detections_df = pd.DataFrame(detections)
    detections_df.to_csv(output_csv, index=False)
    print(f"Significant events have been saved to {output_csv}")
else:
    print("No significant events detected.")