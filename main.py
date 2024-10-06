import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy import signal
from datetime import timedelta
from scipy.signal import butter, filtfilt
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import pairwise_distances
import os
from pathlib import Path

# Load seismic data from mseed file
script_dir = Path(__file__).parent
mseed_file = script_dir / 'XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed'
if not os.path.exists(mseed_file):
    raise FileNotFoundError(f"File not found: {mseed_file}")

st = read(mseed_file)
tr = st[0]  # Take the first trace

# Parameters for STA/LTA algorithm
sta_window = 5  # Short-term average window in seconds
lta_window = 50  # Long-term average window in seconds

# Convert the STA/LTA window to number of samples
sampling_rate = tr.stats.sampling_rate
sta_samples = int(sta_window * sampling_rate)
lta_samples = int(lta_window * sampling_rate)

# Filter the trace to bring out particular frequencies (bandpass filter)
minfreq = 0.01
maxfreq = 0.5


# Butterworth filter function
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

# Dynamic threshold calculation based on the STA/LTA characteristics
threshold_on = np.mean(cft) + 2 * np.std(cft)  # Dynamic "on" threshold
threshold_off = np.mean(cft) + np.std(cft)  # Dynamic "off" threshold

# Find the onset and end times of events
onsets = trigger_onset(cft, threshold_on, threshold_off)

# Extract features for anomaly detection
features = []
for onset in onsets:
    event_start_idx = onset[0]
    event_end_idx = onset[1]
    duration = (event_end_idx - event_start_idx) / sampling_rate
    amplitude = max(abs(tr_filtered.data[event_start_idx:event_end_idx]))
    energy = np.sum(tr_filtered.data[event_start_idx:event_end_idx] ** 2)
    features.append([duration, amplitude, energy])

# Scale features using RobustScaler
if features:
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)

    # Initialize significant_events to store detected significant events
    significant_events = []

    # Use Mahalanobis distance for anomaly detection
    robust_cov = MinCovDet().fit(features_scaled)
    mahal_dist = robust_cov.mahalanobis(features_scaled)

    # Anomalies based on Mahalanobis distance threshold
    threshold_mahal = np.percentile(mahal_dist, 95)  # 95th percentile
    anomalies = mahal_dist > threshold_mahal

    # Identify significant events
    for i, onset in enumerate(onsets):
        if anomalies[i]:  # Event is considered significant if Mahalanobis distance is high
            event_start = tr.stats.starttime + timedelta(seconds=onset[0] / sampling_rate)
            event_end = tr.stats.starttime + timedelta(seconds=onset[1] / sampling_rate)
            duration = (onset[1] - onset[0]) / sampling_rate
            amplitude = max(abs(tr_filtered.data[onset[0]:onset[1]]))
            significant_events.append((event_start, event_end, duration, amplitude))

    # Ensure significant_events exists before using it in the plot
    if significant_events:
        # Plot the results with event highlights
        fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

        # Original Seismic Signal
        axs[0].plot(tr.times(), tr.data, 'k')
        axs[0].set_title('Original Seismic Signal')
        axs[0].set_ylabel('Amplitude')

        # Highlight detected events on the original signal
        for event in significant_events:
            # event[0] and event[1] are datetime objects; subtracting start time directly
            event_start_idx = int(
                (event[0] - tr.stats.starttime) * sampling_rate)  # Convert event start time to sample index
            event_end_idx = int(
                (event[1] - tr.stats.starttime) * sampling_rate)  # Convert event end time to sample index
            axs[0].axvspan(tr.times()[event_start_idx], tr.times()[event_end_idx], color='red', alpha=0.3)

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
    else:
        print("No significant events detected.")

    # Save significant events to CSV
    detections = []
    for event in significant_events:
        event_start, event_end, duration, amplitude = event
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
