import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.fftpack import fft, dct  # Import dct directly
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load the seismic data
data = pd.read_csv('xa.s12.00.mhz.1969-12-16HR00_evid00006.csv')

# Extract relevant columns
time_abs = pd.to_datetime(data['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])  # Absolute time in datetime format
time_rel = data['time_rel']  # Relative time in seconds
velocity = data['velocity']  # Seismic velocity in m/s

# Sliding window parameters
global_window_size = 128  # Global window size (corresponds to a 20s window)
global_step_size = 1  # Global 1 second shift


# Function to extract sliding windows
def sliding_window(signal, win_size, step):
    """Extract sliding windows from signal."""
    return np.lib.stride_tricks.sliding_window_view(signal, win_size)[::step]


# Apply sliding window to the velocity data
windows = sliding_window(velocity, global_window_size, global_step_size)


# Function to compute features from each velocity window
def extract_features(window):
    """Compute relevant features from each window."""
    # Envelope using Hilbert transform
    envelope = np.abs(hilbert(window))

    # Central frequency using Fourier Transform
    spectrum = np.abs(fft(window))
    central_freq = np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)

    # Cepstral coefficients using DCT
    cepstrum = np.abs(dct(window, type=2, norm='ortho'))

    # Spectral attributes: instantaneous frequency and bandwidth
    instantaneous_freq = np.diff(np.angle(hilbert(window)))
    bandwidth = np.std(instantaneous_freq)

    # Return feature vector: envelope stats, central frequency, bandwidth, and 3 cepstral coefficients
    return np.array([np.mean(envelope), np.std(envelope), central_freq, bandwidth, *cepstrum[:3]])


# Apply feature extraction to all velocity windows
features = np.array([extract_features(w) for w in windows])

# Normalize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Gaussian Mixture Model (GMM) for event classification
gmm = GaussianMixture(n_components=3, covariance_type='full')  # Assume 3 classes
gmm.fit(features_scaled)

# Classify each window
labels = gmm.predict(features_scaled)

# Get log likelihood for filtering purposes
log_likelihood = gmm.score_samples(features_scaled)

# Create a DataFrame with time, labels, and log likelihood
results = pd.DataFrame({
    'Time_abs': time_abs[global_window_size - 1::global_step_size],  # Adjust time for window size
    'Label': labels,
    'Log_Likelihood': log_likelihood
})

# Filter based on log likelihood threshold
filtered_results = results[(log_likelihood > 1)]  # You can adjust the threshold here


# Further filtering by event length and time separation
def filter_events(events, min_length=120, min_time_between=80 * 60):  # Min length: 2 mins, min separation: 80 mins
    filtered = []
    last_event_time = None
    for idx, event in events.iterrows():
        # Calculate event duration based on the number of windows and apply the minimum event length
        event_duration = min_length  # Placeholder for actual duration calculation if needed
        if last_event_time is None or (event['Time_abs'] - last_event_time).total_seconds() > min_time_between:
            if event_duration >= min_length:  # Apply the min_length condition here
                filtered.append(event)
            last_event_time = event['Time_abs']
    return pd.DataFrame(filtered)


# Apply event filtering
final_events = filter_events(filtered_results)

# Save the final filtered events to a CSV file
final_events.to_csv('/mnt/data/quake_classification_filtered.csv', index=False)

# Output final results
print("Classification completed and saved to quake_classification_filtered.csv.")
