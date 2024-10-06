# Mars Seismic Event Detection & Analysis

This project focuses on detecting and analyzing significant seismic events from Mars using signal processing techniques and machine learning. The code reads seismic data, applies filtering, detects events using the STA/LTA algorithm, and leverages machine learning to classify and log significant events.

# Table of Contents
- Project Structure
- Requirements
- Setup Instructions
- Running the Project
- Workflow
- Event Detection
- Machine Learning
- Logging & Visualization
- Resources
- Troubleshooting
- Acknowledgements

# Project Structure
```
├── data                          # Directory for Mars seismic data (.mseed files)
│   └── XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.mseed
│
├── results                       # Output directory for detected events and anomalies
│   ├── detected_events_catalog.csv
│   └── historical_anomalies.csv
│
├── src                           # Source code files
│   ├── main.py                   # Main script for data processing and event detection
│   ├── filtering.py              # Functions for signal filtering (bandpass, Butterworth)
│   ├── event_detection.py        # STA/LTA algorithm and event detection functions
│   ├── feature_extraction.py     # Extracts features for machine learning
│   ├── anomaly_detection.py      # One-Class SVM for identifying significant events
│   ├── visualization.py          # Plotting and visualizing the seismic data
│   ├── data_logging.py           # Functions to log and compare events with historical data
│   └── utils.py                  # Utility functions for data handling and processing
│
└── README.md                     # This file (documentation)
```

# Requirements
Ensure you have the following dependencies installed:

- Python 3.9+
- ObsPy
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib

You can install the necessary packages using:

```bash
pip install -r requirements.txt
```

# Setup Instructions
1. Clone the repository:

```bash
git clone https://github.com/username/Mars-Seismic-Detection.git
cd Mars-Seismic-Detection
```

2. Place your seismic data files (`.mseed`) in the `data` directory.

# Running the Project
To run the project, you can use the following command:

```bash
python src/main.py --mode <mode> --data <data_path> [other_arguments]
```

# Command-Line Arguments:
- `--mode`: Specify the mode of operation. Choose from `train`, `retrain`, or `infer`. (required)
- `--data`: Path to the data file or directory containing seismic data. (required)
- `--model`: Path to the model file (default is `seismic_model.pth`).
- `--epochs`: Number of training epochs (default is `20`).
- `--batch_size`: Size of the batch for training (default is `64`).
- `--window_size`: Size of the window for processing (default is `512`).
- `--minfreq`: Minimum filter frequency (default is `0.5` Hz).
- `--maxfreq`: Maximum filter frequency (default is `20.0` Hz).
- `--output`: Path to the output CSV file for detected events (default is `detected_events.csv`).
- `--threshold`: Detection threshold for events (default is `0.5`).
- `--target_sampling_rate`: Target sampling rate for resampling the data (default is `20.0` Hz).
- `--save_plots`: Optional flag to save plots as PNG files.
- `--plots_dir`: Directory to save plots (default is `plots`).
- `--max_files`: Maximum number of files to process for training (optional).
- `--step`: Step size for generating samples in the dataset (default is `256`).
- `--learning_rate`: Learning rate for training (default is `0.001`).

# Workflow
The workflow includes several key steps:
1. **Data Preprocessing**: Raw seismic data is read from `.mseed` files and passed through bandpass and Butterworth filters to enhance the signal quality.
2. **Event Detection**: The STA/LTA algorithm identifies potential seismic events based on amplitude changes over time.
3. **Feature Extraction**: Key features (duration, amplitude, energy) are extracted from the detected events for machine learning.
4. **Machine Learning**: A One-Class SVM model is trained to classify events as significant or insignificant.
5. **Logging & Visualization**: Detected events are logged into a CSV file, and visualizations (waveform plots, spectrograms) are generated.

# Event Detection
- **STA/LTA Algorithm**: Detects seismic events by comparing the short-term average (STA) and long-term average (LTA) of the signal’s amplitude.
- **Thresholding**: Events are triggered when the STA/LTA ratio exceeds the defined thresholds.

# Machine Learning
- **Feature Scaling**: Extracted features are scaled using `StandardScaler` to prepare them for machine learning.
- **Anomaly Detection**: One-Class SVM is trained on extracted features to identify significant events.
- **Historical Comparison**: Detected events are compared with historical anomalies to find similar patterns using pairwise distances.

# Logging & Visualization
- **Event Catalog**: Detected significant events are saved in `detected_events_catalog.csv`.
- **Plotting**: The original seismic signal, filtered signal, STA/LTA ratio, and spectrogram are plotted to visualize the event detection process.

# Resources
- [ObsPy Documentation](https://docs.obspy.org)
- [SciPy Documentation](https://docs.scipy.org)

# Troubleshooting
1. **FileNotFoundError**: Ensure the seismic data file paths are correct and located in the `data` directory.
2. **No Events Detected**: Check the thresholds for the STA/LTA algorithm. Adjust the `threshold_on` and `threshold_off` parameters to improve detection sensitivity.
3. **Model Accuracy**: Experiment with different SVM parameters (`nu`, `kernel`, etc.) for better classification results.
4. **Performance Issues**: If the code is slow, reduce the size of the window for the STA/LTA algorithm or optimize your filtering steps.

Here's the updated section for running the project, including examples for inference, training, and fine-tuning:

---

# Running the Project
To run the project, you can use the following command:

```bash
python src/main.py --mode <mode> --data <data_path> [other_arguments]
```

# Command-Line Arguments:
- `--mode`: Specify the mode of operation. Choose from `train`, `retrain`, or `infer`. (required)
- `--data`: Path to the data file or directory containing seismic data. (required)
- `--model`: Path to the model file (default is `seismic_model.pth`).
- `--epochs`: Number of training epochs (default is `20`).
- `--batch_size`: Size of the batch for training (default is `64`).
- `--window_size`: Size of the window for processing (default is `512`).
- `--minfreq`: Minimum filter frequency (default is `0.5` Hz).
- `--maxfreq`: Maximum filter frequency (default is `20.0` Hz).
- `--output`: Path to the output CSV file for detected events (default is `detected_events.csv`).
- `--threshold`: Detection threshold for events (default is `0.5`).
- `--target_sampling_rate`: Target sampling rate for resampling the data (default is `20.0` Hz).
- `--save_plots`: Optional flag to save plots as PNG files.
- `--plots_dir`: Directory to save plots (default is `plots`).
- `--max_files`: Maximum number of files to process for training (optional).
- `--step`: Step size for generating samples in the dataset (default is `256`).
- `--learning_rate`: Learning rate for training (default is `0.001`).

# Examples

1. **Training a New Model**:
   To train a new model using a dataset located in `data/training_data`, use the following command:

   ```bash
   python src/main.py --mode train --data data/training_data --epochs 50 --batch_size 32 --output trained_model.pth
   ```

   In this example:
   - The model will be trained for 50 epochs.
   - The batch size is set to 32.
   - The trained model will be saved as `trained_model.pth`.

2. **Fine-Tuning an Existing Model**:
   To fine-tune an existing model (e.g., `pretrained_model.pth`) using new training data, use:

   ```bash
   python src/main.py --mode retrain --data data/new_training_data --model pretrained_model.pth --epochs 20 --batch_size 16 --output fine_tuned_model.pth
   ```

   In this example:
   - The existing model will be fine-tuned for 20 epochs.
   - The batch size is set to 16.
   - The updated model will be saved as `fine_tuned_model.pth`.

3. **Running Inference**:
   To perform inference on a new dataset located in `data/inference_data` using a trained model, run:

   ```bash
   python src/main.py --mode infer --data data/inference_data --model trained_model.pth --output detected_events.csv --threshold 0.6 --save_plots
   ```

   In this example:
   - The trained model (`trained_model.pth`) will be used for inference.
   - Detected events will be saved to `detected_events.csv`.
   - The threshold for detection is set to 0.6.
   - Plots will be saved as PNG files in the default directory.

# Workflow
The workflow includes several key steps:
1. **Data Preprocessing**: Raw seismic data is read from `.mseed` files and passed through bandpass and Butterworth filters to enhance the signal quality.
2. **Event Detection**: The STA/LTA algorithm identifies potential seismic events based on amplitude changes over time.
3. **Feature Extraction**: Key features (duration, amplitude, energy) are extracted from the detected events for machine learning.
4. **Machine Learning**: A One-Class SVM model is trained to classify events as significant or insignificant.
5. **Logging & Visualization**: Detected events are logged into a CSV file, and visualizations (waveform plots, spectrograms) are generated.

# Acknowledgements
This project is a part of ongoing research in Mars seismic event detection.
