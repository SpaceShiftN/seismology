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
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, bottleneck_channels=32, kernel_sizes=[9, 19, 39], use_residual=True):
        super(InceptionModule, self).__init__()
        self.use_residual = use_residual

        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False) if in_channels > 1 else nn.Identity()

        self.conv_layers = nn.ModuleList()
        for ks in kernel_sizes:
            padding = ks // 2
            self.conv_layers.append(
                nn.Conv1d(bottleneck_channels, 32, kernel_size=ks, padding=padding, bias=False)
            )

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, 32, kernel_size=1, bias=False)
        )

        self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        x = self.bottleneck(x)

        conv_outputs = [conv(x) for conv in self.conv_layers]
        conv_outputs.append(self.maxpool_conv(residual))

        x = torch.cat(conv_outputs, dim=1)
        x = self.batch_norm(x)
        x = self.relu(x)

        if self.use_residual:
            x += residual
            x = self.relu(x)

        return x

class InceptionTime(nn.Module):
    def __init__(self, num_blocks=6, in_channels=1, num_classes=2):
        super(InceptionTime, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(InceptionModule(in_channels if i == 0 else 128))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Добавляем размер канала
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x


# Определение класса набора данных
class SeismicDataset(Dataset):
    def __init__(self, data_info_list, window_size=512):
        """
        data_info_list: список словарей с ключами 'data', 'labels', 'filename'
        """
        self.data_info_list = data_info_list
        self.window_size = window_size
        self.indices = []
        for idx, info in enumerate(self.data_info_list):
            data_length = len(info['data']) - self.window_size
            for i in range(data_length):
                self.indices.append((idx, i))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx, offset = self.indices[idx]
        info = self.data_info_list[data_idx]
        data = info['data']
        labels = info['labels']
        x = data[offset:offset+self.window_size]
        y = labels[offset:offset+self.window_size]
        y = 1 if y.max() > 0 else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Определение модели
class SeismicCNN(nn.Module):
    def __init__(self, window_size=512):
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
        x = x.unsqueeze(1)  # Добавляем размер канала
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Разворачиваем тензор
        x = self.fc_layers(x)
        return x

# Функции для фильтрации и обработки данных
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

# Функция для загрузки данных из файла или директории
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
    
    # Сначала собираем все частоты дискретизации
    for mseed_file in mseed_files:
        st = read(mseed_file)
        tr = st[0]
        sampling_rate = tr.stats.sampling_rate
        sampling_rates.append(sampling_rate)
    
    # Определяем целевую частоту дискретизации
    sampling_rate_counter = Counter(sampling_rates)
    if target_sampling_rate is None:
        target_sampling_rate = sampling_rate_counter.most_common(1)[0][0]
        print(f"Определена целевая частота дискретизации: {target_sampling_rate} Hz")
    else:
        print(f"Используется заданная целевая частота дискретизации: {target_sampling_rate} Hz")
    
    data_info_list = []

    for mseed_file in mseed_files:
        st = read(mseed_file)
        tr = st[0]
        original_sampling_rate = tr.stats.sampling_rate
        
        # Ресемплирование при необходимости
        if original_sampling_rate != target_sampling_rate:
            tr.resample(target_sampling_rate)
            print(f"Файл {mseed_file} ресемплирован с {original_sampling_rate} Hz до {target_sampling_rate} Hz")
        
        sampling_rate = tr.stats.sampling_rate
        
        tr_filtered = tr.copy()
        tr_filtered.data = butter_bandpass_filter(tr_filtered.data, minfreq, maxfreq, sampling_rate)
        
        # Параметры STA/LTA
        sta_window = 5  # секунд
        lta_window = 60  # секунд
        sta_samples = int(sta_window * sampling_rate)
        lta_samples = int(lta_window * sampling_rate)
        threshold_on = 2.5
        threshold_off = 0.9

        # Вычисление STA/LTA характеристической функции
        cft = classic_sta_lta(tr_filtered.data, sta_samples, lta_samples)
        
        labels = classic_sta_lta_labels(tr_filtered, sta_samples, lta_samples, threshold_on, threshold_off)
        scaler = StandardScaler()
        data = tr_filtered.data
        data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        data_info = {
            'data': data,
            'labels': labels,
            'filename': mseed_file,
            'tr': tr,  # Сохраняем оригинальный трейс
            'tr_filtered': tr_filtered,  # Сохраняем отфильтрованный трейс
            'cft': cft,  # Сохраняем STA/LTA характеристическую функцию
            'sampling_rate': sampling_rate
        }
        
        data_info_list.append(data_info)
    
    dataset = SeismicDataset(data_info_list, window_size)
    return dataset, target_sampling_rate

# Функция для объединения окон событий
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

# Основная функция
def main():
    parser = argparse.ArgumentParser(description='Сейсмическая обработка данных')
    parser.add_argument('--mode', type=str, choices=['train', 'retrain', 'infer'], required=True, help='Режим работы: train, retrain или infer')
    parser.add_argument('--data', type=str, required=True, help='Путь к файлу или директории с данными')
    parser.add_argument('--model', type=str, default='seismic_model.pth', help='Путь к файлу модели')
    parser.add_argument('--epochs', type=int, default=20, help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=64, help='Размер батча')
    parser.add_argument('--window_size', type=int, default=512, help='Размер окна')
    parser.add_argument('--minfreq', type=float, default=0.01, help='Минимальная частота фильтра')
    parser.add_argument('--maxfreq', type=float, default=0.5, help='Максимальная частота фильтра')
    parser.add_argument('--output', type=str, default='detected_events.csv', help='Путь к выходному CSV файлу')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог для обнаружения событий')
    parser.add_argument('--target_sampling_rate', type=float, help='Целевая частота дискретизации для ресемплирования данных')
    parser.add_argument('--save_plots', action='store_true', help='Сохранять графики в файлы PNG')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Директория для сохранения графиков')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используемое устройство: {device}')
    
    model = SeismicCNN(window_size=args.window_size).to(device)
    
    if args.mode == 'train':
        # Загружаем данные
        dataset, sampling_rate = load_data(args.data, args.window_size, args.minfreq, args.maxfreq, args.target_sampling_rate)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        # Получение веса классов
        labels = []
        for info in dataset.data_info_list:
            labels.extend(info['labels'])

        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Обучение модели
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        scaler = torch.amp.GradScaler('cuda')

        
        model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            for inputs, labels_batch in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloader.dataset)
            print(f"Эпоха [{epoch+1}/{args.epochs}], Потеря: {epoch_loss:.4f}")
        
        # Сохраняем модель
        torch.save(model.state_dict(), args.model)
        print(f"Модель сохранена в {args.model}")
    
    elif args.mode == 'infer':
        # Загружаем существующую модель
        if not os.path.exists(args.model):
            print("Ошибка: Файл модели не найден.")
            sys.exit(1)
        model.load_state_dict(torch.load(args.model))
        model.eval()
        
        # Загружаем данные
        dataset, sampling_rate = load_data(args.data, args.window_size, args.minfreq, args.maxfreq, args.target_sampling_rate)
        inference_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
        
        # Инференс
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
        
        # Постобработка и сохранение результатов
        merged_events = merge_event_windows(event_indices, args.window_size)
        detections = []
        for event in merged_events:
            event_start_idx = event[0]
            event_end_idx = event[1]
            # Найти соответствующий файл и трейс
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
                continue  # Если не нашли соответствующий файл, пропускаем
            
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
        
        # Сохраняем результаты
        detections_df = pd.DataFrame(detections)
        detections_df.to_csv(args.output, index=False)
        print(f"Результаты сохранены в {args.output}")
        
        # Сохранение графиков
        if args.save_plots:
            os.makedirs(args.plots_dir, exist_ok=True)
            for idx, info in enumerate(dataset.data_info_list):
                tr = info['tr']
                tr_filtered = info['tr_filtered']
                cft = info['cft']
                mseed_file = info['filename']
                filename = os.path.basename(mseed_file)
                plot_file = os.path.join(args.plots_dir, f"{filename}.png")
                
                # Построение графика
                fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
                
                # Оригинальный сейсмический сигнал
                axs[0].plot(tr.times(), tr.data, 'k')
                axs[0].set_title('Original Seismic Signal')
                axs[0].set_ylabel('Amplitude')
                
                # Отфильтрованный сейсмический сигнал
                axs[1].plot(tr_filtered.times(), tr_filtered.data, 'b')
                axs[1].set_title('Filtered Seismic Signal (Bandpass)')
                axs[1].set_ylabel('Amplitude')
                
                # Характеристическая функция STA/LTA
                axs[2].plot(tr_filtered.times(), cft, 'b')
                axs[2].hlines([2.5, 0.9], tr_filtered.times()[0], tr_filtered.times()[-1], colors=['r', 'g'], linestyles='--')
                axs[2].set_title('STA/LTA Characteristic Function')
                axs[2].set_ylabel('STA/LTA Ratio')
                
                # Спектрограмма
                f, t_spec, Sxx = signal.spectrogram(tr_filtered.data, tr_filtered.stats.sampling_rate, nperseg=256, noverlap=128)
                im = axs[3].pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')
                axs[3].set_title('Spectrogram of Filtered Seismic Signal')
                axs[3].set_xlabel('Time (s)')
                axs[3].set_ylabel('Frequency (Hz)')
                cbar = plt.colorbar(im, ax=axs[3], orientation='vertical')
                cbar.set_label('Power (dB)')
                
                # Отмечаем обнаруженные события на первом графике
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
                print(f"График сохранен в {plot_file}")
    
    else:
        print("Неверный режим работы. Используйте 'train', 'retrain' или 'infer'.")
    
if __name__ == '__main__':
    main()