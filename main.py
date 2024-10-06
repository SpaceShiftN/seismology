import os
import sys
import argparse
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import butter, filtfilt
from datetime import timedelta
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Определение класса набора данных
class SeismicDataset(Dataset):
    def __init__(self, data_list, label_list, window_size=512):
        self.data_list = data_list
        self.label_list = label_list
        self.window_size = window_size
        self.indices = []
        for idx, data in enumerate(self.data_list):
            data_length = len(data) - self.window_size
            for i in range(data_length):
                self.indices.append((idx, i))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx, offset = self.indices[idx]
        data = self.data_list[data_idx]
        labels = self.label_list[data_idx]
        x = data[offset:offset+self.window_size]
        y = labels[offset:offset+self.window_size]
        y = 1 if y.max() > 0 else 0
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Определение модели
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
        fc_input_size = (512 // 8) * 64
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
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
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
    data_list = []
    label_list = []
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
    
    # Определяем наиболее распространенную частоту дискретизации
    from collections import Counter
    sampling_rate_counter = Counter(sampling_rates)
    if target_sampling_rate is None:
        target_sampling_rate = sampling_rate_counter.most_common(1)[0][0]
        print(f"Определена целевая частота дискретизации: {target_sampling_rate} Hz")
    else:
        print(f"Используется заданная целевая частота дискретизации: {target_sampling_rate} Hz")
    
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
        
        labels = classic_sta_lta_labels(tr_filtered, sta_samples, lta_samples, threshold_on, threshold_off)
        data = tr_filtered.data
        
        data_list.append(data)
        label_list.append(labels)
    
    dataset = SeismicDataset(data_list, label_list, window_size)
    return dataset, target_sampling_rate

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
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используемое устройство: {device}')
    
    model = SeismicCNN().to(device)
    
    if args.mode == 'train':
        # Загружаем данные
        dataset, sampling_rate = load_data(args.data, args.window_size, args.minfreq, args.maxfreq, args.target_sampling_rate)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Обучение модели
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
            print(f"Эпоха [{epoch+1}/{args.epochs}], Потеря: {epoch_loss:.4f}")
        
        # Сохраняем модель
        torch.save(model.state_dict(), args.model)
        print(f"Модель сохранена в {args.model}")
    
    elif args.mode == 'retrain':
        # Загружаем существующую модель
        if not os.path.exists(args.model):
            print("Ошибка: Файл модели не найден.")
            sys.exit(1)
        model.load_state_dict(torch.load(args.model))
        
        # Загружаем данные
        dataset, sampling_rate = load_data(args.data, args.window_size, args.minfreq, args.maxfreq)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Дообучение модели
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
            print(f"Эпоха дообучения [{epoch+1}/{args.epochs}], Потеря: {epoch_loss:.4f}")
        
        # Сохраняем обновленную модель
        torch.save(model.state_dict(), args.model)
        print(f"Модель обновлена и сохранена в {args.model}")
    
    elif args.mode == 'infer':
        # Загружаем существующую модель
        if not os.path.exists(args.model):
            print("Ошибка: Файл модели не найден.")
            sys.exit(1)
        model.load_state_dict(torch.load(args.model))
        model.eval()
        
        # Загружаем данные
        dataset, sampling_rate = load_data(args.data, args.window_size, args.minfreq, args.maxfreq)
        inference_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Инференс
        event_windows = []
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(inference_loader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                predicted = (probabilities >= args.threshold).cpu().numpy()
                batch_start_idx = batch_idx * inference_loader.batch_size
                batch_indices = np.arange(batch_start_idx, batch_start_idx + len(predicted))
                event_indices = batch_indices[predicted == 1]
                event_windows.extend(event_indices.tolist())
        
        # Постобработка и сохранение результатов
        merged_events = merge_event_windows(event_windows, args.window_size)
        detections = []
        for event in merged_events:
            event_start_idx = event[0]
            event_end_idx = event[1]
            # Найти соответствующий файл и трейс
            cumulative_length = 0
            for idx, data in enumerate(dataset.data_list):
                data_length = len(data) - args.window_size
                if event_start_idx < cumulative_length + data_length:
                    data_idx = idx
                    local_start_idx = event_start_idx - cumulative_length
                    local_end_idx = event_end_idx - cumulative_length
                    tr = read(dataset.data_list[data_idx])[0]
                    break
                cumulative_length += data_length
            else:
                continue  # Если не нашли соответствующий файл, пропускаем
            
            event_start_time = tr.stats.starttime + local_start_idx * tr.stats.delta
            event_end_time = tr.stats.starttime + local_end_idx * tr.stats.delta
            duration = event_end_time - event_start_time
            amplitude = np.max(np.abs(tr.data[local_start_idx:local_end_idx]))
            detection = {
                'filename': os.path.basename(dataset.data_list[data_idx]),
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
    
    else:
        print("Неверный режим работы. Используйте 'train', 'retrain' или 'infer'.")

if __name__ == '__main__':
    main()
