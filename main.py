import os
import sys
import argparse
import numpy as np
import pandas as pd
import obspy
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import butter, filtfilt
from scipy import signal
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings

# Подавляем предупреждения
warnings.filterwarnings("ignore", category=FutureWarning)

# Определение класса набора данных
class SeismicDataset(Dataset):
    def __init__(self, data_info_list, window_size=512):
        self.data_info_list = data_info_list
        self.window_size = window_size
        self.indices = []
        for idx, info in enumerate(self.data_info_list):
            data_length = len(info['data']) - window_size
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

# Определение модели SeismicCNN
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

# Функция для загрузки данных с дополнительными выводами
def load_data(data_path, window_size, minfreq, maxfreq, target_sampling_rate=None):
    data_info_list = []
    sampling_rates = []

    print("Начинается сбор списка файлов...")
    if os.path.isfile(data_path):
        mseed_files = [data_path]
    else:
        mseed_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.mseed'):
                    mseed_files.append(os.path.join(root, file))
    print(f"Найдено файлов: {len(mseed_files)}")

    if len(mseed_files) == 0:
        print("Ошибка: Не найдено файлов с расширением .mseed")
        sys.exit(1)

    # Сбор частот дискретизации
    for mseed_file in mseed_files:
        st = read(mseed_file)
        tr = st[0]
        sampling_rate = tr.stats.sampling_rate
        sampling_rates.append(sampling_rate)

    # Определение целевой частоты дискретизации
    sampling_rate_counter = Counter(sampling_rates)
    if target_sampling_rate is None:
        target_sampling_rate = sampling_rate_counter.most_common(1)[0][0]
        print(f"Определена целевая частота дискретизации: {target_sampling_rate} Hz")
    else:
        print(f"Используется заданная целевая частота дискретизации: {target_sampling_rate} Hz")

    data_info_list = []
    for idx, mseed_file in enumerate(mseed_files):
        print(f"Загрузка файла {idx+1}/{len(mseed_files)}: {mseed_file}")
        st = read(mseed_file)
        tr = st[0]
        original_sampling_rate = tr.stats.sampling_rate

        # Ресемплирование при необходимости
        if original_sampling_rate != target_sampling_rate:
            print(f"Ресемплирование с {original_sampling_rate} Hz до {target_sampling_rate} Hz")
            tr.resample(target_sampling_rate)

        sampling_rate = tr.stats.sampling_rate

        # Фильтрация сигнала
        tr_filtered = tr.copy()
        tr_filtered.data = butter_bandpass_filter(tr_filtered.data, minfreq, maxfreq, sampling_rate)

        # Параметры STA/LTA
        sta_window = 5  # секунд
        lta_window = 60  # секунд
        sta_samples = int(sta_window * sampling_rate)
        lta_samples = int(lta_window * sampling_rate)
        threshold_on = 2.5
        threshold_off = 0.9

        # Получение меток событий
        labels = classic_sta_lta_labels(tr_filtered, sta_samples, lta_samples, threshold_on, threshold_off)
        data = tr_filtered.data

        # Нормализация данных
        scaler = StandardScaler()
        data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

        data_info = {
            'data': data,
            'labels': labels,
            'filename': mseed_file,
            'tr': tr,  # Сохраняем оригинальный трейс
            'tr_filtered': tr_filtered,  # Сохраняем отфильтрованный трейс
            'cft': classic_sta_lta(tr_filtered.data, sta_samples, lta_samples),  # Сохраняем STA/LTA характеристическую функцию
            'sampling_rate': sampling_rate
        }

        data_info_list.append(data_info)
    print("Загрузка и предобработка данных завершены.")

    dataset = SeismicDataset(data_info_list, window_size)
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
    parser.add_argument('--save_plots', action='store_true', help='Сохранять графики в файлы PNG')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Директория для сохранения графиков')
    args = parser.parse_args()

    # Проверка наличия нескольких GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        print(f'Доступно GPU: {gpu_count}')
    else:
        device = torch.device('cpu')
        gpu_count = 0
        print('CUDA не доступна. Используется CPU.')

    model = SeismicCNN(window_size=args.window_size)

    # Если доступно несколько GPU, используем DataParallel
    if gpu_count > 1:
        print('Используется DataParallel для обучения на нескольких GPU.')
        model = nn.DataParallel(model)
    model.to(device)

    if args.mode == 'train':
        # Загружаем данные
        dataset, sampling_rate = load_data(args.data, args.window_size, args.minfreq, args.maxfreq, args.target_sampling_rate)
        # Устанавливаем num_workers в 0 или значение, соответствующее вашей системе
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # Обработка дисбаланса классов
        all_labels = [label for _, label in dataset]
        class_counts = np.bincount(all_labels)
        if len(class_counts) < 2:
            print("Ошибка: Недостаточно классов в данных для обучения.")
            sys.exit(1)
        class_weights = [len(all_labels) / class_counts[i] for i in range(len(class_counts))]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Обучение модели
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("Начинается обучение модели...")
        model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            for batch_idx, (inputs, labels_batch) in enumerate(dataloader):
                inputs = inputs.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                if batch_idx % 10 == 0:
                    print(f"Эпоха [{epoch+1}/{args.epochs}], Батч [{batch_idx}/{len(dataloader)}], Потеря: {loss.item():.4f}")
            epoch_loss = running_loss / len(dataloader.dataset)
            print(f"Эпоха [{epoch+1}/{args.epochs}] завершена, Средняя потеря: {epoch_loss:.4f}")

        # Сохраняем модель
        if gpu_count > 1:
            # Если используется DataParallel, сохраняем модель без обертки
            torch.save(model.module.state_dict(), args.model)
        else:
            torch.save(model.state_dict(), args.model)
        print(f"Модель сохранена в {args.model}")

    elif args.mode == 'infer':
        # Здесь ваш код для инференса (оставлен без изменений для краткости)
        pass

    else:
        print("Неверный режим работы. Используйте 'train', 'retrain' или 'infer'.")

if __name__ == '__main__':
    main()
