import os
import sys
import argparse
import numpy as np
import pandas as pd
import obspy
from obspy import read, UTCDateTime
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
import time

# Подавляем предупреждения
warnings.filterwarnings("ignore", category=FutureWarning)

# Определение класса набора данных с параметром step
class SeismicDataset(Dataset):
    def __init__(self, data_info_list, window_size=512, step=256):
        self.data_info_list = data_info_list
        self.window_size = window_size
        self.step = step
        self.indices = []
        for idx, info in enumerate(self.data_info_list):
            data_length = len(info['data']) - window_size
            if data_length <= 0:
                print(f"Предупреждение: data_length <= 0 для файла {info['filename']}")
                continue
            for i in range(0, data_length, self.step):
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

# Определение модели SeismicCNN с Dropout
class SeismicCNN(nn.Module):
    def __init__(self, window_size=512):
        super(SeismicCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),  # Добавлен слой Dropout
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2),  # Добавлен слой Dropout
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2)  # Добавлен слой Dropout
        )
        fc_input_size = (window_size // 8) * 64
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Добавлен слой Dropout для регуляризации
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
def merge_event_windows(event_indices, window_size, step):
    if not event_indices:
        return []
    events = []
    current_event = [event_indices[0], event_indices[0] + window_size]
    for idx in event_indices[1:]:
        if idx <= current_event[1]:
            current_event[1] = idx + window_size
        else:
            events.append(current_event)
            current_event = [idx, idx + window_size]
    events.append(current_event)
    return events

# Функция для загрузки данных с возможностью ограничения количества файлов
def load_data(data_path, window_size, minfreq, maxfreq, target_sampling_rate=None, max_files=None, step=256):
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

    # Ограничиваем количество файлов, если указано
    if max_files is not None:
        mseed_files = mseed_files[:max_files]
        print(f"Используются первые {max_files} файлов для обучения.")

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
            'tr': tr.copy(),  # Сохраняем оригинальный трейс
            'tr_filtered': tr_filtered.copy(),  # Сохраняем отфильтрованный трейс
            'cft': classic_sta_lta(tr_filtered.data, sta_samples, lta_samples),  # Сохраняем STA/LTA характеристическую функцию
            'sampling_rate': sampling_rate
        }

        data_info_list.append(data_info)
    print("Загрузка и предобработка данных завершены.")

    dataset = SeismicDataset(data_info_list, window_size, step=step)
    print(f"Размер датасета: {len(dataset)} выборок")
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
    parser.add_argument('--minfreq', type=float, default=0.5, help='Минимальная частота фильтра')
    parser.add_argument('--maxfreq', type=float, default=20.0, help='Максимальная частота фильтра')
    parser.add_argument('--output', type=str, default='detected_events.csv', help='Путь к выходному CSV файлу')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог для обнаружения событий')
    parser.add_argument('--target_sampling_rate', type=float, default=20.0, help='Целевая частота дискретизации для ресемплирования данных')
    parser.add_argument('--save_plots', action='store_true', help='Сохранять графики в файлы PNG')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Директория для сохранения графиков')
    parser.add_argument('--max_files', type=int, help='Максимальное количество файлов для обучения')
    parser.add_argument('--step', type=int, default=256, help='Шаг для генерации выборок в датасете')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Скорость обучения')
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

    if args.mode == 'train' or args.mode == 'retrain':
        if args.mode == 'retrain':
            # Загрузка сохраненной модели
            if not os.path.exists(args.model):
                print("Ошибка: Файл модели не найден.")
                sys.exit(1)
            state_dict = torch.load(args.model)
            if gpu_count > 1:
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            print(f"Модель {args.model} загружена для дообучения.")

        # Загружаем данные
        dataset, sampling_rate = load_data(
            args.data,
            args.window_size,
            args.minfreq,
            args.maxfreq,
            args.target_sampling_rate,
            max_files=args.max_files,
            step=args.step
        )
        # Устанавливаем num_workers в 0 или значение, соответствующее вашей системе
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

        # Обработка дисбаланса классов
        all_labels = [label.item() for _, label in dataset]
        class_counts = np.bincount(all_labels)
        if len(class_counts) < 2:
            print("Ошибка: Недостаточно классов в данных для обучения.")
            sys.exit(1)
        class_weights = [len(all_labels) / class_counts[i] for i in range(len(class_counts))]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Обучение модели
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        print("Начинается обучение модели...")
        model.train()
        for epoch in range(args.epochs):
            running_loss = 0.0
            start_time = time.time()
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
            end_time = time.time()
            print(f"Эпоха [{epoch+1}/{args.epochs}] завершена, Средняя потеря: {epoch_loss:.4f}, Время эпохи: {end_time - start_time:.2f} секунд")

        # Сохраняем модель
        if gpu_count > 1:
            # Если используется DataParallel, сохраняем модель без обертки
            torch.save(model.module.state_dict(), args.model)
        else:
            torch.save(model.state_dict(), args.model)
        print(f"Модель сохранена в {args.model}")

    elif args.mode == 'infer':
        # Интегрируем код для инференса из предыдущего кода
        # Загрузка сохраненной модели
        if not os.path.exists(args.model):
            print("Ошибка: Файл модели не найден.")
            sys.exit(1)
        state_dict = torch.load(args.model, map_location=device)
        if gpu_count > 1:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        model.eval()
        print(f"Модель {args.model} загружена.")

        # Загружаем данные
        dataset, sampling_rate = load_data(
            args.data,
            args.window_size,
            args.minfreq,
            args.maxfreq,
            args.target_sampling_rate,
            max_files=args.max_files,
            step=args.step
        )
        inference_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

        # Инференс
        print("Начинается инференс...")
        event_indices = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(inference_loader):
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                predicted = (probabilities >= args.threshold).cpu().numpy()
                batch_start_idx = batch_idx * inference_loader.batch_size
                for i, pred in enumerate(predicted):
                    if pred == 1:
                        idx = batch_start_idx + i
                        event_indices.append(idx)

        # Постобработка и сохранение результатов
        merged_events = merge_event_windows(event_indices, args.window_size, args.step)
        detections = []
        cumulative_lengths = [0]
        for info in dataset.data_info_list:
            data_length = len(info['data']) - args.window_size
            cumulative_lengths.append(cumulative_lengths[-1] + (data_length // args.step))

        for event in merged_events:
            event_start_idx = event[0]
            event_end_idx = event[1]

            # Найти соответствующий файл и трейс
            data_idx = None
            for i in range(len(cumulative_lengths) - 1):
                if cumulative_lengths[i] <= event_start_idx < cumulative_lengths[i + 1]:
                    data_idx = i
                    local_start_idx = (event_start_idx - cumulative_lengths[i]) * args.step
                    local_end_idx = (event_end_idx - cumulative_lengths[i]) * args.step + args.window_size
                    info = dataset.data_info_list[data_idx]
                    tr = info['tr']
                    mseed_file = info['filename']
                    break
            if data_idx is None:
                continue  # Если не нашли соответствующий файл, пропускаем

            event_start_time = tr.stats.starttime + local_start_idx / tr.stats.sampling_rate
            event_end_time = tr.stats.starttime + local_end_idx / tr.stats.sampling_rate
            duration = event_end_time - event_start_time
            amplitude = np.max(np.abs(tr.data[int(local_start_idx):int(local_end_idx)]))
            detection = {
                'filename': os.path.basename(mseed_file),
                'start_time': event_start_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                'end_time': event_end_time.strftime('%Y-%m-%dT%H:%M:%S.%f'),
                'duration': duration,
                'amplitude': amplitude
            }
            detections.append(detection)

        # Сохраняем результаты
        if detections:
            detections_df = pd.DataFrame(detections)
            detections_df.to_csv(args.output, index=False)
            print(f"Результаты сохранены в {args.output}")
        else:
            print("Не обнаружено событий.")

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
                cft_times = tr_filtered.times()[len(tr_filtered.times()) - len(cft):]
                axs[2].plot(cft_times, cft, 'b')
                axs[2].hlines([2.5, 0.9], cft_times[0], cft_times[-1], colors=['r', 'g'], linestyles='--')
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

                # Отмечаем обнаруженные события на графиках
                for detection in detections:
                    if detection['filename'] == filename:
                        event_start_time = UTCDateTime(detection['start_time'])
                        event_end_time = UTCDateTime(detection['end_time'])
                        start = event_start_time - tr.stats.starttime
                        end = event_end_time - tr.stats.starttime
                        axs[0].axvspan(start, end, color='red', alpha=0.3)
                        axs[1].axvspan(start, end, color='red', alpha=0.3)
                        axs[2].axvspan(start, end, color='red', alpha=0.3)
                        axs[3].axvspan(start, end, color='red', alpha=0.3)

                plt.tight_layout()
                plt.savefig(plot_file)
                plt.close(fig)
                print(f"График сохранен в {plot_file}")

    else:
        print("Неверный режим работы. Используйте 'train', 'retrain' или 'infer'.")

if __name__ == '__main__':
    main()
