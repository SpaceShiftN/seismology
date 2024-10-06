import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks, butter, filtfilt
from scipy.fftpack import fft, dct
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('xa.s12.00.mhz.1969-12-16HR00_evid00006.csv')

# Извлечение колонок
time_abs = pd.to_datetime(data['time_abs(%Y-%m-%dT%H:%M:%S.%f)'])  # Абсолютное время
time_rel = data['time_rel']  # Относительное время в секундах
velocity = data['velocity']  # Скорость в м/с

# Параметры окна
global_window_size = 128  # Размер окна
global_step_size = 1  # Шаг окна в секундах

# Функция для извлечения скользящих окон
def sliding_window(signal, win_size, step):
    """Извлечение скользящих окон из сигнала."""
    return np.lib.stride_tricks.sliding_window_view(signal, win_size)[::step]

# Применение скользящего окна к данным скорости
windows = sliding_window(velocity, global_window_size, global_step_size)

# Функция для извлечения признаков из окна скорости
def extract_features(window):
    """Вычисление признаков из каждого окна."""
    # Конвертирование сигнала в огибающую с помощью преобразования Хилберта
    envelope = np.abs(hilbert(window))

    # Центральная частота с использованием преобразования Фурье
    spectrum = np.abs(fft(window))
    central_freq = np.sum(np.arange(len(spectrum)) * spectrum) / np.sum(spectrum)

    # Кепстральные коэффициенты с использованием DCT
    cepstrum = np.abs(dct(window, type=2, norm='ortho'))

    # Спектральные атрибуты: мгновенная частота и ширина полосы
    instantaneous_freq = np.diff(np.angle(hilbert(window)))
    bandwidth = np.std(instantaneous_freq)

    # Возвращаем вектор признаков: среднее и стандартное отклонение огибающей, центральная частота, ширина полосы и 3 кепстральных коэффициента
    return np.array([np.mean(envelope), np.std(envelope), central_freq, bandwidth, *cepstrum[:3]])

# Извлечение признаков из всех окон скорости
features = np.array([extract_features(w) for w in windows])

# Нормализация признаков
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Модель Gaussian Mixture для классификации событий
gmm = GaussianMixture(n_components=3, covariance_type='full')  # Предполагаем 3 класса
gmm.fit(features_scaled)

# Классификация каждого окна
labels = gmm.predict(features_scaled)

# Получение логарифма правдоподобия для фильтрации
log_likelihood = gmm.score_samples(features_scaled)

# Создание DataFrame с временем, метками и логарифмом правдоподобия
results = pd.DataFrame({
    'Time_abs': time_abs[global_window_size - 1::global_step_size],  # Корректировка времени для размера окна
    'Label': labels,
    'Log_Likelihood': log_likelihood
})

# Фильтрация на основе порога логарифма правдоподобия
filtered_results = results[(log_likelihood > 1)]  # Порог можно настроить

# Функция для обнаружения P и S волн с использованием STA/LTA
def detect_p_s_waves(signal, fs, sta_window=1.0, lta_window=10.0, threshold_p=3, threshold_s=1.5):
    """
    Обнаружение P и S волн с использованием алгоритма STA/LTA.
    Возвращает индексы прихода P и S волн.
    """
    n_sta = int(sta_window * fs)
    n_lta = int(lta_window * fs)
    
    # Квадрат сигнала для расчета STA/LTA
    squared_signal = signal ** 2
    sta = np.convolve(squared_signal, np.ones(n_sta)/n_sta, mode='same')
    lta = np.convolve(squared_signal, np.ones(n_lta)/n_lta, mode='same')
    
    # Избежание деления на ноль
    lta[lta == 0] = 1e-10
    ratio = sta / lta
    
    # Обнаружение пиков для P волн
    peaks_p, _ = find_peaks(ratio, height=threshold_p)
    if len(peaks_p) > 0:
        p_wave_idx = peaks_p[0]
    else:
        p_wave_idx = None
    
    # Обнаружение пиков для S волн после P волны
    if p_wave_idx is not None:
        peaks_s, _ = find_peaks(ratio[p_wave_idx:], height=threshold_s)
        if len(peaks_s) > 0:
            s_wave_idx = p_wave_idx + peaks_s[0]
        else:
            s_wave_idx = None
    else:
        s_wave_idx = None
    
    return p_wave_idx, s_wave_idx

# Параметры волновых скоростей (в км/с)
V_p = 6.0  # Скорость P-волн
V_s = 3.5  # Скорость S-волн

# Функция для расчета расстояния до эпицентра
def calculate_distance(delta_t, V_p, V_s):
    """
    Вычисление расстояния до эпицентра на основе разницы во времени прихода S и P волн.
    """
    return delta_t * V_s * V_p / (V_p - V_s)

# Частота дискретизации (предполагается равномерная)
delta_t = np.median(np.diff(time_rel))  # Время между сэмплами
fs = 1 / delta_t  # Частота дискретизации

# Применение фильтрации (опционально, для улучшения обнаружения волн)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Применение полосового фильтра
lowcut = 1.0
highcut = 20.0
b, a = butter_bandpass(lowcut, highcut, fs, order=4)
filtered_velocity = filtfilt(b, a, velocity)

# Обнаружение P и S волн в отфильтрованных данных
p_idx, s_idx = detect_p_s_waves(filtered_velocity, fs)

# Проверка наличия обнаруженных P и S волн
if p_idx is not None and s_idx is not None:
    # Вычисление разницы во времени прихода
    delta_t_seconds = (time_rel[s_idx] - time_rel[p_idx])
    
    # Вычисление расстояния до эпицентра
    distance_km = calculate_distance(delta_t_seconds, V_p, V_s)
    print(f"Оцененное расстояние до эпицентра: {distance_km:.2f} км")
    
    # Добавление результатов в DataFrame
    results['P_Arrival'] = np.nan
    results['S_Arrival'] = np.nan
    results.loc[0, 'P_Arrival'] = time_abs[p_idx]
    results.loc[0, 'S_Arrival'] = time_abs[s_idx]
    results['Distance_km'] = np.nan
    results.loc[0, 'Distance_km'] = distance_km
else:
    print("Не удалось обнаружить как P, так и S волны.")

# Сохранение результатов в CSV
results.to_csv('/mnt/data/quake_classification_filtered_with_distance.csv', index=False)

# Визуализация обнаруженных P и S волн
plt.figure(figsize=(15, 5))
plt.plot(time_rel, filtered_velocity, label='Отфильтрованная скорость (м/с)')
if p_idx is not None:
    plt.axvline(x=time_rel[p_idx], color='g', linestyle='--', label='P волна')
if s_idx is not None:
    plt.axvline(x=time_rel[s_idx], color='r', linestyle='--', label='S волна')
plt.xlabel('Время (секунды)')
plt.ylabel('Скорость (м/с)')
plt.legend()
plt.title('Обнаружение P и S волн')
plt.show()

# Вывод финальных результатов
print("Классификация завершена и сохранена в quake_classification_filtered_with_distance.csv.")
