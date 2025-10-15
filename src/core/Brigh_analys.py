import os
import cv2
import numpy as np
from scipy import stats
from scipy.signal import find_peaks


def _read_image_as_gray(image_path: str) -> np.ndarray:
    """
    Читает изображение по пути и возвращает grayscale массив uint8.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def calculate_brightness_stats(image_gray: np.ndarray) -> dict:
    """
    Возвращает статистики яркости для одного grayscale-изображения.
    """
    return {
        'mean_brightness': float(np.mean(image_gray)),
        'std_brightness': float(np.std(image_gray)),
    }


def analyze_histogram_peaks(image_gray: np.ndarray, min_distance: int = 10, min_height: int = 50) -> dict:
    """
    Анализ пиков гистограммы яркости.
    """
    hist, bin_edges = np.histogram(image_gray.flatten(), bins=256, range=[0, 255])
    peaks, properties = find_peaks(hist, distance=min_distance, height=min_height)
    return {
        'histogram': hist.tolist(),
        'peak_positions': peaks.tolist(),
        'peak_heights': properties.get('peak_heights', []).tolist() if 'peak_heights' in properties else [],
        'peak_count': int(len(peaks)),
    }


def analyze_image_brightness(image_path: str) -> dict:
    """
    Комплексный анализ яркости изображения по пути.
    Возвращает словарь, пригодный для агрегации в эталонной модели.
    """
    gray = _read_image_as_gray(image_path)
    stats_dict = calculate_brightness_stats(gray)
    peaks_dict = analyze_histogram_peaks(gray)
    return {
        **stats_dict,
        'peak_count': peaks_dict['peak_count'],
        'peak_positions': peaks_dict['peak_positions'],
    }


def calculate_batch_brightness_stats(folder_path: str) -> list:
    """
    Считает статистики яркости для всех изображений в папке.
    Возвращает список словарей со статистиками по каждому файлу.
    Поддерживаемые расширения: jpg, jpeg, png, bmp.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Папка {folder_path} не существует")

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    results = []
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        try:
            stats_item = analyze_image_brightness(image_path)
            stats_item['image'] = filename
            results.append(stats_item)
        except Exception as e:
            # Пропускаем проблемные файлы, чтобы не прерывать процесс пакетной обработки
            continue

    return results


def calculate_skewness(image_gray: np.ndarray) -> float:
    """
    Вычисляет асимметрию распределения яркости (скошенность) для grayscale-изображения.
    """
    pixels = image_gray.flatten()
    return float(stats.skew(pixels))


def calculate_kurtosis(image_gray: np.ndarray) -> float:
    """
    Вычисляет эксцесс распределения яркости для grayscale-изображения.
    """
    pixels = image_gray.flatten()
    return float(stats.kurtosis(pixels))

