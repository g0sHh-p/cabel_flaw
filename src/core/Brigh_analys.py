import os
import cv2
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter


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
    hist, bin_edges = np.histogram(image_gray.flatten(), bins=256, range=[0, 256])
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


# Функции нормализации освещения

def normalize_lighting_clahe(image_gray: np.ndarray, clip_limit: float = 2.0, 
                           tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Нормализация освещения с помощью CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image_gray: Входное изображение в градациях серого
        clip_limit: Предел обрезания контраста (по умолчанию 2.0)
        tile_grid_size: Размер сетки тайлов для CLAHE
    
    Returns:
        Нормализованное изображение
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    normalized = clahe.apply(image_gray)
    return normalized


def background_correction(image_gray: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """
    Коррекция неравномерного освещения путем вычитания сглаженного фона.
    
    Args:
        image_gray: Входное изображение в градациях серого
        sigma: Стандартное отклонение для гауссова фильтра фона
    
    Returns:
        Изображение с выровненным фоном
    """
    # Создаем модель фона с помощью гауссова фильтра
    background = gaussian_filter(image_gray.astype(np.float32), sigma=sigma)
    
    # Вычитаем фон и нормализуем
    corrected = image_gray.astype(np.float32) - background + 128
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return corrected


def bilateral_denoising(image_gray: np.ndarray, d: int = 9, sigma_color: float = 75.0, 
                       sigma_space: float = 75.0) -> np.ndarray:
    """
    Удаление шума с сохранением границ с помощью билатеральной фильтрации.
    
    Args:
        image_gray: Входное изображение в градациях серого
        d: Диаметр окрестности для фильтрации
        sigma_color: Фильтр по цвету (яркости)
        sigma_space: Фильтр по пространству
    
    Returns:
        Отфильтрованное изображение
    """
    filtered = cv2.bilateralFilter(image_gray, d, sigma_color, sigma_space)
    return filtered


def gamma_correction(image_gray: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Гамма-коррекция для улучшения контраста.
    
    Args:
        image_gray: Входное изображение в градациях серого
        gamma: Значение гаммы (< 1 для осветления, > 1 для затемнения)
    
    Returns:
        Скорректированное изображение
    """
    # Создаем lookup table для гамма-коррекции
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # Применяем гамма-коррекцию
    corrected = cv2.LUT(image_gray, lookup_table)
    return corrected


def comprehensive_lighting_normalization(image_gray: np.ndarray, 
                                       clahe_params: dict = None,
                                       background_sigma: float = 50.0,
                                       bilateral_params: dict = None,
                                       gamma: float = 1.0) -> np.ndarray:
    """
    Комплексная нормализация освещения с применением нескольких методов.
    
    Args:
        image_gray: Входное изображение в градациях серого
        clahe_params: Параметры для CLAHE
        background_sigma: Сигма для коррекции фона
        bilateral_params: Параметры для билатеральной фильтрации
        gamma: Значение гаммы для коррекции
    
    Returns:
        Нормализованное изображение
    """
    # Параметры по умолчанию
    if clahe_params is None:
        clahe_params = {'clip_limit': 2.0, 'tile_grid_size': (8, 8)}
    if bilateral_params is None:
        bilateral_params = {'d': 9, 'sigma_color': 75.0, 'sigma_space': 75.0}
    
    # Применяем последовательность нормализации
    normalized = image_gray.copy()
    
    # 1. Коррекция фона
    normalized = background_correction(normalized, background_sigma)
    
    # 2. Билатеральная фильтрация для удаления шума
    normalized = bilateral_denoising(normalized, **bilateral_params)
    
    # 3. CLAHE для улучшения локального контраста
    normalized = normalize_lighting_clahe(normalized, **clahe_params)
    
    # 4. Гамма-коррекция (если gamma != 1.0)
    if gamma != 1.0:
        normalized = gamma_correction(normalized, gamma)
    
    return normalized


def analyze_image_brightness_with_normalization(image_path: str, 
                                              normalize_lighting: bool = True,
                                              normalization_params: dict = None) -> dict:
    """
    Анализ яркости с опциональной нормализацией освещения.
    
    Args:
        image_path: Путь к изображению
        normalize_lighting: Применять ли нормализацию освещения
        normalization_params: Параметры нормализации
    
    Returns:
        Словарь с результатами анализа (включая информацию о нормализации)
    """
    gray = _read_image_as_gray(image_path)
    
    # Анализируем оригинальное изображение
    original_stats = calculate_brightness_stats(gray)
    original_peaks = analyze_histogram_peaks(gray)
    
    result = {
        **original_stats,
        'peak_count': original_peaks['peak_count'],
        'peak_positions': original_peaks['peak_positions'],
        'normalization_applied': False
    }
    
    # Применяем нормализацию, если требуется
    if normalize_lighting:
        if normalization_params is None:
            normalization_params = {}
        
        # Удаляем enable_normalization из параметров, так как эта функция его не принимает
        filtered_params = {k: v for k, v in normalization_params.items() if k != 'enable_normalization'}
        normalized = comprehensive_lighting_normalization(gray, **filtered_params)
        
        # Анализируем нормализованное изображение
        normalized_stats = calculate_brightness_stats(normalized)
        normalized_peaks = analyze_histogram_peaks(normalized)
        
        # Добавляем результаты нормализации
        result.update({
            'normalized_mean_brightness': normalized_stats['mean_brightness'],
            'normalized_std_brightness': normalized_stats['std_brightness'],
            'normalized_peak_count': normalized_peaks['peak_count'],
            'normalized_peak_positions': normalized_peaks['peak_positions'],
            'normalization_applied': True,
            'brightness_improvement': abs(normalized_stats['std_brightness'] - original_stats['std_brightness'])
        })
    
    return result

