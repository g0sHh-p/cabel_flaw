import os
import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional


def analyze_contour_characteristics(image_path: str) -> Dict[str, float]:
    """
    Анализирует характеристики контуров на изображении
    
    Args:
        image_path: Путь к изображению
    
    Returns:
        Словарь с характеристиками контуров
    """
    try:
        # Загружаем изображение и преобразуем в оттенки серого
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Находим контуры на изображении
        contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return {}
        
        # Находим самый большой контур
        max_contour = max(contours, key=cv2.contourArea)
        
        # Вычисление площади контура
        area = cv2.contourArea(max_contour)
        
        # Вычисление периметра контура
        perimeter = cv2.arcLength(max_contour, closed=True)
        
        # Вычисление диаметра (эквивалентный диаметр круга)
        diameter = 2 * math.sqrt(area / math.pi) if area > 0 else 0
        
        # Ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Анализ формы через моменты
        moments = cv2.moments(max_contour)
        
        # Центральные моменты
        mu20 = moments['mu20']
        mu02 = moments['mu02']
        mu11 = moments['mu11']
        
        # Вычисление ориентации (угла наклона)
        if (mu20 - mu02) != 0:
            theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        else:
            theta = 0
        orientation = math.degrees(theta)
        
        # Вычисление эксцентриситета
        if mu20 > 0 and mu02 > 0:
            # Ковариационная матрица
            cov_matrix = np.array([[mu20, mu11], 
                                  [mu11, mu02]])
            
            # Собственные значения
            eigenvalues = np.linalg.eigvals(cov_matrix)
            lambda_max = max(eigenvalues)
            lambda_min = min(eigenvalues)
            
            # Эксцентриситет (мера "вытянутости")
            if lambda_max > 0:
                eccentricity = math.sqrt(1 - (lambda_min / lambda_max))
            else:
                eccentricity = 0
        else:
            eccentricity = 0
        
        # Компактность (отношение площади к периметру)
        compactness = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'diameter': float(diameter),
            'aspect_ratio': aspect_ratio,
            'orientation': float(orientation),
            'eccentricity': float(eccentricity),
            'compactness': float(compactness),
            'bounding_box_width': float(w),
            'bounding_box_height': float(h)
        }
        
    except Exception as e:
        print(f"Ошибка при анализе {image_path}: {e}")
        return {}


def analyze_batch_contours(folder_path: str) -> Dict[str, float]:
    """
    Анализирует характеристики контуров для всех изображений в папке
    
    Args:
        folder_path: Путь к папке с изображениями
    
    Returns:
        Словарь со средними характеристиками
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Папка {folder_path} не существует")
    
    all_characteristics = []
    
    for img_name in os.listdir(folder_path):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder_path, img_name)
            characteristics = analyze_contour_characteristics(img_path)
            if characteristics:
                all_characteristics.append(characteristics)
    
    if not all_characteristics:
        return {}
    
    # Вычисляем средние значения
    avg_characteristics = {}
    for key in all_characteristics[0].keys():
        values = [char[key] for char in all_characteristics]
        avg_characteristics[f'avg_{key}'] = float(np.mean(values))
        avg_characteristics[f'std_{key}'] = float(np.std(values))
        avg_characteristics[f'min_{key}'] = float(np.min(values))
        avg_characteristics[f'max_{key}'] = float(np.max(values))
    
    avg_characteristics['total_images'] = len(all_characteristics)
    
    return avg_characteristics


def compare_contour_characteristics(image1_path: str, image2_path: str) -> Dict[str, float]:
    """
    Сравнивает характеристики контуров двух изображений
    
    Args:
        image1_path: Путь к первому изображению
        image2_path: Путь ко второму изображению
    
    Returns:
        Словарь с различиями характеристик
    """
    char1 = analyze_contour_characteristics(image1_path)
    char2 = analyze_contour_characteristics(image2_path)
    
    if not char1 or not char2:
        return {}
    
    differences = {}
    for key in char1.keys():
        if key in char2:
            differences[f'diff_{key}'] = abs(char1[key] - char2[key])
            differences[f'rel_diff_{key}'] = abs(char1[key] - char2[key]) / max(char1[key], char2[key]) if max(char1[key], char2[key]) > 0 else 0
    
    return differences


# Пример использования
if __name__ == "__main__":
    # Пример анализа папки
    folder_path = "data/etln_proc"
    if os.path.exists(folder_path):
        avg_stats = analyze_batch_contours(folder_path)
        print("Средние характеристики контуров:")
        for key, value in avg_stats.items():
            if key.startswith('avg_'):
                clean_key = key.replace('avg_', '')
                print(f"  {clean_key}: {value:.3f}")
    
    # Пример анализа одного изображения
    sample_path = "data/etln_proc/processed_6376_bmp.rf.4a04a4a5e89439f07d04dcb0f2c5859c.jpg"
    if os.path.exists(sample_path):
        characteristics = analyze_contour_characteristics(sample_path)
        print("\nХарактеристики контуров для одного изображения:")
        for key, value in characteristics.items():
            print(f"  {key}: {value:.3f}")