"""
Модель эталона для детекции дефектов канатов согласно ТЗ

Этот модуль создает эталонную модель на основе предобработанных изображений
и предоставляет функции для бинарной детекции дефектов канатов.

Поддерживаемые типы дефектов:
1. Отклонение по диаметру каната
2. Отсутствие проволок в пряди
3. Перекрещивание проволок
4. Перекрут пряди в канате (дефект «жучок»)
5. Неравномерный зазор между прядями
6. Выдавливание сердечника или проволоки
7. Отсутствие сердечника в канате
8. Дефект «бурунда»
9. Отсутствие пряди
10. Неправильная свивка
"""

import os
import cv2
import numpy as np
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

from .Brigh_analys import analyze_image_brightness, analyze_image_brightness_with_normalization, calculate_batch_brightness_stats
from .char_comp import analyze_contour_characteristics, analyze_batch_contours
from .texture_analys import analyze_cable_texture


class DefectDetectionModel:
    """
    Класс для создания и использования модели детекции дефектов канатов
    согласно техническому заданию
    """
    
    def __init__(self, model_name: str = "cable_defect_detection_model"):
        """
        Инициализация модели детекции дефектов
        
        Args:
            model_name: Название модели
        """
        self.model_name = model_name
        self.model_data = {}
        self.thresholds = {}  # Добавляем пороги для детального сравнения
        self.statistics = {}
        self.created_at = None
        self.image_count = 0
        
        # Пороги для детекции дефектов (согласно ТЗ)
        self.defect_thresholds = {
            'diameter_deviation': 0.30,      # 15% отклонение диаметра
            'area_deviation': 0.40,          # 20% отклонение площади
            'wire_count_deviation': 0.20,    # 30% отклонение количества проволок
            'texture_anomaly': 0.20,         # 25% отклонение текстуры
            'brightness_anomaly': 0.60,      # 20% отклонение яркости
            'contour_irregularity': 0.40,    # 30% нерегулярность контура
            'overall_defect_threshold': 0.90  # 80% - порог для определения дефекта
        }
        
        # Параметры нормализации освещения
        self.normalization_params = {
            'clahe_params': {'clip_limit': 2.0, 'tile_grid_size': (8, 8)},
            'background_sigma': 50.0,
            'bilateral_params': {'d': 9, 'sigma_color': 75.0, 'sigma_space': 75.0},
            'gamma': 1.0,
            'enable_normalization': True
        }
        
    def build_reference_model(self, processed_images_folder: str) -> Dict[str, Any]:
        """
        Строит эталонную модель на основе предобработанных изображений
        
        Args:
            processed_images_folder: Путь к папке с предобработанными изображениями
        
        Returns:
            Словарь с данными модели
        """
        print(f"Строим модель детекции дефектов из папки: {processed_images_folder}")
        
        if not os.path.exists(processed_images_folder):
            raise ValueError(f"Папка {processed_images_folder} не существует")
        
        # Собираем все изображения
        image_files = [f for f in os.listdir(processed_images_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            raise ValueError(f"В папке {processed_images_folder} нет изображений")
        
        print(f"Найдено {len(image_files)} изображений для анализа")
        
        # Инициализируем списки для сбора данных
        brightness_data = []
        contour_data = []
        texture_data = []
        
        # Анализируем каждое изображение
        for i, img_file in enumerate(image_files):
            print(f"Анализируем {i+1}/{len(image_files)}: {img_file}")
            
            img_path = os.path.join(processed_images_folder, img_file)
            
            try:
                # Анализ яркости с нормализацией освещения
                # Создаем копию параметров без enable_normalization
                norm_params = {k: v for k, v in self.normalization_params.items() if k != 'enable_normalization'}
                brightness_stats = analyze_image_brightness_with_normalization(
                    img_path, 
                    normalize_lighting=self.normalization_params['enable_normalization'],
                    normalization_params=norm_params
                )
                if brightness_stats:
                    brightness_data.append(brightness_stats)
                
                # Анализ контуров
                contour_stats = analyze_contour_characteristics(img_path)
                if contour_stats:
                    contour_data.append(contour_stats)
                
                # Анализ текстуры
                texture_stats = analyze_cable_texture(image_path=img_path)
                if texture_stats:
                    texture_data.append(texture_stats)
                    
            except Exception as e:
                print(f"Ошибка при анализе {img_file}: {e}")
                continue
        
        # Вычисляем статистики эталона
        self._calculate_reference_statistics(brightness_data, contour_data, texture_data)
        
        # Вычисляем пороги для сравнения (3 сигмы от среднего)
        self._calculate_thresholds(brightness_data, contour_data, texture_data)
        
        # Сохраняем данные модели
        self.model_data = {
            'brightness_data': brightness_data,
            'contour_data': contour_data,
            'texture_data': texture_data,
            'statistics': self.statistics,
            'thresholds': self.thresholds,
            'image_count': len(image_files),
            'created_at': datetime.now().isoformat(),
            'model_name': self.model_name
        }
        
        self.image_count = len(image_files)
        self.created_at = datetime.now()
        
        print(f"Модель детекции дефектов построена успешно!")
        print(f"Проанализировано изображений: {self.image_count}")
        
        return self.model_data
    
    def _calculate_reference_statistics(self, brightness_data: List[Dict], 
                                      contour_data: List[Dict], 
                                      texture_data: List[Dict]) -> None:
        """
        Вычисляет статистики эталона
        """
        print("Вычисляем статистики эталона...")
        
        # Статистики яркости
        if brightness_data:
            brightness_stats = {}
            for key in brightness_data[0].keys():
                if key != 'peak_positions' and not key.startswith('normalized_'):  # Пропускаем списки и нормализованные значения
                    values = [item[key] for item in brightness_data if key in item]
                    if values:
                        brightness_stats[key] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'median': float(np.median(values))
                        }
            self.statistics['brightness'] = brightness_stats
        
        # Статистики контуров
        if contour_data:
            contour_stats = {}
            for key in contour_data[0].keys():
                values = [item[key] for item in contour_data if key in item]
                if values:
                    contour_stats[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values))
                    }
            self.statistics['contour'] = contour_stats
        
        # Статистики текстуры
        if texture_data:
            texture_stats = {}
            for key in texture_data[0].keys():
                if key != 'peak_positions':  # Пропускаем списки
                    values = [item[key] for item in texture_data if key in item]
                    if values:
                        texture_stats[key] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'median': float(np.median(values))
                        }
            self.statistics['texture'] = texture_stats
    
    def _calculate_thresholds(self, brightness_data: List[Dict], 
                            contour_data: List[Dict], 
                            texture_data: List[Dict]) -> None:
        """
        Вычисляет пороги для сравнения (3 сигмы от среднего)
        """
        print("Вычисляем пороги для сравнения...")
        
        # Пороги яркости
        if brightness_data and 'brightness' in self.statistics:
            brightness_thresholds = {}
            for key, stats in self.statistics['brightness'].items():
                # Используем 3 сигмы для определения допустимых отклонений
                brightness_thresholds[key] = {
                    'lower_bound': stats['mean'] - 3 * stats['std'],
                    'upper_bound': stats['mean'] + 3 * stats['std'],
                    'tolerance': 3 * stats['std']  # Допустимое отклонение
                }
            self.thresholds['brightness'] = brightness_thresholds
        
        # Пороги контуров
        if contour_data and 'contour' in self.statistics:
            contour_thresholds = {}
            for key, stats in self.statistics['contour'].items():
                contour_thresholds[key] = {
                    'lower_bound': stats['mean'] - 3 * stats['std'],
                    'upper_bound': stats['mean'] + 3 * stats['std'],
                    'tolerance': 3 * stats['std']
                }
            self.thresholds['contour'] = contour_thresholds
        
        # Пороги текстуры
        if texture_data and 'texture' in self.statistics:
            texture_thresholds = {}
            for key, stats in self.statistics['texture'].items():
                texture_thresholds[key] = {
                    'lower_bound': stats['mean'] - 3 * stats['std'],
                    'upper_bound': stats['mean'] + 3 * stats['std'],
                    'tolerance': 3 * stats['std']
                }
            self.thresholds['texture'] = texture_thresholds
    
    def save_model(self, filepath: str) -> None:
        """
        Сохраняет модель в файл
        
        Args:
            filepath: Путь для сохранения модели
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model_data, f)
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Загружает модель из файла
        
        Args:
            filepath: Путь к файлу модели
        """
        with open(filepath, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.statistics = self.model_data['statistics']
        self.thresholds = self.model_data.get('thresholds', {})  # Безопасная загрузка порогов
        self.image_count = self.model_data['image_count']
        self.model_name = self.model_data['model_name']
        self.created_at = datetime.fromisoformat(self.model_data['created_at'])
        
        print(f"Модель загружена из {filepath}")
        print(f"Модель создана: {self.created_at}")
        print(f"Количество эталонных изображений: {self.image_count}")
    
    def detect_defects(self, image_path: str) -> Dict[str, Any]:
        """
        Бинарная детекция дефектов каната согласно ТЗ
        
        Args:
            image_path: Путь к изображению для анализа
        
        Returns:
            Словарь с результатами детекции дефектов
        """
        if not self.model_data:
            raise ValueError("Модель не загружена. Сначала постройте или загрузите модель.")
        
        print(f"Анализируем изображение {image_path} на наличие дефектов...")
        
        # Анализируем тестовое изображение
        try:
            # Создаем копию параметров без enable_normalization
            norm_params = {k: v for k, v in self.normalization_params.items() if k != 'enable_normalization'}
            brightness_stats = analyze_image_brightness_with_normalization(
                image_path,
                normalize_lighting=self.normalization_params['enable_normalization'],
                normalization_params=norm_params
            )
            contour_stats = analyze_contour_characteristics(image_path)
            texture_stats = analyze_cable_texture(image_path=image_path)
        except Exception as e:
            raise ValueError(f"Ошибка при анализе изображения: {e}")
        
        # Выполняем детекцию дефектов
        defect_analysis = {
            'image_path': image_path,
            'analysis_time': datetime.now().isoformat(),
            'has_defect': False,
            'defect_confidence': 0.0,
            'defect_indicators': {},
            'processing_time': 0.0
        }
        
        start_time = datetime.now()
        
        # 1. Анализ отклонения диаметра (дефект #1)
        diameter_defect = self._check_diameter_deviation(contour_stats)
        defect_analysis['defect_indicators']['diameter_deviation'] = diameter_defect
        
        # 2. Анализ количества проволок (дефекты #2, #9)
        wire_defects = self._check_wire_defects(texture_stats, contour_stats)
        defect_analysis['defect_indicators']['wire_defects'] = wire_defects
        
        # 3. Анализ текстуры (дефекты #3, #4, #8, #10)
        texture_defects = self._check_texture_defects(texture_stats)
        defect_analysis['defect_indicators']['texture_defects'] = texture_defects
        
        # 4. Анализ контуров (дефекты #5, #6, #7)
        contour_defects = self._check_contour_defects(contour_stats)
        defect_analysis['defect_indicators']['contour_defects'] = contour_defects
        
        # 5. Анализ яркости (общие аномалии)
        brightness_defects = self._check_brightness_defects(brightness_stats)
        defect_analysis['defect_indicators']['brightness_defects'] = brightness_defects
        
        # Вычисляем общую уверенность в наличии дефекта
        defect_confidence = self._calculate_defect_confidence(defect_analysis['defect_indicators'])
        defect_analysis['defect_confidence'] = defect_confidence
        defect_analysis['has_defect'] = defect_confidence >= self.defect_thresholds['overall_defect_threshold']
        
        # Время обработки
        processing_time = (datetime.now() - start_time).total_seconds()
        defect_analysis['processing_time'] = processing_time
        
        return defect_analysis
    
    def compare_with_reference(self, image_path: str, 
                             tolerance_factor: float = 1.0) -> Dict[str, Any]:
        """
        Сравнивает изображение с эталоном
        
        Args:
            image_path: Путь к изображению для сравнения
            tolerance_factor: Коэффициент для корректировки порогов (1.0 = стандартные пороги)
        
        Returns:
            Словарь с результатами сравнения
        """
        if not self.model_data:
            raise ValueError("Модель не загружена. Сначала постройте или загрузите модель.")
        
        print(f"Сравниваем изображение {image_path} с эталоном...")
        
        # Анализируем тестовое изображение
        try:
            # Создаем копию параметров без enable_normalization
            norm_params = {k: v for k, v in self.normalization_params.items() if k != 'enable_normalization'}
            brightness_stats = analyze_image_brightness_with_normalization(
                image_path,
                normalize_lighting=self.normalization_params['enable_normalization'],
                normalization_params=norm_params
            )
            contour_stats = analyze_contour_characteristics(image_path)
            texture_stats = analyze_cable_texture(image_path=image_path)
        except Exception as e:
            raise ValueError(f"Ошибка при анализе изображения: {e}")
        
        # Выполняем сравнение
        comparison_results = {
            'image_path': image_path,
            'comparison_time': datetime.now().isoformat(),
            'tolerance_factor': tolerance_factor,
            'brightness_comparison': self._compare_brightness(brightness_stats, tolerance_factor),
            'contour_comparison': self._compare_contour(contour_stats, tolerance_factor),
            'texture_comparison': self._compare_texture(texture_stats, tolerance_factor),
            'overall_score': 0.0,
            'is_within_tolerance': True,
            'anomalies': []
        }
        
        # Вычисляем общий балл и определяем аномалии
        self._calculate_overall_score(comparison_results)
        
        return comparison_results
    
    def _check_diameter_deviation(self, contour_stats: Dict) -> Dict[str, Any]:
        """
        Проверка отклонения диаметра каната (дефект #1)
        """
        if 'contour' not in self.statistics or 'diameter' not in self.statistics['contour']:
            return {'detected': False, 'confidence': 0.0, 'reason': 'Нет данных о диаметре'}
        
        reference_diameter = self.statistics['contour']['diameter']['mean']
        test_diameter = contour_stats.get('diameter', 0)
        
        if reference_diameter == 0:
            return {'detected': False, 'confidence': 0.0, 'reason': 'Нулевой эталонный диаметр'}
        
        deviation = abs(test_diameter - reference_diameter) / reference_diameter
        threshold = self.defect_thresholds['diameter_deviation']
        
        return {
            'detected': deviation > threshold,
            'confidence': min(deviation / threshold, 1.0),
            'deviation_percent': deviation * 100,
            'reference_diameter': reference_diameter,
            'test_diameter': test_diameter,
            'threshold': threshold * 100
        }
    
    def _check_wire_defects(self, texture_stats: Dict, contour_stats: Dict) -> Dict[str, Any]:
        """
        Проверка дефектов проволок (дефекты #2, #9 - отсутствие проволок/прядей)
        """
        if 'texture' not in self.statistics or 'wire_count' not in self.statistics['texture']:
            return {'detected': False, 'confidence': 0.0, 'reason': 'Нет данных о проволоках'}
        
        reference_wire_count = self.statistics['texture']['wire_count']['mean']
        test_wire_count = texture_stats.get('wire_count', 0)
        
        if reference_wire_count == 0:
            return {'detected': False, 'confidence': 0.0, 'reason': 'Нулевое эталонное количество проволок'}
        
        # Проверяем значительное отклонение количества проволок
        deviation = abs(test_wire_count - reference_wire_count) / reference_wire_count
        threshold = self.defect_thresholds['wire_count_deviation']
        
        # Дополнительная проверка компактности (может указывать на отсутствие проволок)
        compactness_defect = False
        if 'contour' in self.statistics and 'compactness' in self.statistics['contour']:
            ref_compactness = self.statistics['contour']['compactness']['mean']
            test_compactness = contour_stats.get('compactness', 0)
            compactness_deviation = abs(test_compactness - ref_compactness) / ref_compactness
            compactness_defect = compactness_deviation > 0.25
        
        return {
            'detected': deviation > threshold or compactness_defect,
            'confidence': max(deviation / threshold, 0.3 if compactness_defect else 0.0),
            'wire_count_deviation': deviation * 100,
            'reference_wire_count': reference_wire_count,
            'test_wire_count': test_wire_count,
            'compactness_anomaly': compactness_defect
        }
    
    def _check_texture_defects(self, texture_stats: Dict) -> Dict[str, Any]:
        """
        Проверка дефектов текстуры (дефекты #3, #4, #8, #10 - перекрещивание, перекрут, бурунда, неправильная свивка)
        """
        if 'texture' not in self.statistics:
            return {'detected': False, 'confidence': 0.0, 'reason': 'Нет данных о текстуре'}
        
        texture_anomalies = []
        total_confidence = 0.0
        
        # Проверяем ключевые признаки Харалика
        key_features = ['Angular Second Moment', 'Contrast', 'Correlation', 'Entropy']
        
        for feature in key_features:
            if feature in self.statistics['texture'] and feature in texture_stats:
                ref_mean = self.statistics['texture'][feature]['mean']
                ref_std = self.statistics['texture'][feature]['std']
                test_value = texture_stats[feature]
                
                if ref_std > 0:
                    z_score = abs(test_value - ref_mean) / ref_std
                    if z_score > 2.0:  # Значительное отклонение
                        texture_anomalies.append(feature)
                        total_confidence += min(z_score / 3.0, 1.0)
        
        # Проверяем ориентацию проволок (может указывать на перекрут)
        orientation_anomaly = False
        if 'mean_wire_orientation' in texture_stats and 'mean_wire_orientation' in self.statistics['texture']:
            ref_orientation = self.statistics['texture']['mean_wire_orientation']['mean']
            test_orientation = texture_stats['mean_wire_orientation']
            orientation_diff = abs(test_orientation - ref_orientation)
            if orientation_diff > 30:  # Значительное отклонение ориентации
                orientation_anomaly = True
                total_confidence += 0.3
        
        return {
            'detected': len(texture_anomalies) > 0 or orientation_anomaly,
            'confidence': min(total_confidence / len(key_features), 1.0),
            'texture_anomalies': texture_anomalies,
            'orientation_anomaly': orientation_anomaly,
            'anomaly_count': len(texture_anomalies)
        }
    
    def _check_contour_defects(self, contour_stats: Dict) -> Dict[str, Any]:
        """
        Проверка дефектов контуров (дефекты #5, #6, #7 - неравномерные зазоры, выдавливание, отсутствие сердечника)
        """
        if 'contour' not in self.statistics:
            return {'detected': False, 'confidence': 0.0, 'reason': 'Нет данных о контурах'}
        
        contour_anomalies = []
        total_confidence = 0.0
        
        # Проверяем эксцентриситет (может указывать на выдавливание)
        if 'eccentricity' in self.statistics['contour'] and 'eccentricity' in contour_stats:
            ref_eccentricity = self.statistics['contour']['eccentricity']['mean']
            test_eccentricity = contour_stats['eccentricity']
            eccentricity_diff = abs(test_eccentricity - ref_eccentricity)
            if eccentricity_diff > 0.1:  # Значительное отклонение формы
                contour_anomalies.append('eccentricity')
                total_confidence += min(eccentricity_diff * 2, 1.0)
        
        # Проверяем соотношение сторон (может указывать на деформацию)
        if 'aspect_ratio' in self.statistics['contour'] and 'aspect_ratio' in contour_stats:
            ref_aspect = self.statistics['contour']['aspect_ratio']['mean']
            test_aspect = contour_stats['aspect_ratio']
            if ref_aspect > 0:
                aspect_deviation = abs(test_aspect - ref_aspect) / ref_aspect
                if aspect_deviation > 0.2:  # 20% отклонение
                    contour_anomalies.append('aspect_ratio')
                    total_confidence += min(aspect_deviation, 1.0)
        
        # Проверяем компактность (может указывать на отсутствие сердечника)
        if 'compactness' in self.statistics['contour'] and 'compactness' in contour_stats:
            ref_compactness = self.statistics['contour']['compactness']['mean']
            test_compactness = contour_stats['compactness']
            compactness_diff = abs(test_compactness - ref_compactness)
            if compactness_diff > 0.15:  # Значительное отклонение компактности
                contour_anomalies.append('compactness')
                total_confidence += min(compactness_diff * 3, 1.0)
        
        return {
            'detected': len(contour_anomalies) > 0,
            'confidence': min(total_confidence / 3, 1.0),
            'contour_anomalies': contour_anomalies,
            'anomaly_count': len(contour_anomalies)
        }
    
    def _check_brightness_defects(self, brightness_stats: Dict) -> Dict[str, Any]:
        """
        Проверка аномалий яркости (общие дефекты)
        """
        if 'brightness' not in self.statistics:
            return {'detected': False, 'confidence': 0.0, 'reason': 'Нет данных о яркости'}
        
        brightness_anomalies = []
        total_confidence = 0.0
        
        # Выбираем ключи для анализа (нормализованные, если доступны)
        mean_key = 'normalized_mean_brightness' if 'normalized_mean_brightness' in brightness_stats else 'mean_brightness'
        std_key = 'normalized_std_brightness' if 'normalized_std_brightness' in brightness_stats else 'std_brightness'
        
        # Проверяем среднюю яркость
        if 'mean_brightness' in self.statistics['brightness'] and mean_key in brightness_stats:
            ref_brightness = self.statistics['brightness']['mean_brightness']['mean']
            ref_std = self.statistics['brightness']['mean_brightness']['std']
            test_brightness = brightness_stats[mean_key]
            
            if ref_std > 0:
                z_score = abs(test_brightness - ref_brightness) / ref_std
                if z_score > 2.5:  # Значительное отклонение яркости
                    brightness_anomalies.append(mean_key)
                    total_confidence += min(z_score / 4.0, 1.0)
        
        # Проверяем стандартное отклонение яркости
        if 'std_brightness' in self.statistics['brightness'] and std_key in brightness_stats:
            ref_std_brightness = self.statistics['brightness']['std_brightness']['mean']
            test_std_brightness = brightness_stats[std_key]
            
            if ref_std_brightness > 0:
                std_deviation = abs(test_std_brightness - ref_std_brightness) / ref_std_brightness
                if std_deviation > 0.3:  # 30% отклонение стандартного отклонения
                    brightness_anomalies.append(std_key)
                    total_confidence += min(std_deviation, 1.0)
        
        return {
            'detected': len(brightness_anomalies) > 0,
            'confidence': min(total_confidence / 2, 1.0),
            'brightness_anomalies': brightness_anomalies,
            'anomaly_count': len(brightness_anomalies)
        }
    
    def _calculate_defect_confidence(self, defect_indicators: Dict) -> float:
        """
        Вычисляет общую уверенность в наличии дефекта
        """
        total_confidence = 0.0
        weight_sum = 0.0
        
        # Веса для разных типов дефектов
        # Максимальное внимание текстуре каната, дефектам свивки и проволок
        # Минимальное внимание отклонениям контура
        weights = {
            'diameter_deviation': 0.10,      # Низкий приоритет (диаметр менее критичен)
            'wire_defects': 0.25,            # Очень высокий приоритет (проволоки критичны)
            'texture_defects': 0.25,         # Максимальный приоритет (текстура и свивка критичны)
            'contour_defects': 0.10,         # Минимальный приоритет (контур менее важен)
            'brightness_defects': 0.05       # Низкий приоритет (яркость менее критична)
        }
        
        for defect_type, weight in weights.items():
            if defect_type in defect_indicators:
                confidence = defect_indicators[defect_type].get('confidence', 0.0)
                total_confidence += confidence * weight
                weight_sum += weight
        
        if weight_sum > 0:
            return total_confidence / weight_sum
        else:
            return 0.0
    
    def set_normalization_params(self, **params) -> None:
        """
        Настройка параметров нормализации освещения
        
        Args:
            **params: Параметры нормализации (clahe_params, background_sigma, 
                     bilateral_params, gamma, enable_normalization)
        """
        self.normalization_params.update(params)
        print(f"Параметры нормализации освещения обновлены: {params}")
    
    
    def _compare_brightness(self, test_stats: Dict, tolerance_factor: float) -> Dict[str, Any]:
        """
        Сравнивает статистики яркости с эталоном
        """
        if 'brightness' not in self.thresholds:
            return {'error': 'Нет данных о порогах яркости'}
        
        comparison = {}
        anomalies = []
        
        for key, test_value in test_stats.items():
            if key in self.thresholds['brightness'] and key != 'peak_positions' and not key.startswith('normalized_'):
                threshold = self.thresholds['brightness'][key]
                adjusted_tolerance = threshold['tolerance'] * tolerance_factor
                
                reference_mean = self.statistics['brightness'][key]['mean']
                lower_bound = reference_mean - adjusted_tolerance
                upper_bound = reference_mean + adjusted_tolerance
                
                is_within_bounds = lower_bound <= test_value <= upper_bound
                deviation = abs(test_value - reference_mean)
                relative_deviation = deviation / reference_mean if reference_mean != 0 else 0
                
                comparison[key] = {
                    'test_value': test_value,
                    'reference_mean': reference_mean,
                    'reference_std': self.statistics['brightness'][key]['std'],
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'deviation': deviation,
                    'relative_deviation': relative_deviation,
                    'is_within_bounds': is_within_bounds
                }
                
                if not is_within_bounds:
                    anomalies.append(f"Яркость {key}: {test_value:.3f} (эталон: {reference_mean:.3f})")
        
        return {
            'comparison': comparison,
            'anomalies': anomalies,
            'total_anomalies': len(anomalies)
        }
    
    def _compare_contour(self, test_stats: Dict, tolerance_factor: float) -> Dict[str, Any]:
        """
        Сравнивает статистики контуров с эталоном
        """
        if 'contour' not in self.thresholds:
            return {'error': 'Нет данных о порогах контуров'}
        
        comparison = {}
        anomalies = []
        
        for key, test_value in test_stats.items():
            if key in self.thresholds['contour']:
                threshold = self.thresholds['contour'][key]
                adjusted_tolerance = threshold['tolerance'] * tolerance_factor
                
                reference_mean = self.statistics['contour'][key]['mean']
                lower_bound = reference_mean - adjusted_tolerance
                upper_bound = reference_mean + adjusted_tolerance
                
                is_within_bounds = lower_bound <= test_value <= upper_bound
                deviation = abs(test_value - reference_mean)
                relative_deviation = deviation / reference_mean if reference_mean != 0 else 0
                
                comparison[key] = {
                    'test_value': test_value,
                    'reference_mean': reference_mean,
                    'reference_std': self.statistics['contour'][key]['std'],
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'deviation': deviation,
                    'relative_deviation': relative_deviation,
                    'is_within_bounds': is_within_bounds
                }
                
                if not is_within_bounds:
                    anomalies.append(f"Контур {key}: {test_value:.3f} (эталон: {reference_mean:.3f})")
        
        return {
            'comparison': comparison,
            'anomalies': anomalies,
            'total_anomalies': len(anomalies)
        }
    
    def _compare_texture(self, test_stats: Dict, tolerance_factor: float) -> Dict[str, Any]:
        """
        Сравнивает статистики текстуры с эталоном
        """
        if 'texture' not in self.thresholds:
            return {'error': 'Нет данных о порогах текстуры'}
        
        comparison = {}
        anomalies = []
        
        for key, test_value in test_stats.items():
            if key in self.thresholds['texture'] and key != 'peak_positions':
                threshold = self.thresholds['texture'][key]
                adjusted_tolerance = threshold['tolerance'] * tolerance_factor
                
                reference_mean = self.statistics['texture'][key]['mean']
                lower_bound = reference_mean - adjusted_tolerance
                upper_bound = reference_mean + adjusted_tolerance
                
                is_within_bounds = lower_bound <= test_value <= upper_bound
                deviation = abs(test_value - reference_mean)
                relative_deviation = deviation / reference_mean if reference_mean != 0 else 0
                
                comparison[key] = {
                    'test_value': test_value,
                    'reference_mean': reference_mean,
                    'reference_std': self.statistics['texture'][key]['std'],
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'deviation': deviation,
                    'relative_deviation': relative_deviation,
                    'is_within_bounds': is_within_bounds
                }
                
                if not is_within_bounds:
                    anomalies.append(f"Текстура {key}: {test_value:.3f} (эталон: {reference_mean:.3f})")
        
        return {
            'comparison': comparison,
            'anomalies': anomalies,
            'total_anomalies': len(anomalies)
        }
    
    def _calculate_overall_score(self, comparison_results: Dict) -> None:
        """
        Вычисляет общий балл сравнения
        """
        total_checks = 0
        passed_checks = 0
        all_anomalies = []
        
        # Подсчитываем проверки по яркости
        if 'brightness_comparison' in comparison_results and 'comparison' in comparison_results['brightness_comparison']:
            for key, result in comparison_results['brightness_comparison']['comparison'].items():
                total_checks += 1
                if result['is_within_bounds']:
                    passed_checks += 1
            all_anomalies.extend(comparison_results['brightness_comparison']['anomalies'])
        
        # Подсчитываем проверки по контурам
        if 'contour_comparison' in comparison_results and 'comparison' in comparison_results['contour_comparison']:
            for key, result in comparison_results['contour_comparison']['comparison'].items():
                total_checks += 1
                if result['is_within_bounds']:
                    passed_checks += 1
            all_anomalies.extend(comparison_results['contour_comparison']['anomalies'])
        
        # Подсчитываем проверки по текстуре
        if 'texture_comparison' in comparison_results and 'comparison' in comparison_results['texture_comparison']:
            for key, result in comparison_results['texture_comparison']['comparison'].items():
                total_checks += 1
                if result['is_within_bounds']:
                    passed_checks += 1
            all_anomalies.extend(comparison_results['texture_comparison']['anomalies'])
        
        # Вычисляем общий балл
        if total_checks > 0:
            overall_score = (passed_checks / total_checks) * 100
        else:
            overall_score = 0
        
        comparison_results['overall_score'] = overall_score
        comparison_results['passed_checks'] = passed_checks
        comparison_results['total_checks'] = total_checks
        comparison_results['is_within_tolerance'] = overall_score >= 80  # 80% проверок должны пройти
        comparison_results['anomalies'] = all_anomalies
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку по модели
        """
        if not self.model_data:
            return {'error': 'Модель не загружена'}
        
        summary = {
            'model_name': self.model_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'image_count': self.image_count,
            'statistics_summary': {},
            'thresholds_summary': {},
            'defect_thresholds': self.defect_thresholds,
            'normalization_params': self.normalization_params
        }
        
        # Сводка по статистикам
        for category, stats in self.statistics.items():
            summary['statistics_summary'][category] = {
                'parameter_count': len(stats),
                'parameters': list(stats.keys())
            }
        
        # Сводка по порогам
        for category, thresholds in self.thresholds.items():
            summary['thresholds_summary'][category] = {
                'parameter_count': len(thresholds),
                'parameters': list(thresholds.keys())
            }
        
        return summary


# Функции для удобного использования
def create_defect_detection_model(processed_images_folder: str, 
                                model_save_path: str = "defect_detection_model.pkl") -> DefectDetectionModel:
    """
    Создает и сохраняет модель детекции дефектов
    
    Args:
        processed_images_folder: Путь к папке с предобработанными изображениями
        model_save_path: Путь для сохранения модели
    
    Returns:
        Объект DefectDetectionModel
    """
    model = DefectDetectionModel()
    model.build_reference_model(processed_images_folder)
    model.save_model(model_save_path)
    return model


def load_defect_detection_model(model_path: str) -> DefectDetectionModel:
    """
    Загружает модель детекции дефектов
    
    Args:
        model_path: Путь к файлу модели
    
    Returns:
        Объект DefectDetectionModel
    """
    model = DefectDetectionModel()
    model.load_model(model_path)
    return model


# Функции для совместимости с reference_model.py
def create_reference_model(processed_images_folder: str, 
                         model_save_path: str = "reference_model.pkl") -> DefectDetectionModel:
    """
    Создает и сохраняет эталонную модель (совместимость с reference_model.py)
    
    Args:
        processed_images_folder: Путь к папке с предобработанными изображениями
        model_save_path: Путь для сохранения модели
    
    Returns:
        Объект DefectDetectionModel
    """
    model = DefectDetectionModel()
    model.build_reference_model(processed_images_folder)
    model.save_model(model_save_path)
    return model


def load_reference_model(model_path: str) -> DefectDetectionModel:
    """
    Загружает эталонную модель (совместимость с reference_model.py)
    
    Args:
        model_path: Путь к файлу модели
    
    Returns:
        Объект DefectDetectionModel
    """
    model = DefectDetectionModel()
    model.load_model(model_path)
    return model


# Пример использования
if __name__ == "__main__":
    # Создание модели детекции дефектов
    processed_folder = "data/etln_proc"
    model_path = "defect_detection_model.pkl"
    
    if os.path.exists(processed_folder):
        print("Создаем модель детекции дефектов...")
        model = create_defect_detection_model(processed_folder, model_path)
        
        # Выводим сводку
        summary = model.get_model_summary()
        print("\nСводка модели:")
        print(f"Название: {summary['model_name']}")
        print(f"Создана: {summary['created_at']}")
        print(f"Изображений: {summary['image_count']}")
        print(f"Параметров яркости: {summary['statistics_summary']['brightness']['parameter_count']}")
        print(f"Параметров контуров: {summary['statistics_summary']['contour']['parameter_count']}")
        print(f"Параметров текстуры: {summary['statistics_summary']['texture']['parameter_count']}")
        
        print("\nПороги детекции дефектов:")
        for key, value in summary['defect_thresholds'].items():
            print(f"  {key}: {value}")
    else:
        print(f"Папка {processed_folder} не найдена")
