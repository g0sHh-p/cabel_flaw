"""
Модель эталона для детекции дефектов канатов

Этот модуль создает эталонную модель на основе предобработанных изображений
и предоставляет функции для бинарной детекции дефектов канатов согласно ТЗ.

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

from .Brigh_analys import analyze_image_brightness, calculate_batch_brightness_stats
from .char_comp import analyze_contour_characteristics, analyze_batch_contours
from .texture_analys import analyze_cable_texture


class ReferenceModel:
    """
    Класс для создания и использования эталонной модели канатов
    для бинарной детекции дефектов согласно ТЗ
    """
    
    def __init__(self, model_name: str = "cable_defect_detection_model"):
        """
        Инициализация модели эталона
        
        Args:
            model_name: Название модели
        """
        self.model_name = model_name
        self.model_data = {}
        self.thresholds = {}
        self.statistics = {}
        self.created_at = None
        self.image_count = 0
        
        # Пороги для детекции дефектов (согласно ТЗ)
        self.defect_thresholds = {
            'diameter_deviation': 0.15,      # 15% отклонение диаметра
            'area_deviation': 0.20,          # 20% отклонение площади
            'wire_count_deviation': 0.30,    # 30% отклонение количества проволок
            'texture_anomaly': 0.25,         # 25% отклонение текстуры
            'brightness_anomaly': 0.20,      # 20% отклонение яркости
            'contour_irregularity': 0.30,    # 30% нерегулярность контура
            'overall_defect_threshold': 0.80  # 80% - порог для определения дефекта
        }
        
    def build_reference_model(self, processed_images_folder: str, 
                            original_images_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Строит эталонную модель на основе предобработанных изображений
        
        Args:
            processed_images_folder: Путь к папке с предобработанными изображениями
            original_images_folder: Путь к папке с оригинальными изображениями (опционально)
        
        Returns:
            Словарь с данными модели
        """
        print(f"Строим эталонную модель из папки: {processed_images_folder}")
        
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
                # Анализ яркости
                brightness_stats = analyze_image_brightness(img_path)
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
        
        # Вычисляем пороги для сравнения
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
        
        print(f"Модель эталона построена успешно!")
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
                if key != 'peak_positions':  # Пропускаем списки
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
        self.thresholds = self.model_data['thresholds']
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
            brightness_stats = analyze_image_brightness(image_path)
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
            brightness_stats = analyze_image_brightness(image_path)
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
    
    def _compare_brightness(self, test_stats: Dict, tolerance_factor: float) -> Dict[str, Any]:
        """
        Сравнивает статистики яркости с эталоном
        """
        if 'brightness' not in self.thresholds:
            return {'error': 'Нет данных о порогах яркости'}
        
        comparison = {}
        anomalies = []
        
        for key, test_value in test_stats.items():
            if key in self.thresholds['brightness'] and key != 'peak_positions':
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
            'thresholds_summary': {}
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
def create_reference_model(processed_images_folder: str, 
                         model_save_path: str = "reference_model.pkl") -> ReferenceModel:
    """
    Создает и сохраняет эталонную модель
    
    Args:
        processed_images_folder: Путь к папке с предобработанными изображениями
        model_save_path: Путь для сохранения модели
    
    Returns:
        Объект ReferenceModel
    """
    model = ReferenceModel()
    model.build_reference_model(processed_images_folder)
    model.save_model(model_save_path)
    return model


def load_reference_model(model_path: str) -> ReferenceModel:
    """
    Загружает эталонную модель
    
    Args:
        model_path: Путь к файлу модели
    
    Returns:
        Объект ReferenceModel
    """
    model = ReferenceModel()
    model.load_model(model_path)
    return model


# Пример использования
if __name__ == "__main__":
    # Создание модели эталона
    processed_folder = "data/etln_proc"
    model_path = "reference_model.pkl"
    
    if os.path.exists(processed_folder):
        print("Создаем эталонную модель...")
        model = create_reference_model(processed_folder, model_path)
        
        # Выводим сводку
        summary = model.get_model_summary()
        print("\nСводка модели:")
        print(f"Название: {summary['model_name']}")
        print(f"Создана: {summary['created_at']}")
        print(f"Изображений: {summary['image_count']}")
        print(f"Параметров яркости: {summary['statistics_summary']['brightness']['parameter_count']}")
        print(f"Параметров контуров: {summary['statistics_summary']['contour']['parameter_count']}")
        print(f"Параметров текстуры: {summary['statistics_summary']['texture']['parameter_count']}")
    else:
        print(f"Папка {processed_folder} не найдена")
