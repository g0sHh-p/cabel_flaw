"""
Модуль core - основные функции для анализа кабелей и канатов

Этот модуль содержит основные функции для:
- Предварительной обработки изображений (preproc.py)
- Анализа яркости изображений (Brigh_analys.py)
- Анализа характеристик контуров (char_comp.py)
- Анализа текстуры методом Харалика (texture_analys.py)
"""

from .preproc import preprocess_images
from .Brigh_analys import (
    calculate_brightness_stats,
    calculate_batch_brightness_stats,
    calculate_skewness,
    calculate_kurtosis,
    analyze_histogram_peaks,
    analyze_image_brightness
)
from .char_comp import (
    analyze_contour_characteristics,
    analyze_batch_contours,
    compare_contour_characteristics
)
from .texture_analys import (
    extract_haralick_features,
    extract_cable_specific_features,
    analyze_cable_texture,
    compare_cable_textures
)
from .reference_model import (
    ReferenceModel,
    create_reference_model,
    load_reference_model
)
from .defect_detection_model import (
    DefectDetectionModel,
    create_defect_detection_model,
    load_defect_detection_model
)

__all__ = [
    # Предварительная обработка
    'preprocess_images',
    
    # Анализ яркости
    'calculate_brightness_stats',
    'calculate_batch_brightness_stats',
    'calculate_skewness',
    'calculate_kurtosis',
    'analyze_histogram_peaks',
    'analyze_image_brightness',
    
    # Анализ контуров
    'analyze_contour_characteristics',
    'analyze_batch_contours',
    'compare_contour_characteristics',
    
    # Анализ текстуры
    'extract_haralick_features',
    'extract_cable_specific_features',
    'analyze_cable_texture',
    'compare_cable_textures',
    
    # Модель эталона
    'ReferenceModel',
    'create_reference_model',
    'load_reference_model',
    
    # Модель детекции дефектов
    'DefectDetectionModel',
    'create_defect_detection_model',
    'load_defect_detection_model'
]

__version__ = "1.0.0"