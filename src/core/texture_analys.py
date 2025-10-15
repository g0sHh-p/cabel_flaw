import cv2
import numpy as np
from scipy import ndimage


def extract_haralick_features(image, distance: int = 1, angles_deg = [0, 45, 90, 135]):
    """
    Извлекает базовые Haralick-признаки (ASM, Contrast, Correlation, Entropy)
    через собственную реализацию GLCM. Без зависимостей от mahotas/skimage.
    Возвращает (dict, np.ndarray) по ключам, используемым моделью.
    """
    # Конвертируем в grayscale, квантуем уровни для компактной GLCM
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.max() <= 1:
        gray = (gray * 255).astype(np.uint8)

    # Квантуем до N уровней для устойчивости и скорости
    levels = 32
    quant = (gray.astype(np.uint16) * (levels - 1) // 255).astype(np.uint8)

    # Вектор направлений
    angles_rad = [a * np.pi / 180.0 for a in angles_deg]
    offsets = []
    for a in angles_rad:
        dx = int(round(np.cos(a))) * distance
        dy = int(round(-np.sin(a))) * distance
        offsets.append((dy, dx))  # (row offset, col offset)

    def glcm_for_offset(imgq: np.ndarray, dy: int, dx: int, L: int) -> np.ndarray:
        h, w = imgq.shape
        glcm = np.zeros((L, L), dtype=np.float64)
        if dy >= 0:
            r1 = slice(0, h - dy)
            r2 = slice(dy, h)
        else:
            r1 = slice(-dy, h)
            r2 = slice(0, h + dy)
        if dx >= 0:
            c1 = slice(0, w - dx)
            c2 = slice(dx, w)
        else:
            c1 = slice(-dx, w)
            c2 = slice(0, w + dx)
        a = imgq[r1, c1].reshape(-1)
        b = imgq[r2, c2].reshape(-1)
        idx = a * L + b
        hist = np.bincount(idx, minlength=L*L).astype(np.float64)
        glcm = hist.reshape(L, L)
        s = glcm.sum()
        if s > 0:
            glcm /= s
        return glcm

    glcms = [glcm_for_offset(quant, dy, dx, levels) for (dy, dx) in offsets]

    def haralick_from_glcm(P: np.ndarray):
        i = np.arange(P.shape[0])
        j = np.arange(P.shape[1])
        I, J = np.meshgrid(i, j, indexing='ij')
        # Angular Second Moment (Energy)
        asm = np.sum(P * P)
        # Contrast
        contrast = np.sum(((I - J) ** 2) * P)
        # Means and stds
        mu_i = np.sum(I * P)
        mu_j = np.sum(J * P)
        sigma_i = np.sqrt(np.sum(((I - mu_i) ** 2) * P))
        sigma_j = np.sqrt(np.sum(((J - mu_j) ** 2) * P))
        # Correlation (защита от деления на 0)
        if sigma_i > 0 and sigma_j > 0:
            correlation = np.sum(((I - mu_i) * (J - mu_j) * P)) / (sigma_i * sigma_j)
        else:
            correlation = 0.0
        # Entropy (log2, избегаем log(0))
        with np.errstate(divide='ignore', invalid='ignore'):
            logp = np.where(P > 0, np.log2(P), 0.0)
        entropy = -np.sum(P * logp)
        return asm, contrast, correlation, entropy

    feats = np.array([haralick_from_glcm(G) for G in glcms], dtype=float)
    mean_feats = feats.mean(axis=0)

    names = [
        'Angular Second Moment',
        'Contrast',
        'Correlation',
        'Entropy'
    ]
    feats_dict = dict(zip(names, mean_feats))
    return feats_dict, mean_feats


def extract_cable_specific_features(image):
    """
    Извлечение специфичных для канатов текстурных признаков
    
    Args:
        image: Бинаризованное изображение каната
    
    Returns:
        dict: Словарь с дополнительными признаками для анализа каната
    """
    # Конвертируем в grayscale если нужно
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Бинаризация если изображение не бинаризовано
    if gray.max() > 1:
        _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = gray
    
    # Находим контуры
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    features = {}
    
    if contours:
        # Основной контур (самый большой)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Площадь каната
        features['cable_area'] = cv2.contourArea(main_contour)
        
        # Периметр каната
        features['cable_perimeter'] = cv2.arcLength(main_contour, True)
        
        # Отношение площади к периметру (компактность)
        if features['cable_perimeter'] > 0:
            features['compactness'] = (4 * np.pi * features['cable_area']) / (features['cable_perimeter'] ** 2)
        else:
            features['compactness'] = 0
        
        # Ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(main_contour)
        features['aspect_ratio'] = w / h if h > 0 else 0
        features['bounding_box_area'] = w * h
        
        # Отношение площади каната к площади ограничивающего прямоугольника
        if features['bounding_box_area'] > 0:
            features['extent'] = features['cable_area'] / features['bounding_box_area']
        else:
            features['extent'] = 0
        
        # Эксцентриситет (форма эллипса)
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
            if a > 0 and b > 0:
                ratio = b**2 / a**2
                if ratio < 1:
                    features['eccentricity'] = np.sqrt(1 - ratio)
                else:
                    features['eccentricity'] = 0
            else:
                features['eccentricity'] = 0
        else:
            features['eccentricity'] = 0
    
    # Анализ текстуры проволок
    # Морфологические операции для выделения отдельных проволок
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx((binary * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    
    # Подсчет количества проволок (связных компонент)
    num_labels, labels = cv2.connectedComponents(opened)
    features['wire_count'] = num_labels - 1  # Исключаем фон
    
    # Анализ ориентации проволок
    if num_labels > 1:
        orientations = []
        for label in range(1, num_labels):
            mask = (labels == label).astype(np.uint8)
            if np.sum(mask) > 10:  # Игнорируем очень маленькие компоненты
                # Находим контур компонента
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        orientations.append(ellipse[2])  # Угол поворота
        
        if orientations:
            features['mean_wire_orientation'] = np.mean(orientations)
            features['orientation_std'] = np.std(orientations)
        else:
            features['mean_wire_orientation'] = 0
            features['orientation_std'] = 0
    
    return features


def analyze_cable_texture(image_path=None, image=None):
    """
    Полный анализ текстуры каната с использованием признаков Харалика и специфичных признаков
    
    Args:
        image_path: Путь к изображению (если не передан image)
        image: Изображение напрямую
    
    Returns:
        dict: Полный набор признаков для анализа каната
    """
    # Загружаем изображение
    if image is None and image_path is not None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    elif image is None:
        raise ValueError("Необходимо передать либо image_path, либо image")
    
    # Извлекаем признаки Харалика
    haralick_dict, haralick_features = extract_haralick_features(image)
    
    # Извлекаем специфичные для каната признаки
    cable_features = extract_cable_specific_features(image)
    
    # Объединяем все признаки
    all_features = {**haralick_dict, **cable_features}
    
    # Визуализация
    # Визуализация убрана для совместимости
    
    return all_features




def compare_cable_textures(image1_path, image2_path):
    """
    Сравнение текстур двух канатов
    
    Args:
        image1_path: Путь к первому изображению
        image2_path: Путь ко второму изображению
    
    Returns:
        dict: Результаты сравнения
    """
    # Анализируем оба изображения
    features1 = analyze_cable_texture(image_path=image1_path)
    features2 = analyze_cable_texture(image_path=image2_path)
    
    # Вычисляем различия
    differences = {}
    for key in features1.keys():
        if key in features2:
            differences[key] = abs(features1[key] - features2[key])
    
    # Сортируем по величине различий
    sorted_differences = sorted(differences.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'features1': features1,
        'features2': features2,
        'differences': dict(sorted_differences),
        'most_different_features': sorted_differences[:5]
    }


# Пример использования
if __name__ == "__main__":
    # Пример анализа одного изображения
    # image = cv2.imread('path/to/your/cable_image.jpg')
    # features = analyze_cable_texture(image=image)
    # print("Признаки текстуры каната:")
    # for name, value in features.items():
    #     print(f"{name}: {value:.4f}")
    
    # Пример сравнения двух изображений
    # comparison = compare_cable_textures('cable1.jpg', 'cable2.jpg')
    # print("\nНаиболее различающиеся признаки:")
    # for name, diff in comparison['most_different_features']:
    #     print(f"{name}: {diff:.4f}")
    
    print("Модуль анализа текстуры каната готов к использованию!")
    print("Используйте функцию analyze_cable_texture() для анализа изображений канатов.")

