import os
import json
import math
import random
from typing import List, Dict, Tuple

import numpy as np

from .Brigh_analys import analyze_image_brightness
from .char_comp import analyze_contour_characteristics
from .texture_analys import analyze_cable_texture


def _normalize_feature_vector(features: Dict[str, float], keys: List[str]) -> np.ndarray:
    """
    Формирует числовой вектор признаков по заданным ключам, заполняя отсутствующие нулями.
    """
    return np.array([float(features.get(k, 0.0)) for k in keys], dtype=float)


def _standardize_matrix(X: np.ndarray) -> np.ndarray:
    """
    Стандартизует матрицу признаков по столбцам (z-score), защищаясь от деления на 0.
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def _farthest_point_sampling(X: np.ndarray, k: int, seed: int = 42) -> List[int]:
    """
    Жадная выборка k объектов, максимально разноудалённых в пространстве признаков.
    Возвращает индексы выбранных элементов.
    """
    n = X.shape[0]
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, n))]
    min_dist = np.full(n, np.inf)
    while len(selected) < k:
        last = selected[-1]
        # Обновляем минимальные расстояния до множества выбранных
        dists = np.linalg.norm(X - X[last], axis=1)
        min_dist = np.minimum(min_dist, dists)
        # Выбираем точку с максимальным минимальным расстоянием
        candidate = int(np.argmax(min_dist))
        if candidate in selected:
            # В редком случае повторного выбора — сдвигаем случайно
            candidate = int(rng.integers(0, n))
        if candidate not in selected:
            selected.append(candidate)
        else:
            # Защита от зацикливания
            for i in range(n):
                if i not in selected:
                    selected.append(i)
                    break
    return selected[:k]


def build_feature_matrix(folder_path: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Строит матрицу признаков для всех изображений в папке.
    Возвращает: (список путей, матрица признаков, список имён признаков)
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Папка {folder_path} не существует")

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    image_paths = [os.path.join(folder_path, f) for f in image_files]

    # Сбор признаков
    brightness_keys = ['mean_brightness', 'std_brightness', 'peak_count']
    # Контуры
    contour_keys = [
        'area', 'perimeter', 'diameter', 'aspect_ratio',
        'orientation', 'eccentricity', 'compactness',
        'bounding_box_width', 'bounding_box_height'
    ]
    # Текстура (частичные ключи — отфильтруем по наличию)
    texture_candidate_keys = [
        'Angular Second Moment', 'Contrast', 'Correlation', 'Entropy',
        'Sum of Squares: Variance', 'Inverse Difference Moment', 'Sum Average',
        'Sum Variance', 'Sum Entropy', 'Difference Variance',
        'Difference Entropy', 'Information Measure 1', 'Information Measure 2',
        'cable_area', 'cable_perimeter', 'compactness', 'aspect_ratio',
        'bounding_box_area', 'extent', 'eccentricity', 'wire_count',
        'mean_wire_orientation', 'orientation_std'
    ]

    rows = []
    kept_paths = []
    for p in image_paths:
        try:
            b = analyze_image_brightness(p)
            c = analyze_contour_characteristics(p)
            t = analyze_cable_texture(image_path=p)
            # Финальный список текстурных ключей = только те, что есть хотя бы в одном примере
            rows.append((p, b, c, t))
            kept_paths.append(p)
        except Exception:
            continue

    if not rows:
        raise ValueError("Не удалось собрать признаки ни для одного изображения")

    # Соберём множество реально встреченных texture-ключей
    found_texture_keys = set()
    for _, _, _, t in rows:
        found_texture_keys.update([k for k in texture_candidate_keys if k in t])
    texture_keys = sorted(found_texture_keys)

    feature_keys = brightness_keys + contour_keys + texture_keys

    X_list = []
    for _, b, c, t in rows:
        vec = []
        vec.extend(_normalize_feature_vector(b, brightness_keys))
        vec.extend(_normalize_feature_vector(c, contour_keys))
        vec.extend(_normalize_feature_vector(t, texture_keys))
        X_list.append(np.array(vec, dtype=float))

    X = np.vstack(X_list)
    X = _standardize_matrix(X)
    return kept_paths, X, feature_keys


def select_diverse_images(folder_path: str, k: int = 100, seed: int = 42) -> List[str]:
    """
    Выбирает k разнообразных изображений из папки, основываясь на признаках.
    Возвращает список путей к выбранным изображениям.
    """
    paths, X, _ = build_feature_matrix(folder_path)
    idxs = _farthest_point_sampling(X, min(k, len(paths)), seed=seed)
    return [paths[i] for i in idxs]


def save_selection(selected_paths: List[str], out_json: str) -> None:
    """
    Сохраняет выбранные пути в JSON.
    """
    data = {"selected": selected_paths}
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Select diverse reference images")
    parser.add_argument("folder", type=str, help="Path to images folder")
    parser.add_argument("--k", type=int, default=100, help="Number of images to select")
    parser.add_argument("--out", type=str, default="selected_reference.json", help="Output JSON path")
    args = parser.parse_args()

    selected = select_diverse_images(args.folder, k=args.k)
    save_selection(selected, args.out)
    print(f"Saved {len(selected)} selected images to {args.out}")


