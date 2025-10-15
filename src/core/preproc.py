import os
import cv2
import numpy as np


def preprocess_images(input_folder, output_folder):
    """
    Обрабатывает эталонные изображения канатов и сохраняет результаты
    """
    
    # Создаем папку для обработанных изображений, если она не существует
    os.makedirs(output_folder, exist_ok=True)
    
    # Проверяем существование исходной папки
    if not os.path.exists(input_folder):
        print(f"Ошибка: Папка {input_folder} не существует!")
        return
    
    # Обрабатываем каждое изображение в папке
    for img_name in os.listdir(input_folder):
        try:
            # Формируем полный путь к изображению
            img_path = os.path.join(input_folder, img_name)
            
            # Загружаем изображение
            image = cv2.imread(img_path)
            
            # Проверяем, что изображение загружено корректно
            if image is None:
                print(f"Не удалось загрузить изображение {img_path}. Пропускаем.")
                continue
            
            # Преобразование в градации серого
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            background = cv2.GaussianBlur(img_gray, (51, 51), 0)
    
            # Вычитаем фон и нормализуем
            corrected = cv2.subtract(img_gray, background)
            corrected = cv2.normalize(corrected, np.zeros(1), 0, 255, cv2.NORM_MINMAX)

            # Улучшение контраста с помощью CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
            enhanced = clahe.apply(corrected)  # Применяем CLAHE к скорректированному изображению
            
            # Размытие после контрастирования
            img_median = cv2.medianBlur(enhanced, 3)
            img_gaussian = cv2.GaussianBlur(img_median, (7, 7), 0)

            # Адаптивная бинаризация
            adp_binary = cv2.adaptiveThreshold(
                img_gaussian, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Морфологические операции для удаления шума
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            closing = cv2.morphologyEx(adp_binary, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            
            # Создаем полный путь для сохранения
            output_path = os.path.join(output_folder, f"processed_{img_name}")
            
            # Сохраняем обработанное изображение
            cv2.imwrite(output_path, opening)  # Сохраняем результат морфологических операций
            
            print(f"Успешно обработан: {img_name}")
            
        except Exception as e:
            print(f"Невозможно обработать файл {img_name}. Ошибка: {str(e)}")

    print("Обработка всех изображений завершена!")


# Пример использования
if __name__ == "__main__":
    input_folder = "C:/Users/EGOR/Desktop/cabel_flaw/data/etalon_imgs"
    output_folder = "C:/Users/EGOR/Desktop/cabel_flaw/data/etln_proc"
    preprocess_images(input_folder, output_folder)