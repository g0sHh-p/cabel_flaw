#!/usr/bin/env python3
"""
Streamlit приложение для анализа единичного изображения каната
на наличие дефектов согласно ТЗ
"""

import streamlit as st
import sys
import os
import cv2
import numpy as np
from PIL import Image
import time

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import (
    create_defect_detection_model,
    load_defect_detection_model,
    DefectDetectionModel,
    preprocess_images
)

# Настройка страницы
st.set_page_config(
    page_title="Детекция дефектов канатов",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🔍 Детекция дефектов канатов")
st.markdown("**Анализ единичного изображения на наличие дефектов согласно ТЗ**")

# Боковая панель
st.sidebar.header("⚙️ Настройки")

# Инициализация состояния сессии
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

def load_or_create_model():
    """Загружает или создает модель детекции дефектов"""
    model_path = "defect_detection_model.pkl"
    
    if os.path.exists(model_path):
        try:
            model = load_defect_detection_model(model_path)
            st.session_state.model = model
            st.session_state.model_loaded = True
            return True
        except Exception as e:
            st.error(f"Ошибка при загрузке модели: {e}")
            return False
    else:
        st.warning("Модель не найдена. Создаем новую модель...")
        
        processed_folder = "data/etln_proc"
        if os.path.exists(processed_folder):
            try:
                with st.spinner("Создание модели детекции дефектов..."):
                    model = create_defect_detection_model(processed_folder, model_path)
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                    st.success("Модель успешно создана!")
                    return True
            except Exception as e:
                st.error(f"Ошибка при создании модели: {e}")
                return False
        else:
            st.error(f"Папка с эталонными изображениями не найдена: {processed_folder}")
            return False

def analyze_image(image_file):
    """Анализирует загруженное изображение"""
    if not st.session_state.model_loaded:
        st.error("Модель не загружена!")
        return None
    
    try:
        # Сохраняем временный файл
        temp_path = f"temp_{int(time.time())}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        # Анализируем изображение
        with st.spinner("Анализируем изображение..."):
            results = st.session_state.model.detect_defects(temp_path)
        
        # Удаляем временный файл
        os.remove(temp_path)
        
        return results
        
    except Exception as e:
        st.error(f"Ошибка при анализе изображения: {e}")
        return None

def display_results(results):
    """Отображает только итог: дефект обнаружен / не обнаружен"""
    if not results:
        return
    if results.get('has_defect'):
        st.error("🚨 **ДЕФЕКТ ОБНАРУЖЕН**")
        confidence = results.get('defect_confidence', 0.0) * 100.0
        st.write(f"Уверенность: {confidence:.1f}%")
    else:
        st.success("✅ **ДЕФЕКТОВ НЕТ**")
        confidence = results.get('defect_confidence', 0.0) * 100.0
        st.write(f"Уверенность: {confidence:.1f}%")

def main():
    """Основная функция приложения"""
    
    # Загружаем модель
    if not st.session_state.model_loaded:
        if st.sidebar.button("🔄 Загрузить модель"):
            load_or_create_model()
    
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Модель загружена")
        
        # Информация о модели
        with st.sidebar.expander("ℹ️ Информация о модели"):
            summary = st.session_state.model.get_model_summary()
            st.write(f"**Название:** {summary['model_name']}")
            st.write(f"**Создана:** {summary['created_at']}")
            st.write(f"**Изображений:** {summary['image_count']}")
            st.write(f"**Параметров яркости:** {summary['statistics_summary']['brightness']['parameter_count']}")
            st.write(f"**Параметров контуров:** {summary['statistics_summary']['contour']['parameter_count']}")
            st.write(f"**Параметров текстуры:** {summary['statistics_summary']['texture']['parameter_count']}")
        
        # Основной интерфейс
        st.header("📤 Загрузка изображения")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите изображение каната для анализа",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Поддерживаемые форматы: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Отображаем загруженное изображение
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📷 Загруженное изображение")
                image = Image.open(uploaded_file)
                st.image(image, caption="Исходное изображение", use_container_width=True)
                
                # Информация об изображении
                st.write(f"**Размер:** {image.size[0]} x {image.size[1]} пикселей")
                st.write(f"**Формат:** {image.format}")
                st.write(f"**Режим:** {image.mode}")
            
            with col2:
                st.subheader("🔍 Результаты анализа")
                
                # Кнопка анализа
                if st.button("🚀 Анализировать изображение", type="primary"):
                    results = analyze_image(uploaded_file)
                    if results:
                        display_results(results)
        
        # Дополнительные настройки
        with st.sidebar.expander("⚙️ Дополнительные настройки"):
            st.write("**Пороги детекции:**")
            thresholds = st.session_state.model.defect_thresholds
            for key, value in thresholds.items():
                st.write(f"• {key}: {value}")
        
        # Информация о поддерживаемых дефектах
        with st.sidebar.expander("📋 Поддерживаемые дефекты"):
            defects = [
                "1. Отклонение по диаметру каната",
                "2. Отсутствие проволок в пряди",
                "3. Перекрещивание проволок",
                "4. Перекрут пряди (дефект «жучок»)",
                "5. Неравномерный зазор между прядями",
                "6. Выдавливание сердечника или проволоки",
                "7. Отсутствие сердечника в канате",
                "8. Дефект «бурунда»",
                "9. Отсутствие пряди",
                "10. Неправильная свивка"
            ]
            for defect in defects:
                st.write(defect)
    
    else:
        st.warning("⚠️ Модель не загружена. Нажмите кнопку 'Загрузить модель' в боковой панели.")
        
        # Инструкции
        st.info("""
        **Инструкции по использованию:**
        
        1. Убедитесь, что в папке `data/etln_proc/` есть предобработанные эталонные изображения
        2. Нажмите кнопку "Загрузить модель" в боковой панели
        3. Загрузите изображение каната для анализа
        4. Нажмите "Анализировать изображение"
        5. Просмотрите результаты анализа
        
        **Требования:**
        - Точность: ≥ 80%
        - Время обработки: ≤ 10 сек/изображение
        - Поддержка форматов: JPG, JPEG, PNG, BMP
        """)

if __name__ == "__main__":
    main()
