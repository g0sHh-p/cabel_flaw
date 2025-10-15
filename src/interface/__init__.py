"""
Интерфейсы для детекции дефектов канатов

Этот модуль содержит интерфейсы для взаимодействия с системой детекции дефектов:
- Streamlit веб-приложение для анализа единичного изображения
- Консольный скрипт для пакетной обработки изображений
"""

__version__ = "1.0.0"
__author__ = "Cable Flaw Detection System"

# Импорты для удобного использования
from .streamlit import main as run_streamlit_app
from .console import main as run_console_script

__all__ = [
    'run_streamlit_app',
    'run_console_script'
]
