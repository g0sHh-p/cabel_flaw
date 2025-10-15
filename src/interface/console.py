#!/usr/bin/env python3
"""
Консольный скрипт для пакетной обработки изображений канатов
на наличие дефектов согласно ТЗ
"""

import sys
import os
import argparse
import time
from pathlib import Path
import pandas as pd

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import (
    load_defect_detection_model,
    DefectDetectionModel
)


def batch_analyze_images(model, input_folder, output_file="results.csv"):
    """
    Пакетная детекция дефектов
    
    Args:
        model: Модель детекции дефектов
        input_folder: Папка с изображениями для анализа
        output_file: Файл для сохранения результатов
    
    Returns:
        list: Результаты анализа
    """
    results = []
    image_files = []
    
    # Собираем все изображения
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(input_folder).glob(ext))
        image_files.extend(Path(input_folder).glob(ext.upper()))
    
    if not image_files:
        print(f"В папке {input_folder} не найдено изображений")
        return results
    
    print(f"Найдено {len(image_files)} изображений для анализа")
    print("Начинаем пакетный анализ...")
    
    start_time = time.time()
    
    for i, img_path in enumerate(image_files, 1):
        print(f"Анализируем {i}/{len(image_files)}: {img_path.name}")
        
        try:
            # Анализируем изображение
            result = model.detect_defects(str(img_path))

            # Готовим статус
            status_str = "ДЕФЕКТ ОБНАРУЖЕН" if result['has_defect'] else "ДЕФЕКТОВ НЕТ"

            # Сохраняем только файл и статус (для pandas-отчёта)
            results.append({
                'file': img_path.name,
                'status': status_str
            })

            # Вывод в консоль
            print(f"   {status_str} (уверенность: {result['defect_confidence']:.3f})")
            
        except Exception as e:
            print(f"   Ошибка: {e}")
            results.append({
                'file': img_path.name,
                'status': f"ОШИБКА: {e}"
            })
    
    # Сохраняем результаты через pandas
    if results:
        save_results_to_csv(results, output_file)
    
    # Выводим статистику
    total_time = time.time() - start_time
    defect_count = sum(1 for r in results if r['has_defect'] is True)
    error_count = sum(1 for r in results if r['has_defect'] is None)
    success_count = len(results) - error_count
    
    print(f"\nСтатистика анализа:")
    print(f"   Всего изображений: {len(results)}")
    print(f"   Успешно обработано: {success_count}")
    print(f"   Ошибок: {error_count}")
    print(f"   Дефектов обнаружено: {defect_count}")
    print(f"   Процент дефектов: {defect_count/success_count*100:.1f}%" if success_count > 0 else "   Процент дефектов: N/A")
    print(f"   Общее время: {total_time:.2f} сек")
    print(f"   Среднее время на изображение: {total_time/len(results):.3f} сек")
    print(f"   Результаты сохранены в: {output_file}")
    
    return results


def save_results_to_csv(results, output_file):
    """
    Сохраняет результаты в CSV через pandas.DataFrame
    """
    # Минимальный отчёт: файл — статус
    df = pd.DataFrame(results, columns=['file', 'status'])
    df.to_csv(output_file, index=False, encoding='utf-8')


# Удалено создание модели из консольного скрипта — оставлен только пакетный анализ папки


def main():
    """Основная функция консольного скрипта"""
    parser = argparse.ArgumentParser(
        description="Консольный скрипт: пакетный анализ изображений на дефекты",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Пример:
   python console.py analyze --input data/etln_proc --output results.csv --model defect_detection_model.pkl
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Команда анализа (только папка)
    analyze_parser = subparsers.add_parser('analyze', help='Пакетный анализ изображений (папка)')
    analyze_parser.add_argument('--input', '-i', required=True, 
                               help='Папка с изображениями для анализа')
    analyze_parser.add_argument('--output', '-o', default='results.csv',
                               help='Файл для сохранения результатов (по умолчанию: results.csv)')
    analyze_parser.add_argument('--model', '-m', default='defect_detection_model.pkl',
                               help='Путь к модели (по умолчанию: defect_detection_model.pkl)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'analyze':
            # Загружаем модель
            if os.path.exists(args.model):
                print(f"Загружаем модель из {args.model}")
                model = load_defect_detection_model(args.model)
                print("Модель загружена успешно")
            else:
                print(f"Модель {args.model} не найдена")
                print("Создайте модель заранее (через Streamlit или отдельный скрипт)")
                return
            
            # Проверяем входной путь
            input_path = Path(args.input)
            if input_path.is_dir():
                # Пакетный анализ
                batch_analyze_images(model, str(input_path), args.output)
            else:
                print(f"Путь {args.input} не существует")
                return
    
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
