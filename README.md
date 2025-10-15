# Cable Flaw Detection

Проект для детекции дефектов на изображениях металлических канатов. Содержит:
- Streamlit-приложение для анализа единичного изображения
- Консольный скрипт для пакетного анализа папки изображений

## Требования
- Python 3.12 (рекомендуется виртуальное окружение)
- Windows PowerShell (или cmd)

## Предобработка изображений
Исходные эталонные изображения кладите в `data/etalon_imgs/`. Предобработанные будут сохранены в `data/etln_proc/`.
```powershell
python -m src.core.preproc
# или явно
python -c "from src.core.preproc import preprocess_images; preprocess_images('data/etalon_imgs','data/etln_proc')"
```

## Запуск Streamlit
```powershell
streamlit run src/interface/streamlit.py
```
В боковой панели нажмите «Загрузить модель». Если файл `defect_detection_model.pkl` отсутствует, модель будет создана из изображений в `data/etln_proc/`.

## Консольный скрипт (пакетный анализ)
Скрипт анализирует все изображения в папке и формирует CSV-отчёт через pandas в формате «file — status».
```powershell
python src/interface/console.py analyze --input data/etln_proc --output results.csv --model defect_detection_model.pkl
```
- status: «ДЕФЕКТ ОБНАРУЖЕН», «ДЕФЕКТОВ НЕТ» или «ОШИБКА: …»

## Структура проекта
```
.
├─ data/
│  ├─ etalon_imgs/      # исходные изображения эталонов
│  └─ etln_proc/        # предобработанные изображения (генерируется)
├─ src/
│  ├─ core/
│  │  ├─ preproc.py                 # предобработка
│  │  ├─ Brigh_analys.py            # признаки яркости
│  │  ├─ char_comp.py               # контурные признаки
│  │  ├─ texture_analys.py          # GLCM/Haralick-признаки (встроенная реализация)
│  │  ├─ defect_detection_model.py  # модель детекции дефектов
│  │  ├─ select_reference.py        # выборка K разнообразных эталонов
│  │  └─ __init__.py
│  └─ interface/
│     ├─ streamlit.py   # веб-интерфейс (бинарный вывод: дефект/нет)
│     └─ console.py     # консольный пакетный анализ (pandas CSV)
├─ .gitignore
├─ requirements.txt
└─ README.md
```