# Cable Flaw Detection

Проект для детекции дефектов на изображениях металлических канатов. Содержит:
- Streamlit-приложение для анализа единичного изображения
- Консольный скрипт для пакетного анализа папки изображений
- Пайплайн предобработки, построения/загрузки модели и утилиты отбора эталонов

## Требования
- Python 3.12 (рекомендуется виртуальное окружение)
- Windows PowerShell (или cmd)

Зависимости устанавливаются из `requirements.txt` (см. раздел Установка).

## Установка
```powershell
cd C:\Users\EGOR\Desktop\cabel_flaw
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

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
# По умолчанию откроется http://localhost:8501
```
В боковой панели нажмите «Загрузить модель». Если файл `defect_detection_model.pkl` отсутствует, модель будет создана из изображений в `data/etln_proc/`.

## Консольный скрипт (пакетный анализ)
Скрипт анализирует все изображения в папке и формирует CSV-отчёт через pandas в формате «file — status».
```powershell
python src/interface/console.py analyze --input data/etln_proc --output results.csv --model defect_detection_model.pkl
```
- status: «ДЕФЕКТ ОБНАРУЖЕН», «ДЕФЕКТОВ НЕТ» или «ОШИБКА: …»

## Отбор эталонов (опционально)
Выбор разнообразных K изображений по совокупности признаков (яркость/контуры/текстура):
```powershell
python -m src.core.select_reference "data/etalon_imgs" --k 100 --out selected_reference.json
```
При необходимости скопируйте выбранные файлы в `data/etln_proc/` и пересоберите модель.

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
│  │  ├─ reference_model.py         # модель-эталон (статистики/пороги)
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

## Советы по совместимости
- Рекомендуемые версии для стабильной работы в Windows:
  - numpy==2.2.2
  - scipy==1.15.1
  - opencv-python==4.12.0.88
  - pandas==2.2.3
- Если возникают ошибки сборки/импорта:
  - Активируйте venv: `.\.venv\Scripts\Activate.ps1`
  - Переустановите пакет без кэша: `pip install --force-reinstall --no-cache-dir <pkg>==<ver>`

## Типичный рабочий процесс
1) Скопируйте исходные изображения в `data/etalon_imgs/`
2) Запустите предобработку → `data/etln_proc/`
3) Откройте Streamlit и нажмите «Загрузить модель» (создастся при отсутствии)
4) Анализируйте единичные изображения в веб-интерфейсе
5) Для пакетной проверки папки используйте консольный скрипт (CSV-отчёт)

## Лицензия
Укажите здесь условия лицензирования (при необходимости).
