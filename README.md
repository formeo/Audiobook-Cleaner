# Audiobook Cleaner

Очистка аудиокниг от фоновой музыки и шумов с использованием нейросетевых моделей MDX-Net, VR и Roformer.

Python-обёртка над [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) — теми же моделями, что использует UVR5, но без проблем с большими файлами и 32-битной архитектурой.

## Возможности

- Отделение голоса диктора от фоновой музыки
- Удаление шумов (шипение, гул, помехи)
- Удаление эха и реверберации
- Автоматическая нарезка больших файлов
- Пакетная обработка директорий
- Работа на CPU и GPU (NVIDIA CUDA)
- Поддержка форматов: MP3, WAV, FLAC, M4A, OGG, OPUS

## Установка

### Базовая установка

```bash
pip install audio-separator[cpu]
```

### FFmpeg (обязательно)

```bash
# Windows (winget)
winget install ffmpeg

# Windows (chocolatey)
choco install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Установка с GPU (NVIDIA, рекомендуется)

Для ускорения в 2-3 раза на видеокартах NVIDIA:

```bash
# 1. PyTorch с CUDA
pip uninstall torch torchaudio torchvision -y
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. ONNX Runtime с CUDA
pip install onnxruntime-gpu==1.18.1

# 3. cuDNN (необходим для ONNX Runtime)
pip install nvidia-cudnn-cu12==8.9.7.29
```

### Проверка GPU

```bash
# PyTorch видит CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ONNX Runtime видит CUDA
python -c "import onnxruntime as ort; print('Providers:', ort.get_available_providers())"
```

Должно быть:
- `CUDA: True NVIDIA GeForce ...`
- `Providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

## Использование

### Базовое

```bash
# Обработать один файл
python audiobook_cleaner.py audiobook.mp3

# Обработать директорию
python audiobook_cleaner.py --dir /path/to/audiobooks
```

### Большие файлы (рекомендуется)

Для файлов >100 МБ используйте нарезку на части:

```bash
# Нарезка по 5 минут (рекомендуется)
python audiobook_cleaner.py audiobook.mp3 --chunk-duration 300 --no-denoise

# Нарезка по 10 минут (если много RAM)
python audiobook_cleaner.py audiobook.mp3 --chunk-duration 600 --no-denoise
```

Скрипт автоматически нарежет файл, обработает части и склеит результат.

### Выбор модели

```bash
# Список доступных моделей
python audiobook_cleaner.py --list-models

# Использовать конкретную модель
python audiobook_cleaner.py audiobook.mp3 --model vocals_kim
```

### Все параметры

```
Аргументы:
  input                   Входной аудиофайл
  --dir, -d               Обработать все файлы в директории
  --output, -o            Выходная директория
  --model, -m             Модель для сепарации (default: vocals)
  --no-denoise            Без дополнительного шумоподавления
  --cpu                   Принудительно использовать CPU
  --format, -f            Формат выхода: mp3, wav, flac (default: mp3)
  --sample-rate, -sr      Частота дискретизации (default: 44100)
  --segment-size, -s      Размер сегмента в секундах (default: 64)
  --chunk-duration, -c    Нарезка больших файлов по N секунд (default: 0 = выкл)
  --list-models, -l       Показать список моделей
```

## Модели

### Для отделения голоса от музыки

| Шорткат | Модель | Описание |
|---------|--------|----------|
| `vocals` | UVR-MDX-NET-Voc_FT.onnx | По умолчанию, хороший баланс |
| `vocals_kim` | Kim_Vocal_2.onnx | Альтернатива, иногда лучше |
| `roformer` | BS-Roformer | Лучшее качество, но медленнее и требует больше VRAM |

### Для удаления шумов

| Шорткат | Модель | Описание |
|---------|--------|----------|
| `denoise` | UVR-DeNoise.pth | Стандартное шумоподавление |
| `denoise_lite` | UVR-DeNoise-Lite.pth | Облегчённая версия |
| `roformer_denoise` | Mel-Roformer-Denoise | Лучшее качество |

### Для удаления эха/реверберации

| Шорткат | Модель | Описание |
|---------|--------|----------|
| `dereverb` | UVR-DeEcho-DeReverb.pth | Эхо + реверберация |
| `dereverb_normal` | UVR-De-Echo-Normal.pth | Только эхо |

## Примеры

```bash
# Быстрая обработка без шумоподавления
python audiobook_cleaner.py book.mp3 --no-denoise

# Большой файл — нарезка по 5 минут
python audiobook_cleaner.py book.mp3 --chunk-duration 300 --no-denoise

# Максимальное качество (медленно)
python audiobook_cleaner.py book.mp3 --model roformer --chunk-duration 300

# Обработка на слабой машине
python audiobook_cleaner.py book.mp3 --chunk-duration 120 --segment-size 16 --cpu --no-denoise

# Пакетная обработка в WAV
python audiobook_cleaner.py --dir ./books --output ./clean --format wav

# Очистка записи чёрного ящика (CVR) от шума
python audiobook_cleaner.py cvr.mp3 --model denoise --no-denoise
```

## Скорость обработки

Примерное время обработки 5-минутного чанка (300 сек):

| Конфигурация | Время | Примечание |
|--------------|-------|------------|
| CPU (i7-8700) | 2:30 | Без GPU |
| GPU GTX 1660 Super (PyTorch) | 1:40 | ONNX → PyTorch конвертация |
| GPU GTX 1660 Super (ONNX) | 0:30-0:40 | Полное ускорение |

Для 7-часовой аудиокниги (85 чанков по 5 мин):
- CPU: ~3.5 часа
- GPU (PyTorch): ~2.5 часа
- GPU (ONNX): ~45-60 минут

## Как это работает

1. **Шаг 1**: Модель MDX-Net/Roformer разделяет аудио на "голос" и "инструментал"
2. **Шаг 2** (опционально): Модель DeNoise убирает остаточные шумы

На выходе:
- `*_(Vocals).mp3` — чистый голос диктора
- `*_(Instrumental).mp3` — отделённая музыка/шум

## Требования

- Python 3.9+
- FFmpeg
- 8+ GB RAM (16 GB рекомендуется для больших файлов)
- NVIDIA GPU с 4+ GB VRAM (опционально, для ускорения)

## Решение проблем

### "bad allocation" / Out of memory

```bash
# Используйте нарезку + маленький сегмент
python audiobook_cleaner.py book.mp3 --chunk-duration 300 --segment-size 16 --no-denoise
```

### "Model file not found"

Модели скачиваются автоматически. Проверьте интернет-соединение.

### GPU не используется

1. Проверьте установку:
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

2. Если `CUDAExecutionProvider` отсутствует:
```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu==1.18.1
pip install nvidia-cudnn-cu12==8.9.7.29
```

### "CUDAExecutionProvider not available" 

Не хватает cuDNN. Установите:
```bash
pip install nvidia-cudnn-cu12==8.9.7.29
```

### "LoadLibrary failed with error 126"

ONNX Runtime не находит CUDA DLL. Варианты:
1. Установите cuDNN через pip (см. выше)
2. Или установите [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) системно

### Медленная обработка даже с GPU

Если в логах видите:
```
Model converted from onnx to pytorch due to segment size not matching dim_t
```

Модель конвертируется в PyTorch, что медленнее. Это нормально для некоторых конфигураций.

## Лицензия

MIT

## Благодарности

- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
- [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
- Авторам моделей MDX-Net, Demucs и Roformer
