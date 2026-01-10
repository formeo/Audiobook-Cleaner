#!/usr/bin/env python3
"""
Очистка аудиокниг от фоновой музыки и шумов с использованием MDX-Net моделей.
Использует audio-separator — Python-враппер над моделями UVR5/MDX-Net.

Установка зависимостей:
    pip install audio-separator[gpu]  # для GPU (CUDA)
    pip install audio-separator[cpu]  # для CPU

Модели скачиваются автоматически при первом запуске (~100-300 MB каждая).
"""

import argparse
import sys
import subprocess
import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Популярные модели для разных задач
MODELS = {
    # Для отделения вокала/речи от музыки (MDX-Net, .onnx)
    'vocals': 'UVR-MDX-NET-Voc_FT.onnx',
    'vocals_kim': 'Kim_Vocal_2.onnx',
    'vocals_kim1': 'Kim_Vocal_1.onnx',

    # Для удаления шумов (VR, .pth)
    'denoise': 'UVR-DeNoise.pth',
    'denoise_lite': 'UVR-DeNoise-Lite.pth',

    # Для удаления реверберации/эха (VR, .pth)
    'dereverb': 'UVR-DeEcho-DeReverb.pth',
    'dereverb_normal': 'UVR-De-Echo-Normal.pth',

    # Универсальные MDX-Net модели
    'mdx_main': 'UVR_MDXNET_Main.onnx',
    'mdx_kara': 'UVR_MDXNET_KARA_2.onnx',

    # Roformer — лучшее качество, но медленнее
    'roformer': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'roformer_kim': 'vocals_mel_band_roformer.ckpt',
    'roformer_denoise': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
}


def format_time(seconds: float) -> str:
    """Форматирует секунды в человекочитаемый вид."""
    if seconds < 60:
        return f"{seconds:.0f}с"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}м {seconds % 60:.0f}с"
    else:
        return f"{seconds // 3600:.0f}ч {(seconds % 3600) // 60:.0f}м"


def _process_chunk_worker(args: tuple) -> dict:
    """
    Воркер для параллельной обработки чанка.
    Возвращает dict с результатами.
    """
    (
        chunk_idx, chunk_path, chunk_output_dir,
        model_name, denoise, use_gpu, sample_rate, segment_size
    ) = args

    try:
        from audio_separator.separator import Separator
    except ImportError:
        return {'idx': chunk_idx, 'error': 'audio-separator not installed', 'vocals': [], 'instrumental': []}

    chunk_output_dir = Path(chunk_output_dir)
    chunk_path = Path(chunk_path)

    # Проверяем, обработан ли уже
    if chunk_output_dir.exists():
        existing = list(chunk_output_dir.glob("*Vocal*.wav"))
        if existing:
            vocals = [f.absolute() for f in chunk_output_dir.glob("*.wav")
                      if 'vocal' in f.stem.lower() or 'no_noise' in f.stem.lower()]
            instrumental = [f.absolute() for f in chunk_output_dir.glob("*.wav")
                            if 'instrument' in f.stem.lower()]
            return {'idx': chunk_idx, 'skipped': True, 'vocals': vocals, 'instrumental': instrumental}

    chunk_output_dir.mkdir(exist_ok=True)

    try:
        # Создаём сепаратор для этого воркера
        separator = Separator(
            output_dir=str(chunk_output_dir),
            output_format='wav',
            sample_rate=sample_rate,
            use_autocast=use_gpu,
            mdx_params={
                "hop_length": 1024,
                "segment_size": segment_size,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": False,
            }
        )
        separator.load_model(model_name)

        result = separator.separate(str(chunk_path))

        vocals = []
        instrumental = []
        vocals_file = None

        for f in result:
            fp = Path(f)
            if not fp.is_absolute():
                fp = chunk_output_dir / fp.name
            fp = fp.absolute()

            if 'vocal' in fp.stem.lower():
                vocals_file = fp
                vocals.append(fp)
            elif 'instrument' in fp.stem.lower():
                instrumental.append(fp)

        # Denoise если нужно
        if denoise and vocals_file:
            denoise_separator = Separator(
                output_dir=str(chunk_output_dir),
                output_format='wav',
                sample_rate=sample_rate,
                use_autocast=use_gpu,
            )
            denoise_separator.load_model('UVR-DeNoise.pth')
            denoised = denoise_separator.separate(str(vocals_file))

            for f in denoised:
                fp = Path(f)
                if not fp.is_absolute():
                    fp = chunk_output_dir / fp.name
                fp = fp.absolute()
                if 'no_noise' in fp.stem.lower() or 'no noise' in fp.stem.lower():
                    vocals.append(fp)

        return {'idx': chunk_idx, 'vocals': vocals, 'instrumental': instrumental}

    except Exception as e:
        return {'idx': chunk_idx, 'error': str(e), 'vocals': [], 'instrumental': []}


def get_audio_duration(file_path: str) -> float:
    """Получает длительность аудиофайла в секундах через ffprobe."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', str(file_path)
            ],
            capture_output=True, text=True, check=True
        )
        info = json.loads(result.stdout)
        return float(info['format']['duration'])
    except Exception as e:
        logger.warning(f"Не удалось получить длительность: {e}")
        return 0


def split_audio(input_file: str, output_dir: Path, chunk_duration: int) -> list[Path]:
    """Нарезает аудиофайл на части указанной длительности."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / "chunk_%04d.wav"

    # Проверяем, есть ли уже нарезанные чанки
    existing_chunks = sorted(output_dir.glob("chunk_*.wav"))
    if existing_chunks:
        logger.info(f"Найдено {len(existing_chunks)} ранее нарезанных частей")
        return existing_chunks

    logger.info(f"Нарезка на части по {chunk_duration} секунд...")

    subprocess.run([
        'ffmpeg', '-y', '-i', str(input_file),
        '-f', 'segment', '-segment_time', str(chunk_duration),
        '-c:a', 'pcm_s16le',
        str(pattern)
    ], capture_output=True, check=True)

    chunks = sorted(output_dir.glob("chunk_*.wav"))
    logger.info(f"Создано {len(chunks)} частей")
    return chunks


def concat_audio(input_files: list[Path], output_file: Path, output_format: str):
    """Склеивает аудиофайлы в один."""
    if not input_files:
        return

    list_file = output_file.parent / "concat_list.txt"

    # Создаём файл со списком для ffmpeg с абсолютными путями (без BOM!)
    with open(list_file, 'w', encoding='ascii', errors='replace') as f:
        for file in input_files:
            # Используем прямые слэши для совместимости с ffmpeg
            abs_path = str(file.absolute()).replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    logger.info(f"Склейка {len(input_files)} файлов...")

    # Параметры кодирования в зависимости от формата
    codec_args = []
    if output_format == 'mp3':
        codec_args = ['-c:a', 'libmp3lame', '-q:a', '2']
    elif output_format == 'flac':
        codec_args = ['-c:a', 'flac']
    else:  # wav
        codec_args = ['-c:a', 'pcm_s16le']

    result = subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(list_file.absolute()),
        *codec_args,
        str(output_file)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Ошибка ffmpeg: {result.stderr}")
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    list_file.unlink()
    logger.info(f"Результат сохранён: {output_file}")


def clean_audiobook(
        input_file: str,
        output_dir: Optional[str] = None,
        model_name: str = 'UVR-MDX-NET-Voc_FT.onnx',
        denoise: bool = True,
        use_gpu: bool = True,
        output_format: str = 'mp3',
        sample_rate: int = 44100,
        segment_size: int = 64,
        chunk_duration: int = 0,
        workers: int = 1,
) -> list[Path]:
    """
    Очищает аудиокнигу от фоновой музыки и шумов.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    if chunk_duration > 0:
        return clean_audiobook_chunked(
            input_file=input_file,
            output_dir=output_dir,
            model_name=model_name,
            denoise=denoise,
            use_gpu=use_gpu,
            output_format=output_format,
            sample_rate=sample_rate,
            segment_size=segment_size,
            chunk_duration=chunk_duration,
            workers=workers,
        )

    return _process_single_file(
        input_file=input_file,
        output_dir=output_dir,
        model_name=model_name,
        denoise=denoise,
        use_gpu=use_gpu,
        output_format=output_format,
        sample_rate=sample_rate,
        segment_size=segment_size,
    )


def clean_audiobook_chunked(
        input_file: str,
        output_dir: Optional[str] = None,
        model_name: str = 'UVR-MDX-NET-Voc_FT.onnx',
        denoise: bool = True,
        use_gpu: bool = True,
        output_format: str = 'mp3',
        sample_rate: int = 44100,
        segment_size: int = 64,
        chunk_duration: int = 600,
        workers: int = 1,
) -> list[Path]:
    """Обрабатывает большой файл по частям с сохранением прогресса."""

    input_path = Path(input_file)

    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_cleaned"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Директории для промежуточных файлов (сохраняются при падении)
    chunks_dir = output_path / "_chunks"
    processed_dir = output_path / "_processed"
    chunks_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    duration = get_audio_duration(input_file)
    logger.info(f"Входной файл: {input_path}")
    logger.info(f"Длительность: {duration / 60:.1f} минут")
    logger.info(f"Режим: нарезка по {chunk_duration} секунд")
    logger.info(f"Промежуточные файлы: {output_path}")
    if workers > 1:
        logger.info(f"Параллельная обработка: {workers} воркеров")

    # 1. Нарезаем на части
    chunks = split_audio(input_file, chunks_dir, chunk_duration)

    if not chunks:
        logger.error("Не удалось нарезать файл")
        return []

    total_chunks = len(chunks)

    # Подготовка задач
    tasks = []
    for i, chunk in enumerate(chunks, 1):
        chunk_output_dir = processed_dir / f"chunk_{i:04d}"
        tasks.append((
            i, str(chunk), str(chunk_output_dir),
            model_name, denoise, use_gpu, sample_rate, segment_size
        ))

    vocals_files = []
    instrumental_files = []
    start_time = time.time()
    processed_count = 0
    skipped_count = 0

    if workers > 1:
        # Параллельная обработка
        logger.info(f"Запуск {workers} воркеров...")

        with mp.Pool(processes=workers) as pool:
            for result in pool.imap(_process_chunk_worker, tasks):
                idx = result['idx']

                if result.get('skipped'):
                    logger.info(f"[{idx}/{total_chunks}] Пропущен (уже обработан)")
                    skipped_count += 1
                elif result.get('error'):
                    logger.error(f"[{idx}/{total_chunks}] Ошибка: {result['error']}")
                else:
                    processed_count += 1
                    elapsed = time.time() - start_time
                    if processed_count > 0:
                        avg_time = elapsed / processed_count
                        remaining = avg_time * (total_chunks - processed_count - skipped_count)
                        eta = format_time(remaining)
                    else:
                        eta = "..."
                    logger.info(f"[{idx}/{total_chunks}] Готово | ETA: {eta}")

                vocals_files.extend(result.get('vocals', []))
                instrumental_files.extend(result.get('instrumental', []))
    else:
        # Последовательная обработка (оригинальная логика)
        try:
            from audio_separator.separator import Separator
        except ImportError:
            logger.error("Не установлен audio-separator!")
            logger.error("Установите: pip install audio-separator[gpu]  # или [cpu]")
            sys.exit(1)

        logger.info("Загрузка модели...")
        separator = Separator(
            output_dir=str(processed_dir),
            output_format='wav',
            sample_rate=sample_rate,
            use_autocast=use_gpu,
            mdx_params={
                "hop_length": 1024,
                "segment_size": segment_size,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": False,
            }
        )
        separator.load_model(model_name)

        denoise_separator = None
        if denoise:
            denoise_separator = Separator(
                output_dir=str(processed_dir),
                output_format='wav',
                sample_rate=sample_rate,
                use_autocast=use_gpu,
            )
            denoise_separator.load_model('UVR-DeNoise.pth')

        for i, chunk in enumerate(chunks, 1):
            chunk_output_dir = processed_dir / f"chunk_{i:04d}"

            existing_vocals = list(chunk_output_dir.glob("*Vocal*.wav")) if chunk_output_dir.exists() else []

            if existing_vocals:
                logger.info(f"[{i}/{total_chunks}] Пропуск (уже обработан): {chunk.name}")
                for f in chunk_output_dir.glob("*.wav"):
                    if 'vocal' in f.stem.lower() or 'no_noise' in f.stem.lower():
                        vocals_files.append(f.absolute())
                    elif 'instrument' in f.stem.lower():
                        instrumental_files.append(f.absolute())
                skipped_count += 1
                continue

            chunk_output_dir.mkdir(exist_ok=True)
            chunk_start = time.time()

            done = processed_count + skipped_count
            if processed_count > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count
                remaining = avg_time * (total_chunks - done)
                eta = format_time(remaining)
                speed = f"{avg_time:.1f}с/чанк"
            else:
                eta = "..."
                speed = "..."

            logger.info(f"\n[{i}/{total_chunks}] {chunk.name} | ETA: {eta} | {speed}")

            try:
                separator.output_dir = str(chunk_output_dir)
                result = separator.separate(str(chunk))

                vocals_file = None
                for f in result:
                    fp = Path(f)
                    if not fp.is_absolute():
                        fp = chunk_output_dir / fp.name
                    fp = fp.absolute()

                    if 'vocal' in fp.stem.lower():
                        vocals_file = fp
                        vocals_files.append(fp)
                    elif 'instrument' in fp.stem.lower():
                        instrumental_files.append(fp)

                if denoise and denoise_separator and vocals_file:
                    denoise_separator.output_dir = str(chunk_output_dir)
                    denoised = denoise_separator.separate(str(vocals_file))
                    for f in denoised:
                        fp = Path(f)
                        if not fp.is_absolute():
                            fp = chunk_output_dir / fp.name
                        fp = fp.absolute()
                        if 'no_noise' in fp.stem.lower() or 'no noise' in fp.stem.lower():
                            vocals_files.append(fp)

                processed_count += 1
                chunk_time = time.time() - chunk_start
                logger.info(f"    Готово за {format_time(chunk_time)}")

            except Exception as e:
                logger.error(f"Ошибка при обработке части {i}: {e}")
                continue

    # 4. Склеиваем результаты
    output_files = []

    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Обработка завершена за {format_time(total_time)}")
    logger.info(f"Обработано: {processed_count}, пропущено: {skipped_count}")
    logger.info(f"Склейка результатов...")

    if vocals_files:
        final_vocals = []
        seen_chunks = set()
        for f in reversed(vocals_files):
            # Преобразуем в Path если это строка
            if isinstance(f, str):
                f = Path(f)
            chunk_num = f.parent.name
            if chunk_num not in seen_chunks:
                seen_chunks.add(chunk_num)
                final_vocals.append(f)
        final_vocals.reverse()

        final_vocals.sort(key=lambda x: x.parent.name)

        vocals_output = output_path / f"{input_path.stem}_(Vocals).{output_format}"
        concat_audio(final_vocals, vocals_output, output_format)
        output_files.append(vocals_output)

    if instrumental_files:
        # Преобразуем в Path если строки
        instrumental_files = [Path(f) if isinstance(f, str) else f for f in instrumental_files]
        instrumental_files.sort(key=lambda x: x.parent.name)
        inst_output = output_path / f"{input_path.stem}_(Instrumental).{output_format}"
        concat_audio(instrumental_files, inst_output, output_format)
        output_files.append(inst_output)

    logger.info("\nГотово!")
    logger.info(f"Результаты:")
    for f in output_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  - {f} ({size_mb:.1f} MB)")

    logger.info(f"\nПромежуточные файлы сохранены в: {output_path}")
    logger.info(f"Для очистки удалите папки _chunks и _processed")

    return output_files


def _process_single_file(
        input_file: str,
        output_dir: Optional[str] = None,
        model_name: str = 'UVR-MDX-NET-Voc_FT.onnx',
        denoise: bool = True,
        use_gpu: bool = True,
        output_format: str = 'mp3',
        sample_rate: int = 44100,
        segment_size: int = 64,
) -> list[Path]:
    """Обрабатывает один файл."""
    try:
        from audio_separator.separator import Separator
    except ImportError:
        logger.error("Не установлен audio-separator!")
        logger.error("Установите: pip install audio-separator[gpu]  # или [cpu]")
        sys.exit(1)

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")

    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_cleaned"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Входной файл: {input_path}")
    logger.info(f"Выходная директория: {output_path}")
    logger.info(f"Модель: {model_name}")
    logger.info(f"GPU: {'да' if use_gpu else 'нет'}")
    logger.info(f"Размер сегмента: {segment_size} сек")

    separator = Separator(
        output_dir=str(output_path),
        output_format=output_format,
        sample_rate=sample_rate,
        use_autocast=use_gpu,
        mdx_params={
            "hop_length": 1024,
            "segment_size": segment_size,
            "overlap": 0.25,
            "batch_size": 1,
            "enable_denoise": denoise,
        }
    )

    output_files = []

    logger.info("Шаг 1: Отделение голоса от фоновой музыки...")
    separator.load_model(model_name)

    try:
        result = separator.separate(str(input_path))
        logger.info(f"Результат сепарации: {result}")

        for f in result:
            output_files.append(Path(f))

    except Exception as e:
        logger.error(f"Ошибка при сепарации: {e}")
        raise

    if denoise and output_files:
        logger.info("Шаг 2: Дополнительное шумоподавление...")

        vocals_file = None
        for f in output_files:
            if 'vocal' in f.stem.lower() or 'voice' in f.stem.lower():
                vocals_file = f
                break

        if vocals_file is None:
            vocals_file = output_files[0]

        try:
            separator.load_model('UVR-DeNoise.pth')
            denoised = separator.separate(str(vocals_file))
            logger.info(f"Результат шумоподавления: {denoised}")

            for f in denoised:
                if Path(f) not in output_files:
                    output_files.append(Path(f))

        except Exception as e:
            logger.warning(f"Шумоподавление не удалось: {e}")

    logger.info("Готово!")
    for f in output_files:
        logger.info(f"  - {f}")

    return output_files


def process_directory(
        input_dir: str,
        output_dir: Optional[str] = None,
        extensions: tuple = ('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.opus'),
        **kwargs
) -> dict[str, list[Path]]:
    """Обрабатывает все аудиофайлы в директории."""
    input_path = Path(input_dir)
    results = {}

    audio_files = [f for f in input_path.iterdir()
                   if f.is_file() and f.suffix.lower() in extensions]

    logger.info(f"Найдено {len(audio_files)} аудиофайлов")

    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Обработка [{i}/{len(audio_files)}]: {audio_file.name}")

        try:
            file_output_dir = None
            if output_dir:
                file_output_dir = Path(output_dir) / audio_file.stem

            result = clean_audiobook(
                str(audio_file),
                output_dir=str(file_output_dir) if file_output_dir else None,
                **kwargs
            )
            results[str(audio_file)] = result

        except Exception as e:
            logger.error(f"Ошибка: {e}")
            results[str(audio_file)] = []

    return results


def list_models():
    """Выводит список доступных моделей."""
    print("\nПредустановленные модели:")
    print("-" * 60)
    for key, model in MODELS.items():
        print(f"  {key:15} -> {model}")

    print("\nРекомендации:")
    print("-" * 60)
    print("  Аудиокниги с фоновой музыкой:")
    print("    --model vocals  (UVR-MDX-NET-Voc_FT.onnx)")
    print("    --model vocals_kim  (Kim_Vocal_2.onnx)")
    print()
    print("  Удаление шумов (шипение, гул):")
    print("    --model denoise  (UVR-DeNoise.pth)")
    print()
    print("  Удаление эха/реверберации:")
    print("    --model dereverb  (UVR-DeEcho-DeReverb.pth)")
    print()
    print("  Лучшее качество (медленнее):")
    print("    --model roformer")


def main():
    parser = argparse.ArgumentParser(
        description="Очистка аудиокниг от фоновой музыки и шумов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python audiobook_cleaner.py audiobook.mp3
  python audiobook_cleaner.py audiobook.mp3 --model vocals_kim
  python audiobook_cleaner.py audiobook.mp3 --no-denoise
  python audiobook_cleaner.py audiobook.mp3 --chunk-duration 300  # большие файлы
  python audiobook_cleaner.py --dir /path/to/audiobooks
  python audiobook_cleaner.py audiobook.mp3 --cpu
  python audiobook_cleaner.py --list-models
        """
    )

    parser.add_argument('input', nargs='?', help='Входной аудиофайл')
    parser.add_argument('--dir', '-d', help='Обработать директорию')
    parser.add_argument('--output', '-o', help='Выходная директория')
    parser.add_argument('--model', '-m', default='vocals',
                        help='Модель (default: vocals)')
    parser.add_argument('--no-denoise', action='store_true',
                        help='Без дополнительного шумоподавления')
    parser.add_argument('--cpu', action='store_true',
                        help='Использовать CPU')
    parser.add_argument('--format', '-f', default='mp3',
                        choices=['mp3', 'wav', 'flac'],
                        help='Формат выхода (default: mp3)')
    parser.add_argument('--sample-rate', '-sr', type=int, default=44100,
                        help='Sample rate (default: 44100)')
    parser.add_argument('--segment-size', '-s', type=int, default=64,
                        help='Размер сегмента в секундах (default: 64)')
    parser.add_argument('--chunk-duration', '-c', type=int, default=0,
                        help='Нарезать на части по N секунд (default: 0 = выкл)')
    parser.add_argument('--workers', '-w', type=int, default=1,
                        help='Количество параллельных воркеров для CPU (default: 1)')
    parser.add_argument('--list-models', '-l', action='store_true',
                        help='Список моделей')

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.input and not args.dir:
        parser.print_help()
        sys.exit(1)

    model = MODELS.get(args.model, args.model)

    common_args = {
        'model_name': model,
        'denoise': not args.no_denoise,
        'use_gpu': not args.cpu,
        'output_format': args.format,
        'sample_rate': args.sample_rate,
        'segment_size': args.segment_size,
        'chunk_duration': args.chunk_duration,
        'workers': args.workers,
    }

    if args.dir:
        process_directory(args.dir, output_dir=args.output, **common_args)
    else:
        clean_audiobook(args.input, output_dir=args.output, **common_args)


if __name__ == '__main__':
    main()
