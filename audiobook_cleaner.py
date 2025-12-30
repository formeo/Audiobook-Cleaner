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
import tempfile
import json
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    
    logger.info(f"Нарезка на части по {chunk_duration} секунд...")
    
    subprocess.run([
        'ffmpeg', '-y', '-i', str(input_file),
        '-f', 'segment', '-segment_time', str(chunk_duration),
        '-c:a', 'pcm_s16le',  # WAV для лучшего качества обработки
        str(pattern)
    ], capture_output=True, check=True)
    
    chunks = sorted(output_dir.glob("chunk_*.wav"))
    logger.info(f"Создано {len(chunks)} частей")
    return chunks


def concat_audio(input_files: list[Path], output_file: Path, output_format: str):
    """Склеивает аудиофайлы в один."""
    if not input_files:
        return
    
    list_file = input_files[0].parent / "concat_list.txt"
    
    # Создаём файл со списком для ffmpeg
    with open(list_file, 'w', encoding='utf-8') as f:
        for file in input_files:
            f.write(f"file '{file.absolute()}'\n")
    
    logger.info(f"Склейка {len(input_files)} файлов...")
    
    # Параметры кодирования в зависимости от формата
    codec_args = []
    if output_format == 'mp3':
        codec_args = ['-c:a', 'libmp3lame', '-q:a', '2']
    elif output_format == 'flac':
        codec_args = ['-c:a', 'flac']
    else:  # wav
        codec_args = ['-c:a', 'pcm_s16le']
    
    subprocess.run([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(list_file),
        *codec_args,
        str(output_file)
    ], capture_output=True, check=True)
    
    list_file.unlink()
    logger.info(f"Результат сохранён: {output_file}")


def _process_chunk_worker(args: tuple) -> tuple[int, list[Path]]:
    """Воркер для параллельной обработки чанка."""
    (
        chunk_idx, chunk_path, output_dir,
        model_name, denoise, use_gpu,
        sample_rate, segment_size
    ) = args
    
    try:
        results = _process_single_file(
            input_file=str(chunk_path),
            output_dir=str(output_dir),
            model_name=model_name,
            denoise=denoise,
            use_gpu=use_gpu,
            output_format='wav',
            sample_rate=sample_rate,
            segment_size=segment_size,
        )
        return (chunk_idx, results)
    except Exception as e:
        logger.error(f"Ошибка при обработке части {chunk_idx}: {e}")
        return (chunk_idx, [])


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
    
    Args:
        input_file: Путь к входному аудиофайлу
        output_dir: Директория для результатов
        model_name: Название модели для сепарации
        denoise: Применить дополнительное шумоподавление
        use_gpu: Использовать GPU (CUDA)
        output_format: Формат выходных файлов (mp3, wav, flac)
        sample_rate: Частота дискретизации выхода
        segment_size: Размер сегмента в секундах (уменьшить при нехватке памяти)
        chunk_duration: Нарезать на части по N секунд (0 = не нарезать)
        workers: Количество параллельных воркеров (только для CPU)
        
    Returns:
        Список путей к выходным файлам
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_file}")
    
    # Если указана нарезка на части — используем chunked processing
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
) -> list[Path]:
    """Обрабатывает большой файл по частям."""
    
    input_path = Path(input_file)
    
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_cleaned"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    duration = get_audio_duration(input_file)
    logger.info(f"Входной файл: {input_path}")
    logger.info(f"Длительность: {duration/60:.1f} минут")
    logger.info(f"Режим: нарезка по {chunk_duration} секунд")
    
    # Создаём временную директорию для работы
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        chunks_dir = temp_path / "chunks"
        processed_dir = temp_path / "processed"
        
        # 1. Нарезаем на части
        chunks = split_audio(input_file, chunks_dir, chunk_duration)
        
        if not chunks:
            logger.error("Не удалось нарезать файл")
            return []
        
        # 2. Обрабатываем каждую часть
        vocals_files = []
        instrumental_files = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Обработка части [{i}/{len(chunks)}]: {chunk.name}")
            logger.info('='*50)
            
            chunk_output = processed_dir / f"chunk_{i:04d}"
            
            try:
                results = _process_single_file(
                    input_file=str(chunk),
                    output_dir=str(chunk_output),
                    model_name=model_name,
                    denoise=denoise,
                    use_gpu=use_gpu,
                    output_format='wav',  # WAV для промежуточных файлов
                    sample_rate=sample_rate,
                    segment_size=segment_size,
                )
                
                # Сортируем результаты по типу
                for f in results:
                    if 'vocal' in f.stem.lower():
                        vocals_files.append(f)
                    elif 'instrument' in f.stem.lower():
                        instrumental_files.append(f)
                    elif 'no_noise' in f.stem.lower() or 'no noise' in f.stem.lower():
                        # Результат шумоподавления — это тоже вокал
                        vocals_files.append(f)
                        
            except Exception as e:
                logger.error(f"Ошибка при обработке части {i}: {e}")
                continue
        
        # 3. Склеиваем результаты
        output_files = []
        
        if vocals_files:
            # Берём только последние версии (после denoise если был)
            # Сортируем и группируем по номеру чанка
            final_vocals = []
            seen_chunks = set()
            for f in reversed(vocals_files):
                # Извлекаем номер части из пути
                chunk_num = f.parent.name
                if chunk_num not in seen_chunks:
                    seen_chunks.add(chunk_num)
                    final_vocals.append(f)
            final_vocals.reverse()
            
            vocals_output = output_path / f"{input_path.stem}_(Vocals).{output_format}"
            concat_audio(final_vocals, vocals_output, output_format)
            output_files.append(vocals_output)
        
        if instrumental_files:
            inst_output = output_path / f"{input_path.stem}_(Instrumental).{output_format}"
            concat_audio(instrumental_files, inst_output, output_format)
            output_files.append(inst_output)
    
    logger.info("\nГотово!")
    for f in output_files:
        logger.info(f"  - {f}")
    
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
    """
    Обрабатывает один файл.
    """
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
    
    # Инициализация сепаратора
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
    
    # Шаг 1: Отделение голоса от музыки
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
    
    # Шаг 2: Дополнительное шумоподавление
    if denoise and output_files:
        logger.info("Шаг 2: Дополнительное шумоподавление...")
        
        # Ищем файл с вокалом/голосом
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
        logger.info(f"\n{'='*50}")
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
  python audiobook_cleaner.py audiobook.mp3 --segment-size 32  # при нехватке памяти
  python audiobook_cleaner.py audiobook.mp3 --chunk-duration 300  # большие файлы
  python audiobook_cleaner.py --dir /path/to/audiobooks
  python audiobook_cleaner.py audiobook.mp3 --cpu
  python audiobook_cleaner.py --list-models
        """
    )
    
    parser.add_argument('input', nargs='?', help='Входной аудиофайл')
    parser.add_argument('--dir', '-d', help='Обработать директорию')
    parser.add_argument('--output', '-o', help='Выходная директория')
    parser.add_argument('--model', '-m', default='UVR-MDX-NET-Voc_FT.onnx',
                        help='Модель (default: UVR-MDX-NET-Voc_FT.onnx)')
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
    }
    
    if args.dir:
        process_directory(args.dir, output_dir=args.output, **common_args)
    else:
        clean_audiobook(args.input, output_dir=args.output, **common_args)


if __name__ == '__main__':
    main()
