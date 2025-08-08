from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional, Tuple

from .settings import AppSettings


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def transcode_to_wav(input_file: Path, output_wav: Path) -> None:
    """Конвертация аудио/видео в WAV 16kHz mono PCM с нормализацией громкости.

    Использует ffmpeg, аналогично `run.sh`.
    """
    ensure_dir(output_wav.parent)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_wav),
        "-y",
    ]
    subprocess.run(cmd, check=True)


def _which_whisperx() -> list[str]:
    # Ищем whisperx в PATH, иначе пробуем модульный запуск
    from shutil import which

    bin_path = which("whisperx")
    if bin_path:
        return [bin_path]
    return ["python3", "-m", "whisperx"]


def build_whisperx_cmd(
    wav_path: Path,
    settings: AppSettings,
    output_dir: Path,
    hf_token: Optional[str],
    base_output_name: str,
) -> list[str]:
    cmd = _which_whisperx()
    cmd += [
        "--model",
        settings.model,
        "--language",
        settings.language,
        "--device",
        settings.device,
        "--compute_type",
        settings.compute_type,
        "--beam_size",
        str(settings.beam_size),
        "--length_penalty",
        str(settings.length_penalty),
        "--temperature",
        str(settings.temperature),
        "--temperature_increment_on_fallback",
        str(settings.temperature_increment_on_fallback),
        "--vad_method",
        settings.vad_method,
        "--vad_onset",
        str(settings.vad_onset),
        "--vad_offset",
        str(settings.vad_offset),
        "--batch_size",
        str(settings.batch_size),
        "--output_dir",
        str(output_dir),
        "--output_format",
        settings.output_format,
        str(wav_path),
    ]

    if settings.diarize:
        cmd += [
            "--diarize",
            "--min_speakers",
            str(settings.min_speakers),
            "--max_speakers",
            str(settings.max_speakers),
        ]

    if hf_token:
        cmd += ["--hf_token", hf_token]

    return cmd


def transcribe_file(
    input_file: Path,
    work_uploads_dir: Path,
    work_out_dir: Path,
    settings: AppSettings,
    hf_token: Optional[str] = None,
) -> Tuple[Path, str]:
    """Транскрибирует один файл, возвращает путь к сгенерированному файлу транскрипта.

    - Сохраняет исходник во временную директорию uploads
    - Конвертирует в WAV 16kHz mono
    - Запускает whisperx CLI
    - Возвращает путь к файлу результата в `work_out_dir`.
    """
    ensure_dir(work_uploads_dir)
    ensure_dir(work_out_dir)

    unique_id = uuid.uuid4().hex
    src_ext = input_file.suffix.lower()
    raw_path = work_uploads_dir / f"{unique_id}{src_ext}"
    wav_path = work_uploads_dir / f"{unique_id}.wav"

    result_path: Optional[Path] = None
    try:
        shutil.copy2(input_file, raw_path)
        transcode_to_wav(raw_path, wav_path)

        # WhisperX выводит результат с именем исходного файла (без расширения)
        base_output_name = unique_id
        cmd = build_whisperx_cmd(
            wav_path, settings, work_out_dir, hf_token, base_output_name
        )
        subprocess.run(cmd, check=True)

        # Разыщем ожидаемый файл вывода
        expected = work_out_dir / f"{unique_id}.{settings.output_format}"
        if expected.exists():
            result_path = expected
        else:
            # На случай изменений в CLI попробуем найти любой файл с таким base
            candidates = list(work_out_dir.glob(f"{unique_id}.*"))
            if not candidates:
                raise FileNotFoundError(
                    f"Не найден результат транскрипции для {unique_id} в {work_out_dir}"
                )
            result_path = candidates[0]

        return result_path, unique_id
    finally:
        # Удаляем загруженный исходник и промежуточный WAV, хранить их не нужно
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass
