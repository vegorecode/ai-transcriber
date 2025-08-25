# Модели ввода/вывода и базовый предиктор (локальный stub cog)
import gc
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path as PathlibPath
from typing import Any

import numpy as np

# Отключаем автоподтягивание torchvision внутри transformers до импорта whisperx
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
import whisperx

from cog_stub import BaseModel, BasePredictor, Input, Path

logger = logging.getLogger("predict")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
compute_type = "float16"
device = "cuda"
whisper_arch = "./models/faster-whisper-large-v3-russian"

# Базовые дефолты порогов ASR (как в старом проекте/run.sh)
DEFAULT_NO_SPEECH_THRESHOLD = 0.60
DEFAULT_LOGPROB_THRESHOLD = -1.0
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4


class Output(BaseModel):
    segments: Any
    detected_language: str
    debug_info: Any | None = None


class Predictor(BasePredictor):
    def setup(self):
        # VAD теперь поставляется вместе с whisperx; дополнительных копий не требуется
        return

    def predict(
        self,
        audio_file: Path = Input(description="Аудиофайл"),
        model: str = Input(
            description="Модель или локальный путь (например, 'faster-whisper-large-v3-russian')",
            default="faster-whisper-large-v3-russian",
        ),
        language: str = Input(
            description="ISO‑код языка аудио; укажите None, чтобы определить язык автоматически",
            default=None,
        ),
        compute_type: str = Input(
            description="Тип вычислений WhisperX (например, float16, int8)",
            default="float16",
        ),
        language_detection_min_prob: float = Input(
            description="Если язык не задан, будет выполняться итеративное определение на разных частях файла, пока не будет достигнут порог вероятности",
            default=0,
        ),
        language_detection_max_tries: int = Input(
            description="Максимум попыток для итеративного детекта языка; если достигнут лимит, берётся наиболее вероятный",
            default=5,
        ),
        # initial_prompt удалён из публичного API (упрощение интерфейса)
        batch_size: int = Input(
            description="Параллелизм при транскрипции входного аудио", default=32
        ),
        beam_size: int = Input(description="Размер бима для декодирования", default=5),
        temperature: float = Input(description="Температура сэмплирования", default=0),
        temperature_increment_on_fallback: float = Input(
            description="Приращение температуры при fallback‑декодировании",
            default=None,
        ),
        vad_onset: float = Input(
            description="Порог срабатывания VAD (onset)", default=0.300
        ),
        vad_offset: float = Input(
            description="Порог окончания VAD (offset)", default=0.250
        ),
        min_duration_on: float = Input(
            description="Мин. длительность речи (сек) для VAD", default=0.08
        ),
        min_duration_off: float = Input(
            description="Мин. длительность тишины (сек) для VAD", default=0.08
        ),
        pad_onset: float = Input(
            description="Продлить начало каждого VAD-сегмента на указанное кол-во секунд",
            default=0.0,
        ),
        pad_offset: float = Input(
            description="Продлить конец каждого VAD-сегмента на указанное кол-во секунд",
            default=0.0,
        ),
        align_output: bool = Input(
            description="Выровнять вывод для точных тайм‑кодов слов",
            default=True,
        ),
        diarization: bool = Input(
            description="Включить диаризацию (метки спикеров)", default=True
        ),
        diarize: bool = Input(description="Алиас поля 'diarization'", default=None),
        huggingface_access_token: str = Input(
            description="Для диаризации укажите токен Hugging Face (read) и примите условия моделей, указанных в README.",
            default=None,
        ),
        min_speakers: int = Input(
            description="Минимальное число спикеров (если включена диаризация)",
            default=None,
        ),
        max_speakers: int = Input(
            description="Максимальное число спикеров (если включена диаризация)",
            default=None,
        ),
        length_penalty: float = Input(
            description="Штраф за длину при декодировании (beam search)", default=1.1
        ),
        no_speech_threshold: float = Input(
            description="Порог no_speech для ASR (снижайте, чтобы меньше отбрасывать тихую речь)",
            default=None,
        ),
        log_prob_threshold: float = Input(
            description="Порог log_prob для ASR (снижайте, чтобы меньше отбрасывать сомнительные сегменты)",
            default=None,
        ),
        compression_ratio_threshold: float = Input(
            description="Порог compression_ratio для ASR (понизьте, чтобы меньше фильтровать)",
            default=None,
        ),
        output_format: str = Input(
            description="Желаемый формат ответа (информативно)", default="json"
        ),
        debug: bool = Input(
            description="Печатать времена вычислений/инференса и потребление памяти",
            default=False,
        ),
    ) -> Output:
        with torch.inference_mode():
            tmp_files_to_cleanup = []
            try:
                # Только GPU: первая CUDA‑карта
                runtime_device = "cuda"
                runtime_device_index = 0
                # Сохраняем устройство в глобальной переменной для align/diarize
                globals()["device"] = runtime_device
                # Выбираем модель: локальная папка в /models или идентификатор
                model_path = model
                if model and not model.startswith("/") and "/" not in model:
                    local_candidate = f"/models/{model}"
                    model_path = (
                        local_candidate if os.path.exists(local_candidate) else model
                    )
                # Синхронизируем глобальные параметры для вспомогательных функций
                globals()["whisper_arch"] = model_path
                globals()["compute_type"] = compute_type
                asr_options = {
                    "temperatures": [temperature],
                    "no_speech_threshold": (
                        no_speech_threshold
                        if no_speech_threshold is not None
                        else DEFAULT_NO_SPEECH_THRESHOLD
                    ),
                    "log_prob_threshold": (
                        log_prob_threshold
                        if log_prob_threshold is not None
                        else DEFAULT_LOGPROB_THRESHOLD
                    ),
                    "compression_ratio_threshold": (
                        compression_ratio_threshold
                        if compression_ratio_threshold is not None
                        else DEFAULT_COMPRESSION_RATIO_THRESHOLD
                    ),
                }
                if beam_size is not None:
                    asr_options["beam_size"] = beam_size
                if length_penalty is not None:
                    asr_options["length_penalty"] = length_penalty
                if temperature_increment_on_fallback is not None:
                    asr_options["temperature_increment_on_fallback"] = (
                        temperature_increment_on_fallback
                    )

                vad_options = {
                    "vad_method": "pyannote",
                    "vad_onset": vad_onset,
                    "vad_offset": vad_offset,
                    "min_duration_on": min_duration_on,
                    "min_duration_off": min_duration_off,
                    "pad_onset": pad_onset,
                    "pad_offset": pad_offset,
                }

                # Язык определяем позже на уже загруженном аудио (без временных файлов)

                start_time = time.time_ns() / 1e6

                model = whisperx.load_model(
                    model_path,
                    device=runtime_device,
                    device_index=runtime_device_index,
                    compute_type=compute_type,
                    language=language,
                    asr_options=asr_options,
                    vad_options=vad_options,
                )

                if debug:
                    elapsed_time = time.time_ns() / 1e6 - start_time
                    logger.debug("Время загрузки модели: %.2f мс", elapsed_time)

                start_time = time.time_ns() / 1e6

                # Загружаем аудио: если уже нормализованный WAV 16k/mono — читаем напрямую,
                # иначе используем стандартный путь whisperx (ffmpeg внутри)
                audio, audio_report = load_audio_optimized(audio_file)
                # Детект языка по 30‑сек. окнам на уже загруженном массиве
                audio_duration = int(len(audio) / 16000.0 * 1000.0)
                if (
                    language is None
                    and language_detection_min_prob > 0
                    and audio_duration > 30000
                ):
                    segments_duration_ms = 30000
                    language_detection_max_tries = min(
                        language_detection_max_tries,
                        math.floor(audio_duration / segments_duration_ms),
                    )
                    segments_starts = distribute_segments_equally(
                        audio_duration,
                        segments_duration_ms,
                        language_detection_max_tries,
                    )
                    logger.info(
                        "Detecting languages on segments starting at %s",
                        ", ".join(map(str, segments_starts)),
                    )
                    detected_language_details = detect_language(
                        audio,
                        segments_starts,
                        language_detection_min_prob,
                        language_detection_max_tries,
                        asr_options,
                        vad_options,
                    )
                    detected_language_code = detected_language_details["language"]
                    detected_language_prob = detected_language_details["probability"]
                    detected_language_iterations = detected_language_details[
                        "iterations"
                    ]
                    logger.info(
                        "Detected language %s (%.2f) after %s iterations.",
                        detected_language_code,
                        detected_language_prob,
                        detected_language_iterations,
                    )
                    language = detected_language_details["language"]

                if debug:
                    elapsed_time = time.time_ns() / 1e6 - start_time
                    logger.debug("Время загрузки аудио: %.2f мс", elapsed_time)

                start_time = time.time_ns() / 1e6

                result = model.transcribe(audio, batch_size=batch_size)
                detected_language = result["language"]

                if debug:
                    elapsed_time = time.time_ns() / 1e6 - start_time
                    logger.debug("Время транскрипции: %.2f мс", elapsed_time)

                gc.collect()
                torch.cuda.empty_cache()
                del model

                if align_output or diarization:
                    try:
                        result = align(audio, result, debug)
                    except Exception as align_exc:
                        logger.warning(
                            "Выравнивание недоступно, пропускаем (lang=%s): %s",
                            detected_language,
                            str(align_exc),
                        )

                # alias поддержка
                if diarize is not None:
                    diarization = diarize

                if diarization:
                    if huggingface_access_token:
                        result = run_diarization(
                            audio_file,
                            result,
                            debug,
                            huggingface_access_token,
                            min_speakers,
                            max_speakers,
                            runtime_device,
                        )
                    else:
                        logger.info(
                            "Запрошена диаризация, но токен Hugging Face не указан — пропускаем"
                        )

                if debug:
                    logger.info(
                        "макс. резерв памяти GPU за время работы: %.2f ГБ",
                        torch.cuda.max_memory_reserved() / (1024**3),
                    )

                output_segments = result["segments"]
            finally:
                # На всякий случай: временных файлов мы не создаём, но если есть — удалим
                try:
                    for p in tmp_files_to_cleanup:
                        try:
                            PathlibPath(p).unlink(missing_ok=True)  # type: ignore[arg-type]
                        except Exception:
                            pass
                except Exception:
                    pass

            debug_info = None
            if debug:
                try:
                    debug_info = build_debug_info(
                        audio_report=audio_report,
                        runtime_device=runtime_device,
                        model_path=model_path,
                        compute_type=compute_type,
                        asr_options=asr_options,
                        vad_options=vad_options,
                        batch_size=batch_size,
                        beam_size=beam_size,
                        temperature=temperature,
                        temperature_increment_on_fallback=temperature_increment_on_fallback,
                        align_output=align_output,
                        diarization=diarization if diarize is None else diarize,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        length_penalty=length_penalty,
                    )
                except Exception:
                    debug_info = None

            return Output(
                segments=output_segments,
                detected_language=detected_language,
                debug_info=debug_info,
            )


def get_audio_duration(file_path):
    """Длительность аудио в миллисекундах.

    Предпочтительно используем ffprobe (без полной декодировки файла).
    Если ffprobe недоступен или не смог определить длительность —
    выполняем безопасный фоллбек через whisperx.load_audio (дороже по памяти).
    """
    try:
        # ffprobe вернёт длительность в секундах одной строкой
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(file_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        seconds_str = (result.stdout or "").strip()
        if seconds_str:
            seconds = float(seconds_str)
            return int(seconds * 1000.0)
    except Exception:
        # Переходим к фоллбеку ниже
        pass

    # Фоллбек: полная загрузка (как раньше), чтобы не падать без ffmpeg
    try:
        audio = whisperx.load_audio(file_path)
        # whisper/whisperx обычно возвращают сигнал с частотой 16000 Гц
        duration_ms = int(len(audio) / 16000.0 * 1000.0)
        return duration_ms
    except Exception:
        return 0


def detect_language(
    full_audio,
    segments_starts,
    language_detection_min_prob,
    language_detection_max_tries,
    asr_options,
    vad_options,
):
    """Определение языка по нескольким 30‑сек. сегментам без обращения к приватным API.

    Для каждого сегмента выполняется быстрый transcribe (batch_size=1). Язык выбирается
    по большинству голосов; probability — доля голосов за победивший язык.
    """

    model = whisperx.load_model(
        whisper_arch,
        device="cuda",
        device_index=0,
        compute_type=compute_type,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    votes = {}
    total = 0
    winner = None
    winner_ratio = 0.0
    iterations_done = 0

    max_iters = min(language_detection_max_tries, len(segments_starts))

    for idx in range(max_iters):
        start_ms = segments_starts[idx]
        if isinstance(full_audio, np.ndarray):
            # Режем уже загруженный массив, избегая временных файлов
            audio_slice = slice_ms(full_audio, start_ms, 30000, sr=16000)
            result = model.transcribe(audio_slice, batch_size=1)
            lang = result.get("language")
        else:
            # Фоллбек по пути к файлу: используем ffmpeg для точного seek
            audio_segment_file_path = extract_audio_segment(full_audio, start_ms, 30000)
            try:
                audio = whisperx.load_audio(audio_segment_file_path)
                result = model.transcribe(audio, batch_size=1)
                lang = result.get("language")
            finally:
                try:
                    audio_segment_file_path.unlink()
                except Exception:
                    pass

        total += 1
        if lang:
            votes[lang] = votes.get(lang, 0) + 1
            ratio = votes[lang] / total
            if ratio >= winner_ratio:
                winner = lang
                winner_ratio = ratio

            logger.info("Итерация %s — язык: %s (freq=%.2f)", idx + 1, lang, ratio)

            iterations_done = idx + 1

            if language_detection_min_prob > 0 and ratio >= language_detection_min_prob:
                break

    if not winner and votes:
        winner = max(votes, key=votes.get)
        winner_ratio = votes[winner] / max(1, total)

    gc.collect()
    torch.cuda.empty_cache()
    del model

    return {
        "language": winner or "unknown",
        "probability": float(winner_ratio),
        "iterations": iterations_done or total,
    }


def extract_audio_segment(input_file_path, start_time_ms, duration_ms):
    """Нарезка сегмента через ffmpeg без полной декодировки в память.

    Возвращает путь к времённому WAV файлу (16 кГц, моно), совместимому с
    whisper/whisperx.load_audio.
    """
    input_file_path = (
        Path(input_file_path)
        if not isinstance(input_file_path, Path)
        else input_file_path
    )

    start_seconds = max(0.0, float(start_time_ms) / 1000.0)
    duration_seconds = max(0.0, float(duration_ms) / 1000.0)

    # Создаём целевой временный WAV, чтобы декодирование было предсказуемым
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_path = Path(temp_file.name)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(input_file_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        "-y",
        str(temp_file_path),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return temp_file_path
    except Exception:
        # При неудаче — удалим файл и пробросим исключение дальше
        try:
            temp_file_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        raise


def load_audio_optimized(file_path):
    """Загрузка аудио с отчётом о соответствии формату.

    Если файл уже в целевом формате (16 kHz mono), читаем напрямую через torchaudio
    и возвращаем (numpy_array, format_report). Иначе используем стандартный
    whisperx.load_audio (ffmpeg внутри) и возвращаем (numpy_array, format_report)
    с признаком used_fallback=true и причиной несовпадения.
    """
    try:
        import torchaudio  # noqa: WPS433

        info = torchaudio.info(str(file_path))
        actual_sr = int(getattr(info, "sample_rate", 0))
        actual_ch = int(getattr(info, "num_channels", 0))
        expected_sr = 16000
        expected_ch = 1

        is_ok = actual_sr == expected_sr and actual_ch == expected_ch
        if is_ok:
            waveform, sr = torchaudio.load(str(file_path))
            if waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            audio_np = waveform.squeeze(0).numpy()
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            report = {
                "source": "direct",
                "is_compatible": True,
                "expected": {"sample_rate": expected_sr, "num_channels": expected_ch},
                "actual": {"sample_rate": actual_sr, "num_channels": actual_ch},
                "used_fallback": False,
            }
            return audio_np, report
        # несовпадение — пояснение
        reason = []
        if actual_sr != expected_sr:
            reason.append(f"sample_rate {actual_sr}!={expected_sr}")
        if actual_ch != expected_ch:
            reason.append(f"num_channels {actual_ch}!={expected_ch}")
        mismatch_reason = ", ".join(reason) if reason else "unknown"
    except Exception:
        # тихо падаем в стандартный путь
        mismatch_reason = "torchaudio_info_failed"
        actual_sr = None
        actual_ch = None

    # Фоллбек: whisperx + ffmpeg
    audio_np = whisperx.load_audio(file_path)
    report = {
        "source": "ffmpeg_fallback",
        "is_compatible": False,
        "expected": {"sample_rate": 16000, "num_channels": 1},
        "actual": {"sample_rate": actual_sr, "num_channels": actual_ch},
        "used_fallback": True,
        "reason": mismatch_reason,
    }
    return audio_np, report


def slice_ms(
    audio: np.ndarray, start_ms: int, dur_ms: int, sr: int = 16000
) -> np.ndarray:
    """Вернуть срез из массива аудио по миллисекундам.

    Срез без копирования для непрерывных массивов (NumPy выдаст view).
    """
    start_sample = int(max(0, start_ms) * sr / 1000)
    end_sample = start_sample + int(max(0, dur_ms) * sr / 1000)
    end_sample = min(end_sample, audio.shape[-1])
    return audio[start_sample:end_sample]


def build_debug_info(
    *,
    audio_report,
    runtime_device,
    model_path,
    compute_type,
    asr_options,
    vad_options,
    batch_size,
    beam_size,
    temperature,
    temperature_increment_on_fallback,
    align_output,
    diarization,
    min_speakers,
    max_speakers,
    length_penalty,
):
    """Собирает слепок конфигурации инференса для debug.

    Возвращает словарь, безопасный для сериализации в JSON.
    """
    return {
        "audio": audio_report,
        "runtime": {
            "device": runtime_device,
            "torch_tf32": {
                "cudnn": bool(torch.backends.cudnn.allow_tf32),
                "cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            },
        },
        "model": {
            "path": model_path,
            "compute_type": compute_type,
            "asr_options": asr_options,
            "vad_options": vad_options,
        },
        "decode": {
            "batch_size": batch_size,
            "beam_size": beam_size,
            "temperature": temperature,
            "temperature_increment_on_fallback": temperature_increment_on_fallback,
            "length_penalty": length_penalty,
        },
        "output": {
            "align_output": align_output,
            "diarization": diarization,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
    }


def distribute_segments_equally(total_duration, segments_duration, iterations):
    available_duration = total_duration - segments_duration

    if iterations > 1:
        spacing = available_duration // (iterations - 1)
    else:
        spacing = 0

    start_times = [i * spacing for i in range(iterations)]

    if iterations > 1:
        start_times[-1] = total_duration - segments_duration

    return start_times


def align(audio, result, debug):
    start_time = time.time_ns() / 1e6

    # Совместимость с разными версиями whisperx: API мог меняться
    # Пробуем рекомендованный путь, иначе пропускаем выравнивание
    model_a = None
    try:
        load_align_model = getattr(whisperx, "load_align_model", None)
        align_fn = getattr(whisperx, "align", None)
        if callable(load_align_model) and callable(align_fn):
            model_a, metadata = load_align_model(
                language_code=result["language"], device="cuda"
            )
            result = align_fn(
                result["segments"],
                model_a,
                metadata,
                audio,
                "cuda",
                return_char_alignments=False,
            )
        else:
            raise RuntimeError("align API is not available in this whisperx build")
    finally:
        if model_a is not None:
            try:
                del model_a
            except Exception:
                pass

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        logger.debug("Duration to align output: %.2f ms", elapsed_time)

    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_diarization(
    audio_file,
    result,
    debug,
    huggingface_access_token,
    min_speakers,
    max_speakers,
    runtime_device,
):
    start_time = time.time_ns() / 1e6

    # Обёртка WhisperX над pyannote — рекомендуемый путь
    diarize_model = whisperx.diarize.DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=huggingface_access_token,
        device="cuda",
    )

    # Передаём путь к файлу (строкой) — пайплайн сам загрузит аудио
    diarize_segments = diarize_model(
        str(audio_file),
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        logger.debug("Duration to diarize segments: %.2f ms", elapsed_time)

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result
