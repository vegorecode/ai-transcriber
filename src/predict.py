# Модели ввода/вывода и базовый предиктор (локальный stub cog)
import gc
import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path as PathlibPath
from typing import Any

# Отключаем автоподтягивание torchvision внутри transformers до импорта whisperx
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
import whisperx
from pydub import AudioSegment

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
            description="Параллелизм при транскрипции входного аудио", default=64
        ),
        beam_size: int = Input(
            description="Размер бима для декодирования", default=None
        ),
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
            default=False,
        ),
        diarization: bool = Input(
            description="Включить диаризацию (метки спикеров)", default=False
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
            description="Штраф за длину при декодировании (beam search)", default=None
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

                audio_duration = get_audio_duration(audio_file)

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
                        audio_file,
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

                # Загружаем аудио как есть: нормализация выполняется вне пайплайна
                audio = whisperx.load_audio(audio_file)

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

            return Output(segments=output_segments, detected_language=detected_language)


def get_audio_duration(file_path):
    """Длительность аудио в миллисекундах."""
    return len(AudioSegment.from_file(file_path))


def detect_language(
    full_audio_file_path,
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
        audio_segment_file_path = extract_audio_segment(
            full_audio_file_path, start_ms, 30000
        )
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
    input_file_path = (
        Path(input_file_path)
        if not isinstance(input_file_path, Path)
        else input_file_path
    )

    audio = AudioSegment.from_file(input_file_path)

    end_time_ms = start_time_ms + duration_ms
    extracted_segment = audio[start_time_ms:end_time_ms]

    file_extension = input_file_path.suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file_path = Path(temp_file.name)
        extracted_segment.export(temp_file_path, format=file_extension.lstrip("."))

    return temp_file_path


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
