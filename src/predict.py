# Модели ввода/вывода и базовый предиктор (локальный stub cog)
import gc
import logging
import math
import os
import sys
import tempfile
import time
from typing import Any

import torch
import whisperx
from pydub import AudioSegment
from scipy.spatial.distance import cosine
from whisperx.audio import N_SAMPLES, log_mel_spectrogram

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
compute_type = (
    "float16"  # можно сменить на "int8" при дефиците GPU‑памяти (точность снизится)
)
device = "cuda"
whisper_arch = "./models/faster-whisper-large-v3-russian"

# Жёстко заданные пороги (можно переопределить через ENV); поддерживаем имена как в run.sh
NO_SPEECH_THRESHOLD = float(
    os.environ.get("NO_SPEECH_THRESHOLD")
    or os.environ.get("ASR_NO_SPEECH_THRESHOLD", "0.6")
)
LOGPROB_THRESHOLD = float(
    os.environ.get("LOGPROB_THRESHOLD")
    or os.environ.get("ASR_LOGPROB_THRESHOLD", "-1.0")
)
COMPRESSION_RATIO_THRESHOLD = float(
    os.environ.get("COMPRESSION_RATIO_THRESHOLD")
    or os.environ.get("ASR_COMPRESSION_RATIO_THRESHOLD", "2.4")
)


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
        device: str = Input(description="Устройство: 'cuda' или 'cpu'", default="cuda"),
        device_index: int = Input(
            description="Индекс CUDA‑GPU (опционально)", default=None
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
        initial_prompt: str = Input(
            description="Необязательный подсказочный текст для первого окна транскрипции",
            default=None,
        ),
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
            description="Порог срабатывания VAD (onset)", default=0.500
        ),
        vad_offset: float = Input(
            description="Порог окончания VAD (offset)", default=0.363
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
        output_format: str = Input(
            description="Желаемый формат ответа (информативно)", default="json"
        ),
        debug: bool = Input(
            description="Печатать времена вычислений/инференса и потребление памяти",
            default=False,
        ),
        speaker_verification: bool = Input(
            description="Включить верификацию спикеров", default=False
        ),
        speaker_samples: list = Input(
            description="Список образцов голоса для верификации. Каждый элемент: dict с 'url' и опциональными 'name'/'file_path'. Если name не указан — берётся имя файла.",
            default=[],
        ),
    ) -> Output:
        with torch.inference_mode():
            # Вычисляем устройство с учётом индекса
            runtime_device = (
                device
                if device != "cuda" or device_index is None
                else f"cuda:{device_index}"
            )
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
                "initial_prompt": initial_prompt,
                # Жёстко заданные пороги, не приходят извне
                "no_speech_threshold": NO_SPEECH_THRESHOLD,
                "log_prob_threshold": LOGPROB_THRESHOLD,
                "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
            }
            if beam_size is not None:
                asr_options["beam_size"] = beam_size
            if length_penalty is not None:
                asr_options["length_penalty"] = length_penalty
            if temperature_increment_on_fallback is not None:
                asr_options["temperature_increment_on_fallback"] = (
                    temperature_increment_on_fallback
                )

            vad_options = {"vad_onset": vad_onset, "vad_offset": vad_offset}

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
                    audio_duration, segments_duration_ms, language_detection_max_tries
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
                detected_language_iterations = detected_language_details["iterations"]

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
                runtime_device,
                compute_type=compute_type,
                language=language,
                asr_options=asr_options,
                vad_options=vad_options,
            )

            if debug:
                elapsed_time = time.time_ns() / 1e6 - start_time
                logger.debug("Время загрузки модели: %.2f мс", elapsed_time)

            start_time = time.time_ns() / 1e6

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

            if align_output:
                if (
                    detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH
                    or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF
                ):
                    result = align(audio, result, debug)
                else:
                    logger.warning(
                        "Невозможно выровнять: язык %s не поддерживается в align",
                        detected_language,
                    )

            # alias поддержка
            if diarize is not None:
                diarization = diarize

            if diarization:
                if huggingface_access_token:
                    result = run_diarization(
                        audio,
                        result,
                        debug,
                        huggingface_access_token,
                        min_speakers,
                        max_speakers,
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

        return Output(segments=result["segments"], detected_language=detected_language)


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
    iteration=1,
):
    model = whisperx.load_model(
        whisper_arch,
        device,
        compute_type=compute_type,
        asr_options=asr_options,
        vad_options=vad_options,
    )

    start_ms = segments_starts[iteration - 1]

    audio_segment_file_path = extract_audio_segment(
        full_audio_file_path, start_ms, 30000
    )

    audio = whisperx.load_audio(audio_segment_file_path)

    model_n_mels = model.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(
        audio[:N_SAMPLES],
        n_mels=model_n_mels if model_n_mels is not None else 80,
        padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0],
    )
    encoder_output = model.model.encode(segment)
    results = model.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]

    logger.info(
        "Итерация %s — язык: %s (%.2f)", iteration, language, language_probability
    )

    audio_segment_file_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()
    del model

    detected_language = {
        "language": language,
        "probability": language_probability,
        "iterations": iteration,
    }

    if (
        language_probability >= language_detection_min_prob
        or iteration >= language_detection_max_tries
    ):
        return detected_language

    next_iteration_detected_language = detect_language(
        full_audio_file_path,
        segments_starts,
        language_detection_min_prob,
        language_detection_max_tries,
        asr_options,
        vad_options,
        iteration + 1,
    )

    if (
        next_iteration_detected_language["probability"]
        > detected_language["probability"]
    ):
        return next_iteration_detected_language

    return detected_language


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

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        logger.debug("Duration to align output: %.2f ms", elapsed_time)

    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    return result


def run_diarization(
    audio,
    result,
    debug,
    huggingface_access_token,
    min_speakers,
    max_speakers,
):
    start_time = time.time_ns() / 1e6

    diarize_model = whisperx.DiarizationPipeline(
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=huggingface_access_token,
        device=device,
    )
    diarize_segments = diarize_model(
        audio, min_speakers=min_speakers, max_speakers=max_speakers
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    if debug:
        elapsed_time = time.time_ns() / 1e6 - start_time
        logger.debug("Duration to diarize segments: %.2f ms", elapsed_time)

    gc.collect()
    torch.cuda.empty_cache()
    del diarize_model

    return result


def identify_speaker_for_segment(segment_embedding, known_embeddings, threshold=0.1):
    """
    Compare segment_embedding to known speaker embeddings using cosine similarity.
    Returns the speaker name with the highest similarity above the threshold,
    or "Unknown" if none match.
    """
    best_match = "Unknown"
    best_similarity = -1
    for speaker, known_emb in known_embeddings.items():
        similarity = 1 - cosine(segment_embedding, known_emb)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = speaker
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return "Unknown", best_similarity
