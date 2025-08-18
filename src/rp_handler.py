# Точка входа сервера: загрузка переменных окружения и настройка логирования
import os

from dotenv import find_dotenv, load_dotenv

# Загружаем переменные окружения из .env (если есть)
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

import logging

import torch
from huggingface_hub import login, whoami

from speaker_processing import (
    identify_speakers_on_segments,
    load_known_speakers_from_samples,
    relabel_speakers_by_avg_similarity,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Логирование
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Берём токен Hugging Face из окружения (если есть)
hf_token = os.environ.get("HF_TOKEN", "").strip()

if hf_token:
    try:
        logger.debug(f"HF_TOKEN загружен: {repr(hf_token[:10])}…")
        # Аутентификация Hugging Face (без записи в git‑credential)
        login(token=hf_token, add_to_git_credential=False)
        user = whoami(token=hf_token)
        logger.info(f"Аутентифицировано в Hugging Face как: {user['name']}")
    except Exception:
        logger.error("Не удалось аутентифицироваться в Hugging Face", exc_info=True)
else:
    logger.warning(
        "Переменная окружения HF_TOKEN не задана. Диаризация может быть недоступна."
    )

import logging
import os
import shutil
import sys

import runpod
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from predict import Predictor
from rp_schema import INPUT_VALIDATIONS

# Отдельный логгер для обработчика
logger = logging.getLogger("rp_handler")
logger.setLevel(logging.DEBUG)  # capture everything at DEBUG or above

# Консольный хендлер
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)

# Файловый хендлер (подробный лог в контейнер)
file_handler = logging.FileHandler("container_log.txt", mode="a")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

# Регистрируем хендлеры
logger.addHandler(console_handler)
logger.addHandler(file_handler)


MODEL = Predictor()
MODEL.setup()


def cleanup_job_files(job_id, jobs_directory="/jobs"):
    job_path = os.path.join(jobs_directory, job_id)
    if os.path.exists(job_path):
        try:
            shutil.rmtree(job_path)
            logger.info(f"Удалена временная директория задания: {job_path}")
        except Exception as e:
            logger.error(f"Ошибка удаления {job_path}: {str(e)}", exc_info=True)
    else:
        logger.debug(f"Временная директория не найдена: {job_path}")


# --------------------------------------------------------------------
# Основная точка входа (обработчик задания)
# --------------------------------------------------------------------
error_log = []


def run(job):
    job_id = job["id"]
    job_input = job["input"]

    # Сбор предупреждений для ответа
    warnings = []

    # ------------- проверяем вход по схеме --------------------------
    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    # ------------- 1) скачиваем основное аудио ----------------------
    try:
        audio_file_path = download_files_from_urls(job_id, [job_input["audio_file"]])[0]
        logger.debug(f"Audio downloaded → {audio_file_path}")
    except Exception as e:
        logger.error("Скачивание аудио не удалось", exc_info=True)
        return {"error": f"download audio: {e}"}

    # ------------- 2) загрузка примеров голосов (опционально) -------
    speaker_profiles = job_input.get("speaker_samples", [])
    embeddings = {}
    if speaker_profiles:
        try:
            # выбрать токен из job_input, затем из окружения
            token = job_input.get("huggingface_access_token") or hf_token
            embeddings = load_known_speakers_from_samples(
                speaker_profiles,
                huggingface_access_token=token,
            )
            logger.info(f"Загружено эталонных голосов: {len(embeddings)}")
        except Exception as e:
            logger.error("Не удалось загрузить профили спикеров", exc_info=True)
            warnings.append(f"Профили спикеров пропущены: {e}")
        # urls = [s.get("url") for s in speaker_profiles if s.get("url")]
        # if urls:
        #     try:
        #         local_paths = download_files_from_urls(job_id, urls)
        #         for s, path in zip(speaker_profiles, local_paths):
        #             s["file_path"] = path  # mutate in-place
        #             logger.debug(f"Profile {s.get('name')} → {path}")

        #         # Now enroll profiles using the updated speaker_profiles with local file paths
        #         embeddings = enroll_profiles(speaker_profiles)
        #         logger.info(f"Enrolled {len(embeddings)} speaker profiles successfully.")
        #     except Exception as e:
        #         logger.error("Enrollment failed", exc_info=True)
        #         output_dict["warning"] = f"Enrollment skipped: {e}"
    # ----------------------------------------------------------------

    # ------------- 3) запуск WhisperX / VAD / диаризации ------------
    predict_input = {
        "audio_file": audio_file_path,
        "model": job_input.get("model", "faster-whisper-large-v3-russian"),
        "language": job_input.get("language"),
        "device": job_input.get("device", "cuda"),
        "device_index": job_input.get("device_index"),
        "compute_type": job_input.get("compute_type", "float16"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get(
            "language_detection_max_tries", 5
        ),
        "initial_prompt": job_input.get("initial_prompt"),
        "batch_size": job_input.get("batch_size", 64),
        "beam_size": job_input.get("beam_size"),
        "temperature": job_input.get("temperature", 0),
        "temperature_increment_on_fallback": job_input.get(
            "temperature_increment_on_fallback"
        ),
        "vad_onset": job_input.get("vad_onset", 0.50),
        "vad_offset": job_input.get("vad_offset", 0.363),
        "align_output": job_input.get("align_output", False),
        "diarization": job_input.get("diarization", False),
        "diarize": job_input.get("diarize"),
        "huggingface_access_token": job_input.get("huggingface_access_token"),
        "min_speakers": job_input.get("min_speakers"),
        "max_speakers": job_input.get("max_speakers"),
        "length_penalty": job_input.get("length_penalty"),
        "output_format": job_input.get("output_format", "json"),
        "debug": job_input.get("debug", False),
    }

    try:
        result = MODEL.predict(**predict_input)
    except Exception as e:
        logger.error("Ошибка распознавания WhisperX", exc_info=True)
        return {"error": f"predict: {e}"}

    output_dict = {
        "segments": result.segments,
        "detected_language": result.detected_language,
    }
    # ------------------------------------------------embedding-info----------------
    # 4) идентификация спикеров (опционально)
    if embeddings:
        try:
            segments_with_speakers = identify_speakers_on_segments(
                segments=output_dict["segments"],
                audio_path=audio_file_path,
                enrolled=embeddings,
                threshold=0.1,  # Adjust threshold as needed
            )
            segments_with_final_labels = relabel_speakers_by_avg_similarity(
                segments_with_speakers
            )
            output_dict["segments"] = segments_with_final_labels
            logger.info("Идентификация спикеров завершена.")
        except Exception as e:
            logger.error("Сбой идентификации спикеров", exc_info=True)
            warnings.append(f"Идентификация пропущена: {e}")
    else:
        logger.info("Нет загруженных профилей — шаг идентификации пропущен.")

    # Добавляем предупреждения к ответу, если есть
    if warnings:
        output_dict["warnings"] = warnings

    # 5) приложим логи контейнера к ответу (хвост файла)
    try:
        log_path = os.path.abspath("container_log.txt")
        logs = _tail_file(log_path, max_bytes=200_000)
        output_dict["logs"] = logs
    except Exception as e:
        logger.warning(f"Не удалось прочитать логи: {e}")

    # 5) очистка и возврат результата
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    return output_dict


if os.getenv("RUNPOD_DISABLE_SERVERLESS") != "1":
    runpod.serverless.start({"handler": run})


def _tail_file(path: str, max_bytes: int = 200_000) -> str:
    """Прочитать хвост файла (последние max_bytes) в виде строки."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            data = f.read()
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return data.decode("latin-1", errors="replace")
    except Exception:
        return ""


#     embeddings = {} # ensure the name is always bound
#     if job_input.get("speaker_verification", True):
#         logger.info(f"Speaker-verification requested: True")
#         try:
#             embeddings = load_known_speakers_from_samples(
#                 speaker_profiles,
#                 huggingface_access_token=predict_input["huggingface_access_token"]
#             )
#             logger.info(f"  • Enrolled {len(embeddings)} profiles")
#         except Exception as e:
#             logger.error("Failed loading speaker profiles", exc_info=True)
#             output_dict["warning"] = f"enrollment skipped: {e}"

#         embedding_log_data = None  # Initialize here to avoid UnboundLocalError

#         if embeddings:  # only attempt verification if we actually got something
#             try:
#                 output_dict, embedding_log_data = process_diarized_output(
#                     output_dict,
#                     audio_file_path,
#                     embeddings,
#                     huggingface_access_token=job_input.get("huggingface_access_token"),
#                     return_logs=False # <-- set to True for debugging
#             except Exception as e:
#                 logger.error("Error during speaker verification", exc_info=True)
#                 output_dict["warning"] = f"verification skipped: {e}"
#         else:
#             logger.info("No embeddings to verify against; skipping verification step")

#     if embedding_log_data:
#         output_dict["embedding_logs"] = embedding_log_data

#     # Очистка
#     try:
#         rp_cleanup.clean(["input_objects"])
#         cleanup_job_files(job_id)
#     except Exception as e:
#         logger.warning(f"Cleanup issue: {e}", exc_info=True)

#     if error_log:
#         output_dict["error_log"] = "\n".join(error_log)

#     return output_dict

# runpod.serverless.start({"handler": run})
