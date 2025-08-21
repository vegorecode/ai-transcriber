# Точка входа сервера: загрузка переменных окружения и настройка логирования
import logging
import os
import sys

# Гарантированно отключаем vision в Transformers ещё до возможных импортов transformers
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import runpod
import torch
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login, whoami
from runpod.serverless.utils import download_files_from_urls, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from predict import Predictor
from rp_schema import INPUT_VALIDATIONS

# Загружаем переменные окружения из .env (если есть)
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

# Примечание: функционал верификации спикеров по образцам отключён и подлежит удалению.

# Только GPU: падать, если CUDA недоступна
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU недоступна: этот воркер поддерживает только GPU")

# Логирование
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Берём токен Hugging Face из окружения (если есть)
hf_token = os.environ.get("HF_TOKEN", "").strip()

if hf_token:
    try:
        token_preview = (hf_token[:4] + "…") if hf_token else ""
        logger.debug(f"HF_TOKEN загружен (prefix): {repr(token_preview)}")
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


def cleanup_job_files(
    job_id, audio_file_path=None, jobs_directories=("/app/jobs", "/jobs")
):
    """Удаляет временную директорию задания, пытаясь корректно определить путь.

    Предпочитает путь, вычисленный относительно фактического `audio_file_path`,
    затем пробует стандартные каталоги из `jobs_directories`.
    """
    candidates = []
    if audio_file_path:
        try:
            # /app/jobs/<job_id>/downloaded_files/<file>
            downloaded_dir = os.path.dirname(audio_file_path)
            # приоритет — удалить каталог задания целиком
            job_dir_from_two_levels_up = os.path.abspath(
                os.path.join(downloaded_dir, "..", "..")
            )
            # запасной вариант — удалить папку downloaded_files
            job_dir_from_one_level_up = os.path.abspath(
                os.path.join(downloaded_dir, "..")
            )
            candidates.extend([job_dir_from_two_levels_up, job_dir_from_one_level_up])
        except Exception:
            pass

    for base in jobs_directories:
        candidates.append(os.path.join(base, job_id))

    seen = set()
    unique_candidates = []
    for path in candidates:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            unique_candidates.append(norm)

    for job_path in unique_candidates:
        if os.path.exists(job_path):
            try:
                import shutil

                shutil.rmtree(job_path)
                logger.info(f"Удалена временная директория задания: {job_path}")
                return
            except Exception as e:
                logger.error(f"Ошибка удаления {job_path}: {str(e)}", exc_info=True)
                return

    logger.debug(f"Временная директория не найдена для job_id={job_id}")


# --------------------------------------------------------------------
# Основная точка входа (обработчик задания)
# --------------------------------------------------------------------
error_log = []


def run(job):
    job_id = job["id"]
    job_input = job["input"]

    # Сбор предупреждений для ответа
    warnings = []

    # Диагностика версий
    try:
        import transformers  # noqa: WPS433

        try:
            import torchvision as _tv  # noqa: WPS433

            tv_ver = _tv.__version__
        except Exception:
            tv_ver = "n/a"
        import torch as _t  # noqa: WPS433

        logger.info(
            "Libs: torch=%s, transformers=%s, torchvision=%s, NO_TORCHVISION=%s",
            getattr(_t, "__version__", "?"),
            getattr(transformers, "__version__", "?"),
            tv_ver,
            os.environ.get("TRANSFORMERS_NO_TORCHVISION"),
        )
    except Exception:
        logger.debug("Version logging failed", exc_info=True)

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

    # ------------- 2) загрузка примеров голосов — отключено ---------
    # Функциональность speaker_samples не используется и будет удалена.
    # -----------------------------------------------------------------

    # ------------- 3) запуск WhisperX / VAD / диаризации ------------
    predict_input = {
        "audio_file": audio_file_path,
        "model": job_input.get("model", "faster-whisper-large-v3-russian"),
        "language": job_input.get("language"),
        "compute_type": job_input.get("compute_type", "float16"),
        "language_detection_min_prob": job_input.get("language_detection_min_prob", 0),
        "language_detection_max_tries": job_input.get(
            "language_detection_max_tries", 5
        ),
        "batch_size": job_input.get("batch_size", 16),
        "beam_size": job_input.get("beam_size", 5),
        "temperature": job_input.get("temperature", 0),
        "temperature_increment_on_fallback": job_input.get(
            "temperature_increment_on_fallback"
        ),
        "vad_onset": job_input.get("vad_onset", 0.50),
        "vad_offset": job_input.get("vad_offset", 0.363),
        "align_output": job_input.get("align_output", False),
        "diarization": job_input.get("diarization", False),
        "diarize": job_input.get("diarize"),
        # токен: сначала из входа, иначе из ENV (hf_token загружен выше)
        "huggingface_access_token": job_input.get("huggingface_access_token")
        or hf_token,
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
    # 4) идентификация спикеров — отключено

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
        cleanup_job_files(job_id, audio_file_path)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    return output_dict


# Запускаем воркер после определения всех функций
runpod.serverless.start({"handler": run})
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
