# Точка входа сервера: загрузка переменных окружения и настройка логирования
import io
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

    def _is_safe_job_dir(path: str, job_id_value: str) -> bool:
        norm = os.path.normpath(path)
        # Разрешаем удалять только директории, чьё имя ровно равно job_id
        return os.path.basename(norm) == job_id_value and (
            norm.startswith(os.path.normpath("/app/jobs"))
            or norm.startswith(os.path.normpath("/jobs"))
        )

    for job_path in unique_candidates:
        if not _is_safe_job_dir(job_path, job_id):
            logger.debug(f"Пропускаем небезопасный путь очистки: {job_path}")
            continue
        if os.path.exists(job_path):
            try:
                import shutil

                shutil.rmtree(job_path)
                logger.info(f"Удалена временная директория задания: {job_path}")
                return
            except Exception as e:
                # Переходим к следующему кандидату, не прерываясь на ошибке
                logger.error(f"Ошибка удаления {job_path}: {str(e)}", exc_info=True)
                continue

    logger.debug(f"Временная директория не найдена для job_id={job_id}")


# --------------------------------------------------------------------
# Основная точка входа (обработчик задания)
# --------------------------------------------------------------------
# очищено: удалена неиспользуемая переменная error_log


def _merge_with_defaults(raw_input: dict) -> dict:
    """Заполнить отсутствующие поля значениями по умолчанию из INPUT_VALIDATIONS.

    Правило: если ключ отсутствует в запросе, берём default из схемы; если ключ
    присутствует (включая явное значение None), используем переданное значение.
    """
    normalized = {}
    for key, spec in INPUT_VALIDATIONS.items():
        default_value = spec.get("default")
        if key in raw_input:
            normalized[key] = raw_input[key]
        else:
            normalized[key] = default_value
    return normalized


def _parse_env_float(*env_names: str):
    """Вернуть первое найденное значение float из переменных окружения.

    Пример: _parse_env_float("NO_SPEECH_THRESHOLD", "ASR_NO_SPEECH_THRESHOLD")
    """
    for name in env_names:
        val = os.getenv(name)
        if val is not None and val != "":
            try:
                return float(val)
            except Exception:
                # игнорируем некорректные значения
                pass
    return None


def run(job):
    job_id = job["id"]
    job_input = job["input"]

    # Сбор предупреждений для ответа
    warnings = []

    # Пер‑запросный буфер логов: собираем только логи текущей задачи
    buf = io.StringIO()
    temp_handler = logging.StreamHandler(buf)
    temp_handler.setLevel(logging.DEBUG)
    temp_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # Подключаем к логгерам обработчика и предиктора
    logger.addHandler(temp_handler)
    _predict_logger = logging.getLogger("predict")
    _predict_logger.addHandler(temp_handler)

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
        # Возвращаем только логи текущего запроса
        try:
            logger.removeHandler(temp_handler)
            _predict_logger.removeHandler(temp_handler)
        finally:
            temp_handler.close()
        return {"error": validated["errors"]}

    # Нормализуем вход: заполняем отсутствующие ключи дефолтами из схемы
    normalized_input = _merge_with_defaults(job_input)

    # ------------- 1) скачиваем основное аудио ----------------------
    try:
        audio_file_path = download_files_from_urls(job_id, [job_input["audio_file"]])[0]
        logger.debug(f"Audio downloaded → {audio_file_path}")
    except Exception as e:
        logger.error("Скачивание аудио не удалось", exc_info=True)
        return {"error": f"download audio: {e}"}

    # ------------- 3) запуск WhisperX / VAD / диаризации ------------
    # Значения порогов с учётом приоритета: запрос → ENV → дефолт
    _no_speech_value = (
        normalized_input.get("no_speech_threshold")
        if "no_speech_threshold" in job_input
        else (
            _parse_env_float("NO_SPEECH_THRESHOLD", "ASR_NO_SPEECH_THRESHOLD")
            or normalized_input.get("no_speech_threshold")
        )
    )
    _log_prob_value = (
        normalized_input.get("log_prob_threshold")
        if "log_prob_threshold" in job_input
        else (
            _parse_env_float("LOGPROB_THRESHOLD", "ASR_LOGPROB_THRESHOLD")
            or normalized_input.get("log_prob_threshold")
        )
    )
    _compression_ratio_value = (
        normalized_input.get("compression_ratio_threshold")
        if "compression_ratio_threshold" in job_input
        else (
            _parse_env_float(
                "COMPRESSION_RATIO_THRESHOLD", "ASR_COMPRESSION_RATIO_THRESHOLD"
            )
            or normalized_input.get("compression_ratio_threshold")
        )
    )

    predict_input = {
        "audio_file": audio_file_path,
        "model": normalized_input.get("model"),
        "language": normalized_input.get("language"),
        "compute_type": normalized_input.get("compute_type"),
        "language_detection_min_prob": normalized_input.get(
            "language_detection_min_prob"
        ),
        "language_detection_max_tries": normalized_input.get(
            "language_detection_max_tries"
        ),
        "batch_size": normalized_input.get("batch_size"),
        "beam_size": normalized_input.get("beam_size"),
        "temperature": normalized_input.get("temperature"),
        "temperature_increment_on_fallback": normalized_input.get(
            "temperature_increment_on_fallback"
        ),
        "vad_onset": normalized_input.get("vad_onset"),
        "vad_offset": normalized_input.get("vad_offset"),
        "min_duration_on": normalized_input.get("min_duration_on"),
        "min_duration_off": normalized_input.get("min_duration_off"),
        "pad_onset": normalized_input.get("pad_onset"),
        "pad_offset": normalized_input.get("pad_offset"),
        "align_output": normalized_input.get("align_output"),
        "diarization": normalized_input.get("diarization"),
        "diarize": normalized_input.get("diarize"),
        # токен: сначала из входа, иначе из ENV (hf_token загружен выше)
        "huggingface_access_token": normalized_input.get("huggingface_access_token")
        or hf_token,
        "min_speakers": normalized_input.get("min_speakers"),
        "max_speakers": normalized_input.get("max_speakers"),
        "length_penalty": normalized_input.get("length_penalty"),
        "no_speech_threshold": _no_speech_value,
        "log_prob_threshold": _log_prob_value,
        "compression_ratio_threshold": _compression_ratio_value,
        "output_format": normalized_input.get("output_format"),
        "debug": normalized_input.get("debug"),
    }

    try:
        result = MODEL.predict(**predict_input)
    except Exception as e:
        logger.error("Ошибка распознавания WhisperX", exc_info=True)
        # Перед возвратом снимаем пер‑запросный хендлер
        try:
            logger.removeHandler(temp_handler)
            _predict_logger.removeHandler(temp_handler)
        finally:
            temp_handler.close()
        return {"error": f"predict: {e}"}

    output_dict = {
        "segments": result.segments,
        "detected_language": result.detected_language,
    }
    # 4) идентификация спикеров — отключено

    # Добавляем предупреждения к ответу, если есть
    if warnings:
        output_dict["warnings"] = warnings

    # 5) приложим логи только текущего запроса
    output_dict["logs"] = buf.getvalue()

    # 5) очистка и возврат результата
    try:
        rp_cleanup.clean(["input_objects"])
        cleanup_job_files(job_id, audio_file_path)
    except Exception as e:
        logger.warning(f"Cleanup issue: {e}", exc_info=True)

    # Снимаем пер‑запросный хендлер и освобождаем буфер
    try:
        logger.removeHandler(temp_handler)
        _predict_logger.removeHandler(temp_handler)
    finally:
        temp_handler.close()
    return output_dict


# Запускаем воркер после определения всех функций
runpod.serverless.start({"handler": run})
