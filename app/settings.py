from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    confloat,
    conint,
    model_validator,
)


def _get_env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


class AppSettings(BaseModel):
    """Настройки транскрипции с валидаторами.

    Значения по умолчанию берутся из переменных окружения, если доступны,
    иначе используются разумные дефолты, синхронизированные с `run.sh`.
    """

    language: str = Field(default_factory=lambda: _get_env_str("LANGUAGE", "ru"))
    model: str = Field(
        default_factory=lambda: _get_env_str("MODEL", "openai/whisper-large-v3")
    )
    device: Literal["cuda", "cpu"] = Field(
        default_factory=lambda: _get_env_str("DEVICE", "cuda")
    )
    compute_type: Literal["float16", "int8", "float32"] = Field(
        default_factory=lambda: _get_env_str("COMPUTE_TYPE", "float16")
    )

    batch_size: PositiveInt = Field(
        default_factory=lambda: _get_env_int("BATCH_SIZE", 2)
    )
    beam_size: PositiveInt = Field(default_factory=lambda: _get_env_int("BEAM_SIZE", 5))
    temperature: confloat(ge=0.0, le=1.0) = Field(
        default_factory=lambda: _get_env_float("TEMPERATURE", 0.0)
    )
    temperature_increment_on_fallback: confloat(ge=0.0, le=1.0) = Field(
        default_factory=lambda: _get_env_float("TEMP_INC", 0.2)
    )

    vad_method: Literal["pyannote", "silero"] = Field(
        default_factory=lambda: _get_env_str("VAD", "pyannote")
    )
    vad_onset: confloat(ge=0.0, le=1.0) = Field(
        default_factory=lambda: _get_env_float("VAD_ONSET", 0.30)
    )
    vad_offset: confloat(ge=0.0, le=1.0) = Field(
        default_factory=lambda: _get_env_float("VAD_OFFSET", 0.25)
    )

    diarize: bool = True
    min_speakers: conint(ge=1) = Field(
        default_factory=lambda: _get_env_int("MIN_SPK", 1)
    )
    max_speakers: conint(ge=1) = Field(
        default_factory=lambda: _get_env_int("MAX_SPK", 4)
    )

    output_format: Literal["srt", "vtt", "txt", "json"] = Field(
        default_factory=lambda: _get_env_str("FORMAT", "srt")
    )
    length_penalty: confloat(ge=0.0, le=5.0) = Field(
        default_factory=lambda: _get_env_float("LENGTH_PENALTY", 1.1)
    )

    @model_validator(mode="after")
    def _validate_speakers(self) -> "AppSettings":
        if self.max_speakers < self.min_speakers:
            raise ValueError("max_speakers не может быть меньше min_speakers")
        return self


class SettingsStore:
    """Хранилище настроек в JSON-файле с безопасной загрузкой/сохранением.

    При отсутствии файла возвращает настройки по умолчанию (из окружения).
    """

    def __init__(
        self, config_dir: Path | str = "/work/config", filename: str = "settings.json"
    ) -> None:
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.config_dir / filename

    def load(self) -> AppSettings:
        if not self.file_path.exists():
            return AppSettings()
        try:
            content = json.loads(self.file_path.read_text(encoding="utf-8"))
            return AppSettings(**content)
        except Exception:
            # На случай поломанного файла — откат к дефолтам
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        data = json.dumps(settings.model_dump(), ensure_ascii=False, indent=2)
        tmp_path = self.file_path.with_suffix(".tmp")
        tmp_path.write_text(data, encoding="utf-8")
        tmp_path.replace(self.file_path)

    def update_from_dict(self, updates: dict) -> AppSettings:
        current = self.load()
        merged = {**current.model_dump(), **updates}
        new_settings = AppSettings(**merged)
        self.save(new_settings)
        return new_settings
