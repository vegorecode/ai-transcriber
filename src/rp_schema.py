INPUT_VALIDATIONS = {
    "audio_file": {"type": str, "required": True},
    "model": {
        "type": str,
        "required": False,
        "default": "faster-whisper-large-v3-russian",
    },
    "language": {"type": str, "required": False, "default": None},
    # Только GPU: фиксируем cuda:0 и не принимаем CPU‑ветки
    "compute_type": {"type": str, "required": False, "default": "float16"},
    # language_detection_* скрыты из публичной таблицы; доступны, но ставим дефолты
    "language_detection_min_prob": {"type": float, "required": False, "default": 0},
    "language_detection_max_tries": {"type": int, "required": False, "default": 5},
    # initial_prompt удалён из внешнего API
    "batch_size": {"type": int, "required": False, "default": 64},
    "beam_size": {"type": int, "required": False, "default": None},
    "temperature": {"type": float, "required": False, "default": 0},
    "temperature_increment_on_fallback": {
        "type": float,
        "required": False,
        "default": None,
    },
    "vad_onset": {"type": float, "required": False, "default": 0.500},
    "vad_offset": {"type": float, "required": False, "default": 0.363},
    "align_output": {"type": bool, "required": False, "default": False},
    "output_format": {"type": str, "required": False, "default": "json"},
    "diarization": {"type": bool, "required": False, "default": False},
    # alias for compatibility with other workers
    "diarize": {"type": bool, "required": False, "default": None},
    "huggingface_access_token": {"type": str, "required": False, "default": None},
    "min_speakers": {"type": int, "required": False, "default": None},
    "max_speakers": {"type": int, "required": False, "default": None},
    "length_penalty": {"type": float, "required": False, "default": None},
    "debug": {"type": bool, "required": False, "default": False},
}
