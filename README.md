# WhisperX Worker

Высококачественная транскрипция речи на базе WhisperX с опциональной диаризацией.

Полная актуальная документация и примеры запроса/ответа — в файле `README_RUNPOD.md`. Ниже — краткие примеры.

## Параметры входа (актуально)

Список поддерживаемых полей и значения по умолчанию синхронизированы со схемой `src/rp_schema.py` и логикой `src/predict.py`:

- audio_file (string, обязательно)
- model (string, по умолчанию: faster-whisper-large-v3-russian)
- language (string|null, по умолчанию: null — автоопределение)
- compute_type (string, по умолчанию: float16)
- language_detection_min_prob (float, по умолчанию: 0)
- language_detection_max_tries (int, по умолчанию: 5)
- batch_size (int, по умолчанию: 64)
- beam_size (int|null, по умолчанию: null)
- temperature (float, по умолчанию: 0)
- temperature_increment_on_fallback (float|null, по умолчанию: null)
- vad_onset (float, по умолчанию: 0.500)
- vad_offset (float, по умолчанию: 0.363)
- min_duration_on (float, по умолчанию: 0.08)
- min_duration_off (float, по умолчанию: 0.08)
- pad_onset (float, по умолчанию: 0.0)
- pad_offset (float, по умолчанию: 0.0)
- align_output (bool, по умолчанию: false)
- diarization (bool, по умолчанию: false)
- diarize (bool|null, по умолчанию: null; алиас для совместимости)
- huggingface_access_token (string|null, по умолчанию: null; при отсутствии берётся `HF_TOKEN` из окружения)
- min_speakers (int|null)
- max_speakers (int|null)
- length_penalty (float|null)
- no_speech_threshold (float, по умолчанию: 0.6)
- log_prob_threshold (float, по умолчанию: -1.0)
- compression_ratio_threshold (float, по умолчанию: 2.4)
- output_format (string, по умолчанию: json)
- debug (bool, по умолчанию: false)

Поля `device`, `device_index`, `initial_prompt`, `speaker_samples`, `speaker_verification` не поддерживаются.

## Пороговые параметры ASR (запрос и/или ENV)

Параметры фильтрации ASR могут быть переданы в запросе либо заданы через переменные окружения. Приоритеты:

1) Значение из входного запроса (если указано);
2) Значение из ENV (если переменная окружения задана);
3) Встроенное значение по умолчанию.

- `NO_SPEECH_THRESHOLD` (alt: `ASR_NO_SPEECH_THRESHOLD`) — дефолт 0.6
- `LOGPROB_THRESHOLD` (alt: `ASR_LOGPROB_THRESHOLD`) — дефолт -1.0
- `COMPRESSION_RATIO_THRESHOLD` (alt: `ASR_COMPRESSION_RATIO_THRESHOLD`) — дефолт 2.4

## Диаризация

Используется обёртка WhisperX `whisperx.diarize.DiarizationPipeline` с моделью `pyannote/speaker-diarization-3.1`. Для работы необходим токен HF с принятыми условиями.

## Быстрый старт

Сборка монолитного образа (для простоты) или двухслойной схемы — см. `README_RUNPOD.md` и `DEV_GUIDE.md`.

## Лицензия

Apache-2.0. См. файл `LICENSE`.

## Примеры запросов

Базовая транскрипция

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
  }
}
```

Транскрипция с выравниванием и логами

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "align_output": true,
    "batch_size": 32,
    "debug": true
  }
}
```

Полная конфигурация с диаризацией

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "language": "en",
    "batch_size": 32,
    "temperature": 0.2,
    "align_output": true,
    "diarization": true,
    "huggingface_access_token": "YOUR_HF_TOKEN",
    "min_speakers": 2,
    "max_speakers": 5,
    "debug": true
  }
}
```
