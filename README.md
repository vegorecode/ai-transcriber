# WhisperX Worker

Высококачественная транскрипция речи с выравниванием по словам и опциональной диаризацией на базе WhisperX.

## Возможности

- Распознавание речи и автодетект языка
- Выравнивание распознанного текста с тайм‑кодами слов
- Диаризация (метки спикеров) при наличии токена HF
- Пакетная обработка, настройки VAD и декодирования

## Параметры входа

| Параметр | Тип | Обязат. | По умолчанию | Описание |
|---|---|---|---|---|
| `audio_file` | string | да | — | URL аудио для транскрипции |
| `model` | string | нет | `faster-whisper-large-v3-russian` | Модель Whisper/CT2 (путь в `/models` или идентификатор) |
| `language` | string | нет | `null` | ISO‑код языка, либо `null` для автоопределения |
| `device` | string | нет | `cuda` | Устройство: `cuda` или `cpu` |
| `device_index` | int | нет | `null` | Индекс CUDA‑GPU |
| `compute_type` | string | нет | `float16` | `float16`/`int8`/`float32` |
| `language_detection_min_prob` | float | нет | `0` | Порог вероятности при авто‑детекте языка |
| `language_detection_max_tries` | int | нет | `5` | Максимум итераций детекта языка |
| `initial_prompt` | string | нет | `null` | Подсказка для первого окна |
| `batch_size` | int | нет | `64` | Параллелизм декодирования |
| `beam_size` | int | нет | `null` | Размер бима |
| `temperature` | float | нет | `0` | Температура сэмплирования |
| `temperature_increment_on_fallback` | float | нет | `null` | Приращение температуры при fallback |
| `vad_onset` | float | нет | `0.500` | Порог VAD (onset) |
| `vad_offset` | float | нет | `0.363` | Порог VAD (offset) |
| `align_output` | bool | нет | `false` | Выравнивание для тайм‑кодов слов |
| `diarization` | bool | нет | `false` | Включить диаризацию |
| `diarize` | bool | нет | `null` | Алиас `diarization` |
| `huggingface_access_token` | string | нет* | `null` | Токен HF для моделей Pyannote (обязателен, если включена диаризация) |
| `min_speakers` | int | нет | `null` | Минимум спикеров |
| `max_speakers` | int | нет | `null` | Максимум спикеров |
| `length_penalty` | float | нет | `null` | Штраф за длину (beam search) |
| `output_format` | string | нет | `json` | Формат ответа |
| `debug` | bool | нет | `false` | Диагностические тайминги |
| `speaker_samples` | list | нет | `[]` | Сэмплы голосов для верификации |

## Примеры

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

Верификация спикеров по образцам

```json
{
  "input": {
    "audio_file": "https://example.com/audio/sample.mp3",
    "language": "en",
    "batch_size": 32,
    "temperature": 0.2,
    "align_output": true,
    "diarization": true,
    "huggingface_access_token": "YOUR_HF_TOKEN",
    "min_speakers": 2,
    "max_speakers": 5,
    "debug": true,
    "speaker_verification": true,
    "speaker_samples": [
      { "name": "Speaker1", "url": "https://example.com/speaker1.wav" },
      { "name": "Speaker2", "url": "https://example.com/speaker2.wav" }
    ]
  }
}
```

## Формат ответа (без диаризации)

```json
{
  "segments": [ { "start": 0.0, "end": 2.5, "text": "..." } ],
  "detected_language": "en"
}
```

## Формат ответа (с диаризацией)

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "...",
      "speaker": "SPEAKER_01",
      "words": [ { "word": "...", "start": 0.1, "end": 0.7, "speaker": "SPEAKER_01" } ]
    }
  ],
  "detected_language": "en"
}
```

## Замечания по производительности

- Подбирайте `batch_size` под объём GPU‑памяти
- Выравнивание и диаризация увеличивают время и память
- Большие файлы — дольше обработка

## Частые вопросы

1) Предупреждение про несовпадение версии `pyannote.audio` — как правило, безопасно.

2) Диаризация срабатывает только при `diarization=true` (или `diarize=true`) и валидном `huggingface_access_token` с принятыми условиями моделей `pyannote/segmentation-3.0` и `pyannote/speaker-diarization-3.1`. При отсутствии токена шаг будет пропущен.

3) По умолчанию используется русская модель `/models/faster-whisper-large-v3-russian`. Если она предзагружена на этапе сборки, cold start сокращается.

## Важно: изменения в диаризации (PyAnnote)

В проекте диаризация выполняется через PyAnnote и была обновлена для стабильной работы под RunPod Serverless:

- Явно используется `pyannote.audio.Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")` вместо обёртки `whisperx.DiarizationPipeline`.
- В пайплайн передаётся локальный путь к аудиофайлу (а не массив/буфер). Путь гарантированно локальный: RunPod сначала скачивает URL во временный файл, и этот путь переиспользуется.
- Пайплайн переводится на то же устройство, что и ASR, через явный проброс `runtime_device`.
- Отдельный пакет `pyannote.pipeline` устанавливать не требуется (и не рекомендуется) — всё обеспечивается через `pyannote.audio>=3.1`.

Почему так:
- Избегаем сбоев динамического импорта объекта `Pipeline` (частая причина ошибок вида «Could not import module 'Pipeline'» при серверлес-загрузке объектов).
- Поведение кросс-версионно стабильнее и соответствует официальным примерам PyAnnote/WhisperX.

Альтернатива (быстрее, но хрупче): передавать waveform

Можно убрать повторное чтение файла в PyAnnote и передать заранее загруженный сигнал напрямую:

```python
# Пример идеи (не включено по умолчанию из-за хрупкости интерфейса)
waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, num_samples) float32
diarize_segments = diarize_model({
    "waveform": waveform,
    "sample_rate": 16000,
})
```

Минусы подхода:
- Требует строго правильной формы/типа тензора и частоты дискретизации; легко сломать при обновлениях.
- Интерфейс PyAnnote может меняться; путь к файлу — самый стабильный вариант.

Скорость: что это даёт
- Передача waveform экономит повторное чтение/декодирование аудио. Выигрыш зависит от длительности и формата файла: 
  - короткие файлы (до 5 мин): ~0.1–0.6 сек экономии;
  - средние (5–30 мин): ~0.5–2 сек;
  - длинные (30–120 мин): ~2–6 сек.
- На фоне полной диаризации (минуты на длинных файлах) это обычно 1–3% времени, поэтому по умолчанию оставлен стабильный вариант «по пути к файлу».

## Сборка образа

```bash
docker build -t your-username/whisperx-worker:your-tag .
```

## Лицензия

Apache-2.0. См. файл `LICENSE`.
