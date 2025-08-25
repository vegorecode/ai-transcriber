# WhisperX Worker для RunPod Serverless

Высокачественная транскрипция с выравниванием по словам и опциональной диаризацией. Поддерживает русифицированную модель `faster-whisper-large-v3-russian` (локально предзагружена в образ).

## Ключевые особенности

- Русифицированная модель Whisper (`/models/faster-whisper-large-v3-russian`).
- Пороговые параметры ASR можно передать в запросе или через ENV:
  - `no_speech_threshold` (дефолт 0.6)
  - `log_prob_threshold` (дефолт -1.0; совм. `LOGPROB_THRESHOLD`)
  - `compression_ratio_threshold` (дефолт 2.4)
- Логи контейнера возвращаются в поле `logs` ответа (пер‑запросный буфер из `container_log.txt`). В режиме `debug: true` дополнительно возвращается объект `debug_info` со слепком конфигурации инференса и проверкой формата входного аудио (совпадает ли 16 kHz mono и был ли задействован ffmpeg‑фоллбек).
- Поддержка диаризации (`pyannote`) через обёртку WhisperX: `whisperx.diarize.DiarizationPipeline`.

> Важно: воркер — только GPU. Диаризация вызывается через `whisperx.diarize.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", device="cuda")` с передачей локального пути к аудиофайлу.

## Приоритет источников параметров

Единый приоритет для всех основных параметров:

1) Запрос (если ключ присутствует);
2) Переменные окружения (если заданы);
3) Дефолт из `src/rp_schema.py`.

Полный список ENV‑алиасов:

- model: `MODEL`
- language: — (не задаём алиас, чтобы не конфликтовать с системной `LANGUAGE`)
- compute_type: `COMPUTE_TYPE`
- language_detection_min_prob/max_tries: `LANGUAGE_DETECTION_MIN_PROB` / `LANGUAGE_DETECTION_MAX_TRIES`
- batch_size: `BATCH_SIZE`
- beam_size: `BEAM_SIZE`
- temperature: `TEMPERATURE`
- temperature_increment_on_fallback: `TEMPERATURE_INCREMENT_ON_FALLBACK`
- vad_onset/vad_offset: `VAD_ONSET` / `VAD_OFFSET`
- min_duration_on/min_duration_off: `MIN_DURATION_ON` / `MIN_DURATION_OFF`
- pad_onset/pad_offset: `PAD_ONSET` / `PAD_OFFSET`
- align_output: `ALIGN_OUTPUT`
- output_format: `OUTPUT_FORMAT`
- diarization/diarize: `DIARIZATION` / `DIARIZE`
- huggingface_access_token: `HUGGINGFACE_ACCESS_TOKEN`, `HF_TOKEN`
- min_speakers/max_speakers: `MIN_SPEAKERS` / `MAX_SPEAKERS`
- length_penalty: `LENGTH_PENALTY`
- no_speech_threshold: `NO_SPEECH_THRESHOLD`
- log_prob_threshold: `LOG_PROB_THRESHOLD` (совм. `LOGPROB_THRESHOLD`)
- compression_ratio_threshold: `COMPRESSION_RATIO_THRESHOLD`
- debug: `DEBUG`

## Сборка контейнера

Вариант A — монолит (просто, но при каждой сборке тянет модели):

```bash
# В корне папки whisperxrunpod
docker build -t your-dockerhub/whisperx-worker:latest .
```

Вариант B — двухслойная схема (рекомендуется для быстрых деплоев):

1. Соберите базовый образ с зависимостями и моделями (делается редко):

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src=./.hf_token \
  -f Dockerfile.base \
  -t your-dockerhub/whisperx-worker-base:latest .
```

- Собирайте тонкий образ приложения (секунды) при каждом изменении кода:

```bash
docker build -f Dockerfile.app -t your-dockerhub/whisperx-worker:latest .
```

Где `.hf_token` содержит строку вида `hf_...` для приватных моделей `pyannote`.

## Публикация и деплой на RunPod Serverless

1. Опубликуйте образ в реестр (Docker Hub/GHCR):

```bash
docker push your-dockerhub/whisperx-worker:latest
```

- В RunPod Hub создайте Serverless Worker:

  - Image: `your-dockerhub/whisperx-worker:latest`
  - Runs On: GPU
  - Disk: 15-20 GB
  - Переменные окружения:
    - `HF_TOKEN` — если требуется диаризация (`pyannote`).

- В RunPod Hub можно заполнить метаданные воркера (иконка, теги, пресеты) вручную.

## Формат запроса

Worker ожидает JSON вида (упрощённая форма):

```json
{
  "input": {
    "audio_file": "https://example.com/audio.wav",
    "language": null,
    "diarization": false,
    "batch_size": 32,
    "debug": false
  }
}
```

### Пример полного запроса (все ключевые параметры)

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "batch_size": 32,
    "beam_size": 5,
    "temperature": 0.0,
    "vad_onset": 0.350,
    "vad_offset": 0.250,
    "min_duration_on": 0.08,
    "min_duration_off": 0.08,
    "pad_onset": 0.0,
    "pad_offset": 0.0,
    "diarization": true,
    "min_speakers": 1,
    "max_speakers": 4,
    "length_penalty": 1.1,
    "debug": true
  }
}
```

## Параметры запроса: список и значения по умолчанию

- `audio_file` (string, обязательно): ссылка на аудио. Скачивается во временную папку RunPod.
- `model` (string, по умолчанию: `faster-whisper-large-v3-russian`): имя/ID модели или локальный путь `/models/<model>`.
- `language` (string|null, по умолчанию: `null`): ISO‑код языка; `null` — автоопределение.
- `compute_type` (string, по умолчанию: `float16`): допустимы `float16`/`int8`/`float32`.
- `batch_size` (int, по умолчанию: `32`).
- `beam_size` (int, по умолчанию: `5`).
- `temperature` (float, по умолчанию: `0`).
- `vad_onset` (float, по умолчанию: `0.300`).
- `vad_offset` (float, по умолчанию: `0.250`).
- `min_duration_on` (float, по умолчанию: `0.08`).
- `min_duration_off` (float, по умолчанию: `0.08`).
- `pad_onset` (float, по умолчанию: `0.0`).
- `pad_offset` (float, по умолчанию: `0.0`).
- `align_output` (bool, по умолчанию: `true`).
- `diarization` (bool, по умолчанию: `true`): включает диаризацию. Алиас `diarize` допустим.
- `huggingface_access_token` (string|null): токен HF; при отсутствии используется `HF_TOKEN` из окружения (если задан).
- `min_speakers` / `max_speakers` (int|null): границы числа спикеров при диаризации.
- `length_penalty` (float, по умолчанию: `1.1`).
- `no_speech_threshold` (float, по умолчанию: `0.6`): приоритет — запрос → ENV → дефолт.
- `log_prob_threshold` (float, по умолчанию: `-1.0`): приоритет — запрос → ENV → дефолт.
- `compression_ratio_threshold` (float, по умолчанию: `2.4`): приоритет — запрос → ENV → дефолт.
- `debug` (bool, по умолчанию: `false`): подробные тайминги; хвост логов включён всегда. При `true` поле `debug_info` содержит:
  - `audio`: `{ is_compatible, expected, actual, used_fallback, source, reason }`
  - `runtime`: `{ device, torch_tf32 }`
  - `model`: `{ path, compute_type, asr_options, vad_options }`
  - `decode`: `{ batch_size, beam_size, temperature, temperature_increment_on_fallback, length_penalty }`
  - `output`: `{ align_output, diarization, min_speakers, max_speakers }`

## Формат ответа

```json
{
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "..." }
  ],
  "detected_language": "ru",
  "warnings": ["..."],
  "logs": "... последние строки из container_log.txt ..."
}
```

- При диаризации сегменты включают `speaker`.

## Логи

- Все события пишутся в `container_log.txt` и добавляются в ответ (`logs`).
- Для подробных таймингов укажите `debug: true`.
- В режиме `debug: true` дополнительно фиксируются:
  - время загрузки модели;
  - время загрузки аудио;
  - время транскрипции;
  - время выравнивания (если `align_output=true`);
  - время диаризации (если `diarization=true`);
  - максимальный резерв памяти GPU за время выполнения.

## Примечание по диаризации (производительность)

- По умолчанию `DiarizationPipeline` читает аудиофайл по локальному пути — это самый устойчивый режим для RunPod.
- Теоретически можно передать заранее загруженный сигнал (`{"waveform": torch.Tensor, "sample_rate": 16000}`), чтобы избежать повторного чтения. Практический выигрыш невелик, поэтому оставлен стабильный режим по пути.

## Локальный запуск

Локальный запуск для отладки не поддерживается в этом репозитории. Предусмотрен только режим RunPod Serverless. Для проверки используйте серверлес-пресеты и образ из реестра.

## Переменные окружения

- `HF_TOKEN` — токен Hugging Face (логинится на старте воркера). Используется:
  - для аутентификации в HF (глобально);
  - при сборке для докачки приватных моделей через BuildKit secret.
- `NO_SPEECH_THRESHOLD` — порог пропуска тишины; по умолчанию `0.6`.
- `LOG_PROB_THRESHOLD` — порог средней лог‑вероятности; по умолчанию `-1.0` (совм. `LOGPROB_THRESHOLD`).
- `COMPRESSION_RATIO_THRESHOLD` — порог отношения сжатия; по умолчанию `2.4`.

Пороговые параметры используются внутри `predict.py` и могут быть переданы в запросе либо через ENV (приоритет: запрос → ENV → дефолт).

## Частые проблемы

- Отсутствует `libsndfile1` → добавлено в `Dockerfile` для `librosa`.
- Нет токена HF → диаризация будет пропущена, появится предупреждение в логах.
- Большие файлы → увеличьте `containerDiskInGb` и/или снизьте `batch_size`.
- Для полного отсутствия докачек на старте — передайте `HF_TOKEN` на этапе сборки через BuildKit secret.
