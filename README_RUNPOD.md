## WhisperX Worker для RunPod Serverless

Высококачественная транскрипция с выравниванием по словам и опциональной диаризацией. Поддерживает русифицированную модель `faster-whisper-large-v3-russian` (локально предзагружена в образ).

### Ключевые особенности

- Русифицированная модель Whisper (`/models/faster-whisper-large-v3-russian`)
- Жёстко заданные пороги для модели (не приходят извне):
  - `no_speech_threshold = 0.6`
  - `log_prob_threshold = -1.0`
  - `compression_ratio_threshold = 2.4`
- Логи контейнера возвращаются в поле `logs` ответа (хвост `container_log.txt`).
- Поддержка диаризации (`pyannote`) через обёртку WhisperX: `whisperx.diarize.DiarizationPipeline`.
  
> Важно: воркер — только GPU. Диаризация вызывается через `whisperx.diarize.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.1", device="cuda")` с передачей локального пути к аудиофайлу.

### Сборка контейнера

Вариант A — монолит (просто, но при каждой сборке тянет модели):

```bash
# В корне папки whisperxrunpod
docker build -t your-dockerhub/whisperx-worker:latest .
```

Вариант B — двухслойная схема (рекомендуется для быстрых деплоев):

1) Соберите базовый образ с зависимостями и моделями (делается редко):

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src=./.hf_token \
  -f Dockerfile.base \
  -t your-dockerhub/whisperx-worker-base:latest .
```

2) Для каждого изменения кода собирайте тонкий образ приложения (секунды):

```bash
docker build -f Dockerfile.app -t your-dockerhub/whisperx-worker:latest .
```

Где `.hf_token` содержит строку вида `hf_...` для приватных моделей `pyannote`.

### Публикация и деплой на RunPod Serverless

1. Опубликуйте образ в реестр (Docker Hub/GHCR):
   ```bash
   docker push your-dockerhub/whisperx-worker:latest
   ```
2. В RunPod Hub создайте Serverless Worker:
   - Image: `your-dockerhub/whisperx-worker:latest`
   - Runs On: GPU
   - Disk: 30–50 GB (если предзагружаете модели)
   - Переменные окружения:
     - `HF_TOKEN` — если требуется диаризация (`pyannote`).
3. В RunPod Hub можно заполнить метаданные воркера (иконка, теги, пресеты) вручную.

### Формат запроса

Worker ожидает JSON вида (упрощённая форма):
```json
{
  "input": {
    "audio_file": "https://example.com/audio.wav",
    "language": null,
    "diarization": false,
    "batch_size": 64,
    "debug": false
  }
}
```

#### Пример полного запроса (все ключевые параметры)

```json
{
  "input": {
    "audio_file": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "batch_size": 16,
    "beam_size": 5,
    "temperature": 0.0,
    "temperature_increment_on_fallback": 0.2,
    "vad_onset": 0.350,
    "vad_offset": 0.250,
    "diarization": true,
    "min_speakers": 1,
    "max_speakers": 4,
    "length_penalty": 1.0,
    "debug": true
  }
}
```

Поддерживаемые поля совпадают со схемой `src/rp_schema.py` и логикой `src/rp_handler.py`/`src/predict.py`. Пороговые параметры `no_speech_threshold`, `log_prob_threshold`, `compression_ratio_threshold` не приходят в запросе, но могут быть переопределены переменными окружения (см. ниже).

### Параметры запроса: полный список, обязательность и источники значений по умолчанию

| Параметр | Тип | Обязат. | Значение по умолчанию | Источник/фоллбэк | Примечания |
|---|---|---|---|---|---|
| `audio_file` | string (URL) | да | — | — | Ссылка на аудио. Скачивается во временную папку RunPod. |
| `model` | string | нет | `faster-whisper-large-v3-russian` | Значение по умолчанию из кода (`rp_schema.py`/`predict.py`). Если в контейнере существует `/models/<model>`, используется локальный путь, иначе переданное имя/ID. |
| `language` | string или `null` | нет | `null` | По умолчанию автоопределение языка. |
| `device` | — | — | — | Не поддерживается: воркер всегда использует `cuda:0`. |
| `device_index` | — | — | — | Не поддерживается. |
| `compute_type` | string | нет | `float16` | Значение из кода. Допустимы `float16`/`int8`/`float32`. |
<!-- language_detection_* и initial_prompt скрыты для упрощения интерфейса -->
| `batch_size` | int | нет | `64` | Значение из кода. Увеличивайте/уменьшайте под доступную VRAM. |
| `beam_size` | int или `null` | нет | `null` | Добавляется в опции декодирования, если задан. |
| `temperature` | float | нет | `0` | Значение из кода. |
| `temperature_increment_on_fallback` | float или `null` | нет | `null` | Применяется при fallback‑декодировании, если задано. |
| `vad_onset` | float | нет | `0.500` | Значение из кода. Порог VAD (onset). |
| `vad_offset` | float | нет | `0.363` | Значение из кода. Порог VAD (offset). |
<!-- align_output и output_format скрыты: ответ всегда JSON -->
| `diarization` | bool | нет | `false` | Включает диаризацию. Допускается алиас `diarize` (для совместимости), но в новых конфигурациях используйте `diarization`. |
| `huggingface_access_token` | string или `null` | нет | `null` | Не обязателен: при его отсутствии воркер использует `HF_TOKEN` из окружения (если задан) для диаризации. |
| `min_speakers` | int или `null` | нет | `null` | Передаётся в пайплайн диаризации, если `diarization=true`. |
| `max_speakers` | int или `null` | нет | `null` | Передаётся в пайплайн диаризации, если `diarization=true`. |
| `length_penalty` | float или `null` | нет | `null` | Опция beam‑поиска; добавляется при наличии. |
| `debug` | bool | нет | `false` | При `true` включает подробные тайминги (загрузка модели/аудио, транскрипция, выравнивание, диаризация) и метрику пика выделенной памяти GPU; хвост `container_log.txt` прикладывается к ответу всегда. |
<!-- функциональность speaker_verification / speaker_samples удалена как неиспользуемая -->

Замечания по приоритетам/поведению:
- Допускается алиас `diarize`, но рекомендуется использовать `diarization`.
- Диаризация выполняется только при `diarization=true` (или `diarize=true`). Для доступа к моделям `pyannote` используется `huggingface_access_token`, при его отсутствии — `HF_TOKEN` из окружения (если задан).  
  Если ни одного токена нет — шаг диаризации будет пропущен и добавится предупреждение в логи.

### Формат ответа

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

### Логи

- Все события пишутся в `container_log.txt` и добавляются в ответ (`logs`).
- Для подробных таймингов укажите `debug: true`.
- В режиме `debug: true` дополнительно фиксируются:
  - время загрузки модели;
  - время загрузки аудио;
  - время транскрипции;
  - время выравнивания (если `align_output=true`);
  - время диаризации (если `diarization=true`);
  - максимальный резерв памяти GPU за время выполнения.
  
### Примечание по диаризации (производительность)

- По умолчанию `DiarizationPipeline` читает аудиофайл по локальному пути — это самый устойчивый режим для RunPod.
- Теоретически можно передать заранее загруженный сигнал (`{"waveform": torch.Tensor, "sample_rate": 16000}`), чтобы избежать повторного чтения. Практический выигрыш невелик, поэтому оставлен стабильный режим по пути.

### Локальный запуск

Локальный запуск для отладки не поддерживается в этом репозитории. Предусмотрен только режим RunPod Serverless. Для проверки используйте серверлес-пресеты и образ из реестра.

### Переменные окружения

- `HF_TOKEN` — токен Hugging Face (логинится на старте воркера). Используется:
  - для аутентификации в HF (глобально);
  - как фоллбэк для загрузки образцов в `speaker_samples`, если не задан `huggingface_access_token` в запросе;
  - при сборке для докачки приватных моделей через BuildKit secret.
- `NO_SPEECH_THRESHOLD` (alt: `ASR_NO_SPEECH_THRESHOLD`) — порог пропуска тишины; по умолчанию `0.6`.
- `LOGPROB_THRESHOLD` (alt: `ASR_LOGPROB_THRESHOLD`) — порог средней лог‑вероятности; по умолчанию `-1.0`.
- `COMPRESSION_RATIO_THRESHOLD` (alt: `ASR_COMPRESSION_RATIO_THRESHOLD`) — порог отношения сжатия; по умолчанию `2.4`.

Пороговые параметры используются внутри `predict.py` и не могут быть заданы во входном JSON; только через ENV. Если ENV не указаны, берутся встроенные значения по умолчанию, указанные выше.

### Частые проблемы

- Отсутствует `libsndfile1` → добавлено в `Dockerfile` для `librosa`.
- Нет токена HF → диаризация будет пропущена, появится предупреждение в логах.
- Большие файлы → увеличьте `containerDiskInGb` и/или снизьте `batch_size`.
- Для полного отсутствия докачек на старте — передайте `HF_TOKEN` на этапе сборки через BuildKit secret.


