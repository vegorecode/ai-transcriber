## WhisperX Worker для RunPod Serverless

Высококачественная транскрипция с выравниванием по словам и опциональной диаризацией. Поддерживает русифицированную модель `faster-whisper-large-v3-russian` (локально предзагружена в образ).

### Ключевые особенности

- Русифицированная модель Whisper (`/models/faster-whisper-large-v3-russian`)
- Жёстко заданные пороги для модели (не приходят извне):
  - `no_speech_threshold = 0.6`
  - `log_prob_threshold = -1.0`
  - `compression_ratio_threshold = 2.4`
- Логи контейнера возвращаются в поле `logs` ответа (хвост `container_log.txt`).
- Поддержка диаризации (`pyannote`), верификация спикеров по образцам.

### Сборка контейнера

```bash
# В корне папки whisperxrunpod
docker build -t your-dockerhub/whisperx-worker:latest .
```

Примечания:
- Для скачивания приватных моделей (`pyannote`) при сборке можно передать секрет с токеном HF через BuildKit:
  ```bash
  DOCKER_BUILDKIT=1 docker build \
    --secret id=hf_token,src=./.hf_token \
    -t your-dockerhub/whisperx-worker:latest .
  ```
  Где файл `.hf_token` содержит строку вида `hf_...`.

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

Worker ожидает JSON вида:
```json
{
  "input": {
    "audio_file": "https://example.com/audio.wav",
    "align_output": true,
    "diarization": false,
    "language": null,
    "batch_size": 32,
    "debug": true
  }
}
```

Поддерживаемые поля совпадают с `README.md` и схемой `src/rp_schema.py`. Пороговые параметры `no_speech_threshold`, `log_prob_threshold`, `compression_ratio_threshold` уже заданы в коде и извне не принимаются.

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

- При диаризации сегменты включают `speaker`, при верификации — могут включать `speaker_id` и `similarity`.

### Логи

- Все события пишутся в `container_log.txt` и добавляются в ответ (`logs`).
- Для подробных таймингов укажите `debug: true`.

### Локальный запуск

Локальный запуск для отладки не поддерживается в этом репозитории. Предусмотрен только режим RunPod Serverless. Для проверки используйте серверлес-пресеты и образ из реестра.

### Переменные окружения

- `HF_TOKEN` — токен Hugging Face (нужен для `pyannote` моделей).
- `NO_SPEECH_THRESHOLD`, `LOGPROB_THRESHOLD`, `COMPRESSION_RATIO_THRESHOLD` — переопределяют значения по умолчанию, если заданы.
- `ASR_NO_SPEECH_THRESHOLD`, `ASR_LOGPROB_THRESHOLD`, `ASR_COMPRESSION_RATIO_THRESHOLD` — альтернативные имена (обратная совместимость).

### Частые проблемы

- Отсутствует `libsndfile1` → добавлено в `Dockerfile` для `librosa`.
- Нет токена HF → диаризация будет пропущена, появится предупреждение в логах.
- Большие файлы → увеличьте `containerDiskInGb` и/или снизьте `batch_size`.
- Для полного отсутствия докачек на старте — передайте `HF_TOKEN` на этапе сборки через BuildKit secret.


