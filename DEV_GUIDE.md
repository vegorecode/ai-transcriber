## Гайд по разработке и деплою (быстрые билды без докачки моделей)

Этот гайд показывает, как разделить образ на два слоя:
- базовый слой с весами и зависимостями;
- тонкий слой приложения, содержащий только код.

Так вы сможете менять код и собирать образ за секунды, не перекачивая модели.

### Структура

- `Dockerfile.base` — базовый образ: CUDA/PyTorch, Python-зависимости и предзагруженные модели (`/models`).
- `Dockerfile.app` — тонкий образ приложения: наследуется от base, копирует `src/` и запускает `rp_handler.py`.

### Разовая подготовка базового образа

1) Убедитесь, что у вас есть файл `.hf_token` с содержимым вида `hf_...` (чтобы предзагрузить приватные модели `pyannote`).

2) Соберите и запушьте базовый образ (делается редко):

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src=./.hf_token \
  -f Dockerfile.base \
  -t <your-namespace>/whisperx-worker-base:latest .

docker push <your-namespace>/whisperx-worker-base:latest
```

Пример для `delamane`:

```bash
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,src=./.hf_token -f Dockerfile.base -t delamane/whisperx-worker-base:latest .
docker push delamane/whisperx-worker-base:latest
```

### Настройка `Dockerfile.app`

Верхняя строка должна ссылаться на ваш базовый образ:

```dockerfile
FROM <your-namespace>/whisperx-worker-base:latest
```

Пример:

```dockerfile
FROM delamane/whisperx-worker-base:latest
```

### Быстрая итерация разработки (каждый раз при изменении кода)

1) Соберите и запушьте тонкий образ приложения:

```bash
docker build -f Dockerfile.app -t <your-namespace>/whisperx-worker:latest .
docker push <your-namespace>/whisperx-worker:latest
```

Пример:

```bash
docker build -f Dockerfile.app -t delamane/whisperx-worker:latest .
docker push delamane/whisperx-worker:latest
```

2) В RunPod Serverless укажите образ приложения `<your-namespace>/whisperx-worker:latest` и переменную окружения `HF_TOKEN`. Для обновления кода — просто перезапускайте воркер или создавайте новую ревизию пресета после push.

### Когда пересобирать базовый образ

- Меняются версии CUDA/PyTorch/библиотек в `builder/requirements.txt`;
- Меняете состав/версии моделей или логику их предзагрузки;
- Нужны системные пакеты/обновления (apt-get) поверх базового образа.

Во всех остальных случаях достаточно собирать только `Dockerfile.app`.

### Частые ошибки и решения

- "Собрал base под тег app":
  - Неправильно: `-t <ns>/whisperx-worker:latest -f Dockerfile.base`
  - Правильно: `-t <ns>/whisperx-worker-base:latest -f Dockerfile.base`
- RunPod тянет не тот образ:
  - Проверьте, что в RunPod указан `<ns>/whisperx-worker:latest`, а не `-base`.
- Диаризация не сработала:
  - Проверьте `HF_TOKEN` в окружении воркера. Без него шаг диаризации будет пропущен.
- Обновили base, а app всё ещё старый:
  - Выполните `docker pull <ns>/whisperx-worker-base:latest` локально перед сборкой app, чтобы убедиться, что используете свежую базу.

### Альтернатива (быстрый старт с уже собранным монолитом)

Если у вас уже есть старый монолитный образ `<ns>/whisperx-worker:latest`, можно временно использовать его как базовый:

```bash
docker pull <ns>/whisperx-worker:latest
docker tag <ns>/whisperx-worker:latest <ns>/whisperx-worker-base:latest
docker push <ns>/whisperx-worker-base:latest
```

Затем переходите к сборке app. Рекомендуется в дальнейшем собрать корректный базовый образ из `Dockerfile.base`.


