FROM delamane/whisperx-worker-base:latest

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# Только код и стартовая команда — сборка занимает секунды
COPY src/ /app/

CMD ["python3", "-u", "/app/rp_handler.py"]


