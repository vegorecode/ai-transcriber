FROM delamane/whisperx-worker-base:latest

SHELL ["/bin/bash", "-c"]
WORKDIR /app

COPY src/ /app/

CMD ["python3", "-u", "/app/rp_handler.py"]
