#!/bin/sh
# ------------------------------------------------------------
#  run.sh — «одношаговый» запуск WhisperX + CT2-конвертация
#             с полной поддержкой моделей Whisper Large-v3
#             + предварительная конвертация в WAV 16kHz mono+нормализация
# ------------------------------------------------------------
set -eu

# --------------------- системные мелочи ----------------------
export HF_HUB_ENABLE_HF_TRANSFER=0        # тише логи hf_transfer
[ -d /venv/bin ] && PATH="/venv/bin:$PATH"  # whisperx в PATH

LANGUAGE="${LANGUAGE:-ru}"
MODEL="${MODEL:-openai/whisper-large-v3}"
DEVICE="${DEVICE:-cuda}"
COMPUTE_TYPE="${COMPUTE_TYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
BEAM_SIZE="${BEAM_SIZE:-2}"
TEMPERATURE="${TEMPERATURE:-0}"
TEMP_INC="${TEMP_INC:-0.2}"
VAD="${VAD:-pyannote}"
VAD_ONSET="${VAD_ONSET:-0.30}"
VAD_OFFSET="${VAD_OFFSET:-0.25}"
MIN_SPK="${MIN_SPK:-1}"
MAX_SPK="${MAX_SPK:-4}"
FORMAT="${FORMAT:-srt}"
LENGTH_PENALTY="${LENGTH_PENALTY:-1.1}"

DECODER_OPTS="--beam_size $BEAM_SIZE --length_penalty $LENGTH_PENALTY"
IN_DIR="/work/audio"
CONV_DIR="/tmp/converted"  # writable tmp-директория, файлы удалятся при выходе контейнера
OUT_DIR="/work/out"
EXTS="mp3 wav ogg m4a flac webm mp4 mkv"

echo "==> WhisperX стартует"
echo "    MODEL=$MODEL, LANG=$LANGUAGE, DEVICE=$DEVICE, TYPE=$COMPUTE_TYPE"
echo "    VAD=$VAD, speakers $MIN_SPK..$MAX_SPK, FORMAT=$FORMAT"
mkdir -p "$OUT_DIR" "$CONV_DIR"

WHISPERX_BIN=$(command -v whisperx || echo "python3 -m whisperx")

# Шаг 1: Конвертация файлов в WAV 16kHz mono
found=0
for ext in $EXTS; do
  for f in "$IN_DIR"/*."$ext"; do
    [ -e "$f" ] || continue
    found=1
    base=$(basename "$f" ."$ext")
    conv_file="$CONV_DIR/$base.wav"
    
    # Проверяем, нужно ли конвертировать (если уже WAV 16kHz mono)
    if [ "$ext" = "wav" ] && ffmpeg -i "$f" 2>&1 | grep -q "Audio: pcm_s16le, 16000 Hz, mono"; then
      echo "==> Файл уже в оптимальном формате: $f"
      ln -s "$f" "$conv_file"  # symlink вместо cp, чтобы не копировать в ro
      continue
    fi
    
    echo "==> Конвертирую в WAV 16kHz mono с нормализацией шума: $f"
    ffmpeg -i "$f" -ar 16000 -ac 1 -c:a pcm_s16le -af loudnorm=I=-16:TP=-1.5:LRA=11 "$conv_file" -y
  done
done

[ "$found" -eq 1 ] || { echo "Нет аудиофайлов в $IN_DIR"; exit 1; }

# Шаг 2: Обработка конвертированных WAV
for f in "$CONV_DIR"/*.wav; do
  [ -e "$f" ] || continue
  echo "==> Обрабатываю: $f"
  $WHISPERX_BIN \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --device "$DEVICE" \
    --compute_type "$COMPUTE_TYPE" \
    $DECODER_OPTS \
    --temperature "$TEMPERATURE" \
    --temperature_increment_on_fallback "$TEMP_INC" \
    --vad_method "$VAD" \
    --vad_onset "$VAD_ONSET" \
    --vad_offset "$VAD_OFFSET" \
    --batch_size "$BATCH_SIZE" \
    --diarize --min_speakers "$MIN_SPK" --max_speakers "$MAX_SPK" \
    ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
    --output_dir "$OUT_DIR" --output_format "$FORMAT" \
    "$f"
done

echo "Готово."