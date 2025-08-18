# Профили спикеров: загрузка и сопоставление
import os
import tempfile

import librosa
import numpy as np
import requests
import torch
from pyannote.audio import Inference
from scipy.spatial.distance import cdist

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EMBED = Inference(
    "pyannote/embedding", device=_DEVICE, use_auth_token=os.getenv("HF_TOKEN")
)

_CACHE = {}  # name → 512‑D vector

# ---------------------------------------------------------------------
# 1)  Скачать аудио профиля (один раз)  → 512‑D эмбеддинг  → кэш
# ---------------------------------------------------------------------


def _l2(x: np.ndarray) -> np.ndarray:  # нормализация L2
    return x / np.linalg.norm(x)


def load_embeddings(profiles):
    """Скачать профили вида {name,url} и вернуть {'name': np.array(512)}."""
    out = {}
    for p in profiles:
        name, url = p["name"], p["url"]
        if name in _CACHE:
            out[name] = _CACHE[name]
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(requests.get(url, timeout=30).content)
            tmp.flush()
            wav, _ = librosa.load(tmp.name, sr=16_000, mono=True)
            vec = _EMBED(torch.tensor(wav).unsqueeze(0)).cpu().numpy().flatten()
            vec = _l2(vec)
            _CACHE[name] = vec
            out[name] = vec
    return out


# ---------------------------------------------------------------------
# 2)  Заменить диаризационные метки на имя ближайшего профиля
# ---------------------------------------------------------------------
def relabel(diarize_df, transcription, embeds, threshold=0.75):
    """Заменить SPEAKER_XX на имя профиля при схожести выше порога."""
    names = list(embeds.keys())
    vecstack = np.stack(list(embeds.values()))  # (N,128)

    for seg in transcription["segments"]:
        dia_spk = seg.get("speaker")  # e.g. SPEAKER_00
        if not dia_spk:
            continue

        # Приблизительная эмбеддинга сегмента: среднее по словам
        word_vecs = [
            w.get("embedding")
            for w in seg.get("words", [])
            if w.get("speaker") == dia_spk and w.get("embedding") is not None
        ]

        if not word_vecs:
            continue

        centroid = np.mean(word_vecs, axis=0, keepdims=True)  # (1,128)
        sim = 1 - cdist(centroid, vecstack, metric="cosine")
        best_idx = int(sim.argmax())
        if sim[0, best_idx] >= threshold:
            real = names[best_idx]
            seg["speaker"] = real
            seg["similarity"] = float(sim[0, best_idx])
            for w in seg.get("words", []):
                w["speaker"] = real
    return transcription
