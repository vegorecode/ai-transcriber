import logging
import os
import tempfile
from collections import defaultdict

import librosa
import numpy as np
import requests
import torch
from pyannote.audio import Inference
from pyannote.core import SlidingWindowFeature
from scipy.spatial.distance import cosine
from speechbrain.pretrained import EncoderClassifier

# -----------------------------------------------------------------
# Ленивая загрузка модели эмбеддингов pyannote (1 раз на процесс)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_EMBED_MODEL = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка переменных окружения (HF_TOKEN) при наличии .env
from dotenv import find_dotenv, load_dotenv

# Подтянуть переменные окружения из .env, если есть
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")

ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device},
)


def spk_embed(wave_16k_mono: np.ndarray) -> np.ndarray:
    """Вернуть 192‑мерный эмбеддинг для моно звука 16 кГц."""
    wav = torch.tensor(wave_16k_mono).unsqueeze(0).to(device)
    return ecapa.encode_batch(wav).squeeze(0).cpu().numpy()


# -----------------------------------------------------------------
#  Select GPU when available, otherwise fall back to CPU once
# ------------------------------------------------------------------
#
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Voice Embedding Model


# ------------------------------------------------------------------
# Хелпер: формирует словарь для Inference(pyannote 3.x)
def to_pyannote_dict(wf, sr=16000):
    """Вернуть словарь в формате, понятном Inference 3.x."""
    if isinstance(wf, np.ndarray):
        wf = torch.tensor(wf, dtype=torch.float32)
    if wf.ndim == 1:  # (time,)  →  (1, time)
        wf = wf.unsqueeze(0)
    return {"waveform": wf, "sample_rate": sr}


# ------------------------------------------------------------------
def to_numpy(arr) -> np.ndarray:
    """Return a 1-D numpy embedding whatever pyannote gives back."""
    if isinstance(arr, np.ndarray):  # already good
        return arr.flatten()
    if torch.is_tensor(arr):  # old style (should not happen)
        return arr.detach().cpu().numpy().flatten()
    # SlidingWindowFeature → .data is an np.ndarray
    if isinstance(arr, SlidingWindowFeature):
        return arr.data.flatten()
    raise TypeError(f"Unsupported embedding type: {type(arr)}")


# Логирование (однократная настройка хендлеров)
logger = logging.getLogger("speaker_processing")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Only add handlers if none exist (to avoid duplicates)
    import sys

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Глобальный кэш эмбеддингов спикеров
_SPEAKER_EMBEDDING_CACHE = {}


# ---------------------------------------------------------------------
# Универсальный helper: приводит разные типы к 1‑D numpy
# ---------------------------------------------------------------------
def _to_numpy_flat(emb):
    """Вернуть 1‑D numpy для Tensor / SlidingWindowFeature или объекта с .data."""
    import numpy as np
    import torch

    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().numpy().flatten()

    if isinstance(emb, SlidingWindowFeature):
        return emb.data.flatten()

    # generic fallback: has `.data`?
    data = getattr(emb, "data", None)
    if isinstance(data, np.ndarray):
        return data.flatten()

    raise TypeError(f"Unsupported embedding type: {type(emb)}")


def load_known_speakers_from_samples(speaker_samples, huggingface_access_token=None):
    """Скачать при необходимости и построить эмбеддинги для образцов голосов.

    Возвращает словарь: имя → np.ndarray (L2-нормированный эмбеддинг)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = Inference(
            "pyannote/embedding", use_auth_token=huggingface_access_token, device=device
        )
        logger.debug("Successfully loaded pyannote embedding model")
    except Exception as e:
        logger.error(f"Failed to load pyannote embedding model: {e}", exc_info=True)
        return {}

    global _SPEAKER_EMBEDDING_CACHE
    known_embeddings: dict[str, np.ndarray] = {}

    for sample in speaker_samples:
        name = sample.get("name")
        url = sample.get("url")

        if not name:
            if url:
                name = os.path.splitext(os.path.basename(url))[0]
                logger.debug(f"No name provided; using '{name}' from URL.")
            else:
                logger.error(f"Skipping sample with missing name and URL: {sample}")
                continue

        # cache first
        if name in _SPEAKER_EMBEDDING_CACHE:
            known_embeddings[name] = _SPEAKER_EMBEDDING_CACHE[name]
            continue

        # resolve filepath
        filepath = None
        if sample.get("file_path"):
            filepath = sample["file_path"]
            logger.debug(f"Loading speaker sample '{name}' from local file: {filepath}")
        elif url:
            try:
                logger.debug(f"Downloading speaker sample '{name}' from URL: {url}")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                suffix = os.path.splitext(url)[1] or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(response.content)
                    tmp.flush()
                    filepath = tmp.name
                    logger.debug(
                        f"Downloaded sample '{name}' to temporary file: {filepath}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to download speaker sample '{name}' from {url}: {e}",
                    exc_info=True,
                )
                filepath = None
        else:
            logger.error(f"Skipping sample '{name}': no file_path or URL provided.")
            filepath = None

        if not filepath:
            continue

        # compute embedding
        try:
            waveform, sr = librosa.load(filepath, sr=16000, mono=True)
            emb = model(to_pyannote_dict(waveform, sr))
            if hasattr(emb, "data"):
                emb_np = np.mean(emb.data, axis=0)
            elif torch.is_tensor(emb):
                emb_np = emb.detach().cpu().numpy()
            else:
                emb_np = np.asarray(emb)
            emb_np = emb_np / np.linalg.norm(emb_np)
            _SPEAKER_EMBEDDING_CACHE[name] = emb_np
            known_embeddings[name] = emb_np
            logger.debug(
                f"Computed embedding for '{name}' (norm={np.linalg.norm(emb_np):.2f})."
            )
        except Exception as e:
            logger.error(
                f"Failed to process speaker sample '{name}' from file {filepath}: {e}",
                exc_info=True,
            )
        finally:
            # remove temp file if downloaded
            if (
                not sample.get("file_path")
                and url
                and filepath
                and os.path.exists(filepath)
            ):
                try:
                    os.remove(filepath)
                    logger.debug(f"Removed temporary file for '{name}': {filepath}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {filepath}: {e}")

    return known_embeddings


def identify_speaker(segment_embedding, known_embeddings, threshold=0.1):
    import numpy as np

    # Убедимся, что вход — 1‑D numpy
    if isinstance(segment_embedding, np.ndarray):
        segment_embedding = segment_embedding.ravel()
    else:
        logger.error("Invalid segment_embedding type, expected numpy.ndarray")
        return "Unknown", -1

    best_match, best_similarity = "Unknown", -1.0
    for speaker, known_emb in known_embeddings.items():
        if not isinstance(known_emb, np.ndarray):
            continue
        known_emb_flat = known_emb.ravel()
        # cosine ожидает 1‑D
        score = 1 - cosine(segment_embedding, known_emb_flat)
        if score > best_similarity:
            best_similarity, best_match = score, speaker

    return (
        (best_match, best_similarity)
        if best_similarity >= threshold
        else ("Unknown", best_similarity)
    )
    """Сравнение эмбеддинга сегмента с известными спикерами (косинус)."""


from datetime import datetime

import numpy as np
import torch


def process_diarized_output(
    output: dict,
    audio_filepath: str,
    known_embeddings: dict,
    huggingface_access_token: str | None = None,
    return_logs: bool = True,
    threshold: float = 0.20,
) -> tuple[dict, dict | None]:
    """Эмбеддинг сегментов, центроиды по кластерам, сопоставление и релейблинг."""

    log_data = {
        "segments": [],
        "centroids": {},
        "relabeling_decisions": [],
        "timestamp": datetime.now().isoformat(),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Inference(
        "pyannote/embedding", use_auth_token=huggingface_access_token, device=device
    )

    segments = output.get("segments", [])
    if not segments:
        return output, None

    # 1) Эмбеддинг каждого диаризованного сегмента
    for seg in segments:
        seg.setdefault("speaker", "Unknown")
        start, end = seg["start"], seg["end"]
        try:
            wav, _ = librosa.load(
                audio_filepath, sr=16000, mono=True, offset=start, duration=end - start
            )
        except Exception as e:
            logger.error(f"Could not load [{start:.2f}-{end:.2f}]: {e}", exc_info=True)
            continue
        if wav.size == 0:
            continue

        emb = embedder({"waveform": torch.tensor(wav)[None], "sample_rate": 16000})
        emb = _to_numpy_flat(emb)
        emb /= np.linalg.norm(emb)
        seg["__embed__"] = emb

        log_data["segments"].append(
            {
                "start": start,
                "end": end,
                "original_speaker": seg["speaker"],
                "embedding": emb.tolist(),
            }
        )

    # 2) Центроиды кластеров
    # clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    # for seg in segments:
    #     clusters[seg["speaker"]].append(seg["__embed__"])

    # centroids = {
    #     lbl: np.mean(mats, axis=0) / np.linalg.norm(np.mean(mats, axis=0))
    #     for lbl, mats in clusters.items() if mats
    # }
    # def process_diarized_output(

    # 2) build cluster centroids
    clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    for seg in segments:
        clusters[seg["speaker"]].append(seg["__embed__"])

    centroids = {
        lbl: np.mean(mats, axis=0) / np.linalg.norm(np.mean(mats, axis=0))
        for lbl, mats in clusters.items()
        if mats
    }
    # 2) build cluster centroids (only on uniform‑length embeddings)
    clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    for seg in segments:
        emb = seg.get("__embed__")
        if isinstance(emb, np.ndarray) and emb.ndim == 1:
            clusters[seg["speaker"]].append(emb)

    centroids: dict[str, np.ndarray] = {}
    for lbl, mats in clusters.items():
        # ensure we have at least one embedding
        if not mats:
            continue
        # check all embeddings have the same dimension
        dims = {emb.shape[0] for emb in mats}
        if len(dims) != 1:
            logger.warning(
                f"Inconsistent embedding dims for '{lbl}': {dims}, skipping centroid."
            )
            continue
        mat_stack = np.vstack(mats)  # shape (n_segments, dim)
        mean_emb = mat_stack.mean(axis=0)  # shape (dim,)
        centroid = mean_emb / np.linalg.norm(mean_emb)
        centroids[lbl] = centroid

    # record centroids in log_data as lists
    for lbl, centroid in centroids.items():
        log_data["centroids"][lbl] = centroid.tolist()

    # 3) Релейблинг сегментов на основе центроидов
    for lbl, centroid in centroids.items():
        name, score = identify_speaker(centroid, known_embeddings, threshold=threshold)
        decision = {
            "original_label": lbl,
            "new_label": name,
            "similarity_score": float(score),
            "threshold": threshold,
            "relabel": name != "Unknown",
        }
        log_data["relabeling_decisions"].append(decision)

        if name == "Unknown":
            continue

        for seg in segments:
            if seg["speaker"] == lbl:
                seg["speaker"] = name
                seg["similarity"] = float(score)

    # 4) Приведение типов и очистка временных полей
    for seg in segments:
        # seg.pop("__embed__", None)
        seg["start"] = float(seg["start"])
        seg["end"] = float(seg["end"])
        seg.setdefault("similarity", None)

    if return_logs:
        return output, log_data
    else:
        return output, log_data


## Альтернативные утилиты


import numpy as np
import torch
from scipy.spatial.distance import cdist


def _get_embed_model(hf_token: str | None = None):
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    token = hf_token or os.getenv("HF_TOKEN")
    try:
        _EMBED_MODEL = Inference(
            "pyannote/embedding", device=DEVICE, use_auth_token=token
        )
        return _EMBED_MODEL
    except Exception as e:
        logger.error(
            "Failed to initialize pyannote embedding model: %s", e, exc_info=True
        )
        raise


def embed_waveform(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """512‑мерная L2‑нормированная эмбеддинга для волновой формы."""
    model = _get_embed_model()
    feat = model({"waveform": torch.tensor(wav).unsqueeze(0), "sample_rate": sr})
    if hasattr(feat, "data"):
        arr = feat.data.mean(axis=0)
    else:
        arr = feat.squeeze(0).cpu().numpy()
    arr = arr.astype(np.float32)
    return arr / np.linalg.norm(arr)


def enroll_profiles(profiles: list[dict]) -> dict[str, np.ndarray]:
    """Запись профилей спикеров по локальным файлам: имя → 512‑мерный вектор."""
    embeddings = {}
    for p in profiles:
        wav, sr = librosa.load(p["file_path"], sr=16000, mono=True)
        embeddings[p["name"]] = embed_waveform(wav, sr)
    return embeddings


def identify_speakers_on_segments(
    segments: list[dict],
    audio_path: str,
    enrolled: dict[str, np.ndarray],
    threshold: float = 0.1,
) -> list[dict]:
    """Определить спикеров на сегментах на основе ранее записанных профилей."""
    names = list(enrolled.keys())
    mat = np.stack([enrolled[n] for n in names])

    for seg in segments:
        wav, sr = librosa.load(
            audio_path,
            sr=16000,
            mono=True,
            offset=seg["start"],
            duration=seg["end"] - seg["start"],
        )
        emb = embed_waveform(wav, sr)
        sims = 1 - cdist(emb[None, :], mat, metric="cosine")[0]
        best = sims.argmax()
        if sims[best] >= threshold:
            seg["speaker_id"] = names[best]
            seg["similarity"] = float(sims[best])
        else:
            seg["speaker_id"] = "Unknown"
            seg["similarity"] = float(sims.max())
    return segments


def relabel_speakers_by_avg_similarity(segments: list[dict]) -> list[dict]:
    """Релейблинг: для каждой диаризационной метки выбрать спикера с наибольшей средней схожестью."""
    # Step 1: collect all similarities per diarized label
    grouped = defaultdict(list)
    for seg in segments:
        spk = seg.get("speaker")
        sim = seg.get("similarity")
        sid = seg.get("speaker_id")
        if spk and sim is not None and sid:
            grouped[spk].append((sid, sim))

    # Step 2: compute average similarity for each speaker_id within each group
    relabel_map = {}
    for orig_spk, samples in grouped.items():
        scores = defaultdict(list)
        for sid, sim in samples:
            scores[sid].append(sim)
        avg = {sid: sum(vals) / len(vals) for sid, vals in scores.items()}
        best_match = max(avg, key=avg.get)
        relabel_map[orig_spk] = best_match

    # Step 3: apply relabeling
    for seg in segments:
        spk = seg.get("speaker")
        if spk in relabel_map:
            seg["speaker"] = relabel_map[spk]

    return segments
