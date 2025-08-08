from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .history import HistoryStore
from .settings import AppSettings, SettingsStore
from .whisper import ensure_dir, transcribe_file

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(os.getenv("WORK_DIR", "/work"))
UPLOADS_DIR = WORK_DIR / "uploads"
OUT_DIR = WORK_DIR / "out"
CONFIG_DIR = WORK_DIR / "config"
TMP_DIR = WORK_DIR / "tmp"


app = FastAPI(title="WhisperX API", version="1.0.0")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@app.on_event("startup")
def on_startup() -> None:
    ensure_dir(UPLOADS_DIR)
    ensure_dir(OUT_DIR)
    ensure_dir(CONFIG_DIR)
    ensure_dir(TMP_DIR)
    # Если файла настроек нет — создадим его на основе текущих env-дефолтов
    store = SettingsStore(CONFIG_DIR)
    settings_path = CONFIG_DIR / "settings.json"
    if not settings_path.exists():
        store.save(AppSettings())


# Глобальное хранилище истории
history = HistoryStore(CONFIG_DIR)


def _process_job(
    job_id: str,
    tmp_path: Path,
    original_filename: str,
    settings: AppSettings,
    hf_token: Optional[str],
) -> None:
    try:
        history.update(job_id, {"status": "processing"})
        # Сохранить снэпшот настроек для этой задачи
        config_snapshot_dir = CONFIG_DIR / "jobs"
        ensure_dir(config_snapshot_dir)
        config_path = config_snapshot_dir / f"{job_id}.settings.json"
        config_json = json.dumps(settings.model_dump(), ensure_ascii=False, indent=2)
        config_path.write_text(config_json, encoding="utf-8")
        result_path, base_id = transcribe_file(
            tmp_path,
            UPLOADS_DIR,
            OUT_DIR,
            settings,
            hf_token,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

        download_name = f"{Path(original_filename).stem}.{settings.output_format}"
        # Сохраняем конфиг рядом с транскриптом как TXT с читаемым содержимым
        config_txt_path = OUT_DIR / f"{base_id}_config.txt"
        config_txt_path.write_text(
            json.dumps(settings.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        history.update(
            job_id,
            {
                "status": "finished",
                "output_path": str(result_path),
                "download_name": download_name,
                "config_path": str(config_txt_path),
                "job_config_path": str(config_path),
                "config_download_name": f"{Path(original_filename).stem}_config.txt",
            },
        )
    except Exception as exc:  # noqa: BLE001
        history.update(job_id, {"status": "failed", "error": str(exc)})
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    store = SettingsStore(CONFIG_DIR)
    settings = store.load()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "settings": settings},
    )


@app.post("/api/transcribe")
async def api_transcribe(
    request: Request,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не передан")

    store = SettingsStore(CONFIG_DIR)
    settings = store.load()
    hf_token = os.getenv("HF_TOKEN") or None

    # Создадим job и сохраним временный файл для фоновой обработки
    job_id = uuid.uuid4().hex
    history.create_new(job_id, file.filename, settings.output_format)

    suffix = Path(file.filename).suffix
    tmp_path = TMP_DIR / f"{job_id}{suffix}"
    data = await file.read()
    tmp_path.write_bytes(data)

    t = threading.Thread(
        target=_process_job,
        args=(job_id, tmp_path, file.filename, settings, hf_token),
        daemon=True,
    )
    t.start()

    return JSONResponse({"id": job_id, "status": "queued"})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    store = SettingsStore(CONFIG_DIR)
    settings = store.load()
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "settings": settings},
    )


@app.post("/settings")
async def update_settings(
    request: Request,
    language: str = Form(...),
    model: str = Form(...),
    device: str = Form(...),
    compute_type: str = Form(...),
    batch_size: int = Form(...),
    beam_size: int = Form(...),
    temperature: float = Form(...),
    temperature_increment_on_fallback: float = Form(...),
    vad_method: str = Form(...),
    vad_onset: float = Form(...),
    vad_offset: float = Form(...),
    diarize: bool = Form(False),
    min_speakers: int = Form(...),
    max_speakers: int = Form(...),
    output_format: str = Form(...),
    length_penalty: float = Form(...),
):
    store = SettingsStore(CONFIG_DIR)
    updates = {
        "language": language,
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "beam_size": beam_size,
        "temperature": temperature,
        "temperature_increment_on_fallback": temperature_increment_on_fallback,
        "vad_method": vad_method,
        "vad_onset": vad_onset,
        "vad_offset": vad_offset,
        "diarize": diarize,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "output_format": output_format,
        "length_penalty": length_penalty,
    }

    try:
        store.update_from_dict(updates)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RedirectResponse(url="/settings", status_code=303)


@app.get("/api/settings")
async def get_settings_json():
    store = SettingsStore(CONFIG_DIR)
    return store.load().model_dump()


@app.post("/api/settings")
async def update_settings_json(payload: dict):
    store = SettingsStore(CONFIG_DIR)
    try:
        new_settings = store.update_from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return new_settings.model_dump()


@app.get("/api/jobs")
async def list_jobs():
    return history.list()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    item = history.get(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return item


@app.get("/download/{job_id}")
async def download_job(job_id: str):
    item = history.get(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    if item.get("status") != "finished" or not item.get("output_path"):
        raise HTTPException(status_code=409, detail="Файл ещё не готов")
    path = Path(item["output_path"]).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл отсутствует")
    filename = item.get("download_name") or path.name
    return FileResponse(
        path=str(path), media_type="application/octet-stream", filename=filename
    )


@app.get("/download-config/{job_id}")
async def download_config(job_id: str):
    item = history.get(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    path_str = item.get("config_path")
    if not path_str:
        raise HTTPException(status_code=404, detail="Конфиг не найден")
    path = Path(path_str).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл отсутствует")
    filename = item.get("config_download_name") or path.name
    return FileResponse(
        path=str(path), media_type="application/json", filename=filename
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    item = history.get(job_id)
    if not item:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    # Удаляем файлы результата и конфига, если есть
    result_path_str = item.get("output_path")
    if result_path_str:
        try:
            Path(result_path_str).unlink(missing_ok=True)
        except Exception:
            pass

    config_path_str = item.get("config_path")
    if config_path_str:
        try:
            Path(config_path_str).unlink(missing_ok=True)
        except Exception:
            pass

    job_config_path_str = item.get("job_config_path")
    if job_config_path_str:
        try:
            Path(job_config_path_str).unlink(missing_ok=True)
        except Exception:
            pass

    # Удаляем запись из истории
    history.delete(job_id)
    return JSONResponse({"ok": True})


@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs(request: Request):
    return templates.TemplateResponse("api_docs.html", {"request": request})
