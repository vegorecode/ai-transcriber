from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class HistoryStore:
    """Простое потокобезопасное хранилище истории транскрипций в JSON-файле.

    Формат записи (словарь):
      - id: str
      - original_filename: str
      - status: str (queued|processing|finished|failed)
      - created_at: str (ISO)
      - updated_at: str (ISO)
      - output_format: str
      - output_path: Optional[str]
      - download_name: Optional[str]
      - config_path: Optional[str]
      - config_download_name: Optional[str]
      - error: Optional[str]
    """

    def __init__(self, config_dir: Path | str, filename: str = "history.json") -> None:
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.config_dir / filename
        self._lock = threading.Lock()

        if not self.file_path.exists():
            self._write_json([])

    def _read_json(self) -> List[Dict[str, Any]]:
        try:
            raw = self.file_path.read_text(encoding="utf-8")
            if not raw.strip():
                return []
            return json.loads(raw)
        except Exception:
            return []

    def _write_json(self, items: List[Dict[str, Any]]) -> None:
        data = json.dumps(items, ensure_ascii=False, indent=2)
        tmp = self.file_path.with_suffix(".tmp")
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(self.file_path)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            items = self._read_json()
        # Сортируем по времени создания по убыванию
        return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            items = self._read_json()
        for it in items:
            if it.get("id") == job_id:
                return it
        return None

    def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            items = self._read_json()
            items.append(record)
            self._write_json(items)
        return record

    def update(self, job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            items = self._read_json()
            found = None
            for idx, it in enumerate(items):
                if it.get("id") == job_id:
                    it.update(updates)
                    it["updated_at"] = updates.get("updated_at", _utc_now_iso())
                    items[idx] = it
                    found = it
                    break
            if found is not None:
                self._write_json(items)
            return found

    def delete(self, job_id: str) -> bool:
        """Удаляет запись из истории. Возвращает True, если запись найдена и удалена."""
        with self._lock:
            items = self._read_json()
            new_items = [it for it in items if it.get("id") != job_id]
            changed = len(new_items) != len(items)
            if changed:
                self._write_json(new_items)
            return changed

    def create_new(
        self,
        job_id: str,
        original_filename: str,
        output_format: str,
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        record = {
            "id": job_id,
            "original_filename": original_filename,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "output_format": output_format,
            "output_path": None,
            "download_name": None,
            "config_path": None,
            "config_download_name": None,
            "error": None,
        }
        return self.append(record)
