"""Run-record persistence: one JSON file per audition under the plugin data dir."""

from __future__ import annotations

import json
import secrets
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

KEEP_RUNS = 20
_SUFFIX_ALPHABET = string.ascii_lowercase + string.digits


def new_run_id(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(timezone.utc)
    suffix = "".join(secrets.choice(_SUFFIX_ALPHABET) for _ in range(4))
    return f"us_{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"


def default_runs_dir() -> Path:
    """``<hermes home>/plugin-data/understudy/runs`` — imported lazily so engine stays Hermes-free."""
    from plugins.plugin_storage import plugin_data_dir
    return plugin_data_dir("understudy") / "runs"


class RunStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else default_runs_dir()

    def _path(self, run_id: str) -> Path:
        safe = "".join(c for c in str(run_id) if c.isalnum() or c in "_-")
        return self.root / f"{safe}.json"

    def _sidecars(self, run_id: str) -> list[Path]:
        """PNG/HTML artifacts next to the record file."""
        return [self._path(run_id).with_suffix(ext) for ext in (".png", ".html")]

    def save(self, record: dict) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(record.get("id") or new_run_id())
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, run_id: Optional[str] = None) -> Optional[dict]:
        if run_id:
            path = self._path(run_id)
            if not path.is_file():
                matches = sorted(self.root.glob(f"{run_id}*.json")) if self.root.is_dir() else []
                path = matches[0] if matches else path
            if not path.is_file():
                return None
        else:
            files = self._files()
            if not files:
                return None
            path = files[0]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(self.root.glob("us_*.json"), key=lambda p: p.name, reverse=True)

    def list_runs(self, limit: int = KEEP_RUNS) -> list[dict]:
        """Newest-first stubs: id, created_at, status, candidates, spend."""
        out = []
        for path in self._files()[: max(0, int(limit))]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            out.append({
                "id": data.get("id") or path.stem,
                "created_at": data.get("created_at"),
                "status": data.get("status"),
                "candidates": [c.get("spec") for c in data.get("candidates") or [] if isinstance(c, dict)],
                "spent_usd": data.get("spent_usd"),
                "current_model": data.get("current_model"),
            })
        return out

    def prune(self, keep: int = KEEP_RUNS) -> int:
        """Drop all but the newest ``keep`` run files; returns the removed count."""
        files = self._files()
        removed = 0
        for path in files[max(0, int(keep)):]:
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
            for sidecar in self._sidecars(path.stem):
                try:
                    sidecar.unlink()
                except OSError:
                    pass
        return removed


def default_store() -> RunStore:
    return RunStore(default_runs_dir())
