"""Per-client usage statistics tracker with optional disk persistence."""

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class ClientStats:
    request_count: int = 0
    error_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    last_request_time: Optional[str] = None
    models_used: Counter = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "last_request_time": self.last_request_time,
            "models_used": dict(self.models_used),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientStats":
        return cls(
            request_count=data.get("request_count", 0),
            error_count=data.get("error_count", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            last_request_time=data.get("last_request_time"),
            models_used=Counter(data.get("models_used", {})),
        )


class UsageStats:
    def __init__(self, persist_path: Optional[str] = None, save_every: int = 100) -> None:
        self._clients: Dict[str, ClientStats] = {}
        self._persist_path: Optional[Path] = Path(persist_path) if persist_path else None
        self._save_every: int = max(1, save_every)
        self._dirty_count: int = 0
        if self._persist_path:
            self._load()

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for name, client_data in data.get("clients", {}).items():
                self._clients[name] = ClientStats.from_dict(client_data)
            logger.info(f"Loaded usage stats for {len(self._clients)} client(s) from {self._persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load usage stats from {self._persist_path}: {e}")

    def _save(self, force: bool = False) -> None:
        if not self._persist_path:
            return
        if not force and self._dirty_count < self._save_every:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"clients": {name: s.to_dict() for name, s in self._clients.items()}}
            self._persist_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            self._dirty_count = 0
        except Exception as e:
            logger.warning(f"Failed to save usage stats to {self._persist_path}: {e}")

    def flush(self) -> None:
        """Force save to disk (e.g. on shutdown)."""
        self._save(force=True)

    def record_request(
        self,
        client_name: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
    ) -> None:
        stats = self._clients.setdefault(client_name, ClientStats())
        stats.request_count += 1
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens
        stats.last_request_time = datetime.now(timezone.utc).isoformat()
        stats.models_used[model] += 1
        if not success:
            stats.error_count += 1
        self._dirty_count += 1
        self._save()

    def get_stats(self, client_name: Optional[str] = None) -> Dict[str, Any]:
        if client_name:
            stats = self._clients.get(client_name)
            if stats:
                return {"clients": {client_name: stats.to_dict()}}
            return {"clients": {}}
        return {"clients": {name: s.to_dict() for name, s in self._clients.items()}}

    def reset_stats(self, client_name: Optional[str] = None) -> None:
        if client_name:
            self._clients.pop(client_name, None)
        else:
            self._clients.clear()
        self._save(force=True)
