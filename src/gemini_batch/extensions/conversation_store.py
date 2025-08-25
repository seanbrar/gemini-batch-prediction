from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .conversation import CacheBinding, ConversationState, Exchange

if TYPE_CHECKING:
    import os


class ConversationStore(Protocol):
    async def load(self, conversation_id: str) -> ConversationState: ...

    async def append(
        self, conversation_id: str, expected_version: int, ex: Exchange
    ) -> ConversationState: ...


class JSONStore:
    """Append-only, versioned JSON store (single file mapping id -> state).

    Uses copy-on-write: write to a temp file and rename for atomicity.
    Shape saved per conversation id:
      {
        "sources": [...],
        "turns": [{"user":..., "assistant":..., "error":..., audit...}, ...],
        "cache": {"key":..., "artifacts": [...], "ttl_seconds": ...} | null,
        "version": int
      }
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)

    async def load(self, conversation_id: str) -> ConversationState:
        data = self._read_all()
        entry = data.get(conversation_id)
        if not isinstance(entry, dict):
            return ConversationState(sources=(), turns=(), cache=None, version=0)
        sources_raw = entry.get("sources", ())
        sources: tuple[str, ...] = (
            tuple(sources_raw) if isinstance(sources_raw, list | tuple) else ()
        )
        turns_raw = entry.get("turns", [])
        turns: list[Exchange] = []
        if isinstance(turns_raw, list):
            for t in turns_raw:
                if not isinstance(t, dict):
                    continue
                turns.append(
                    Exchange(
                        user=str(t.get("user", "")),
                        assistant=str(t.get("assistant", "")),
                        error=bool(t.get("error", False)),
                        estimate_min=t.get("estimate_min"),
                        estimate_max=t.get("estimate_max"),
                        actual_tokens=t.get("actual_tokens"),
                        in_range=t.get("in_range"),
                    )
                )
        cache_raw = entry.get("cache")
        cache = None
        if isinstance(cache_raw, dict) and cache_raw:
            artifacts = tuple(cache_raw.get("artifacts", ()) or ())
            cache = CacheBinding(
                key=str(cache_raw.get("key")),
                artifacts=artifacts,
                ttl_seconds=cache_raw.get("ttl_seconds"),
            )
        version_raw = entry.get("version", 0)
        version = int(version_raw) if isinstance(version_raw, int | float | str) else 0
        return ConversationState(
            sources=sources, turns=tuple(turns), cache=cache, version=version
        )

    async def append(
        self, conversation_id: str, expected_version: int, ex: Exchange
    ) -> ConversationState:
        data = self._read_all()
        entry = data.get(conversation_id)
        if not isinstance(entry, dict):
            entry = {
                "sources": [],
                "turns": [],
                "cache": None,
                "version": 0,
            }
        current_version = entry.get("version", 0)
        if not isinstance(current_version, int):
            current_version = 0
        if current_version != expected_version:
            raise RuntimeError(
                f"OCC conflict: expected {expected_version}, got {current_version}"
            )
        # Append new exchange and bump version
        turns_raw = entry.get("turns", [])
        turns = list(turns_raw) if isinstance(turns_raw, list) else []
        turns.append(_exchange_to_dict(ex))
        entry["turns"] = turns
        entry["version"] = current_version + 1
        data[conversation_id] = entry
        self._write_all(data)
        # Return reconstructed state
        return await self.load(conversation_id)

    def _read_all(self) -> dict[str, dict[str, object]]:
        if not self._path.exists():
            return {}
        try:
            result = json.loads(self._path.read_text(encoding="utf-8"))
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def _write_all(self, data: dict[str, dict[str, object]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        Path.replace(tmp, self._path)


def _exchange_to_dict(ex: Exchange) -> dict[str, object]:
    return asdict(ex)
    # asdict includes our fields already; ensure tuple types serialized
