"""Lightweight in-memory registries for uploads and caches.

These registries are owned by the executor/runtime and are deliberately simple.
They avoid provider SDK calls; they only store/retrieve previously discovered
identifiers so handlers can reuse them. They are intentionally ephemeral and
process-local; persistence across processes is out of scope here.
"""

from __future__ import annotations

from typing import Any, Protocol


class CacheRegistry:
    """Maps deterministic cache keys to provider cache names.

    Minimal API by design; single-process memory only. Concurrency protection
    is expected to be handled by higher layers via single-flight if needed.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory mapping for cache keys.

        Creates a process-local store that maps deterministic cache keys to
        provider cache names. No I/O or network calls occur here.
        """
        self._key_to_name: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        """Return the provider cache name for `key`, if present.

        Args:
            key: Deterministic key produced by the planner.

        Returns:
            The provider cache name or None.
        """
        return self._key_to_name.get(key)

    def set(self, key: str, name: str) -> None:
        """Associate a deterministic key with a provider cache name."""
        self._key_to_name[key] = name


class FileRegistry:
    """Maps local file identifiers (paths or hashes) to uploaded provider refs.

    Values are provider-shaped objects or neutral `FileRefPart` that the
    handler will coerce to a `FileRefPart` if needed.
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory mapping for file uploads.

        Maintains a process-local map from local file identifiers (e.g., paths
        or hashes) to provider-uploaded references for reuse within a run.
        """
        self._id_to_uploaded: dict[str, Any] = {}

    def get(self, local_id: str) -> Any | None:
        """Return the uploaded reference for a local file id, if present."""
        return self._id_to_uploaded.get(local_id)

    def set(self, local_id: str, uploaded: Any) -> None:
        """Associate a local file id with a provider-uploaded reference."""
        self._id_to_uploaded[local_id] = uploaded


class SimpleRegistry(Protocol):
    """Minimal get/set registry protocol used by pipeline handlers.

    This protocol exists to clarify the expected surface used by the handlers
    and to remove incidental ``hasattr`` checks. Implementations are simple,
    in-memory registries with no I/O.
    """

    def get(self, key: str) -> Any | None:
        """Return the value for `key`, if present."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Associate a key with a value."""
        ...
