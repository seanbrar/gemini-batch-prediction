from dataclasses import dataclass  # noqa: D100
from enum import Enum, auto

from google.genai import types


@dataclass(frozen=True)
class PartsPayload:  # noqa: D101
    parts: list[types.Part]


@dataclass(frozen=True)
class ExplicitCachePayload:  # noqa: D101
    cache_name: str
    parts: list[types.Part]


class CacheStrategy(Enum):  # noqa: D101
    GENERATE_RAW = auto()
    GENERATE_WITH_OPTIMIZED_PARTS = auto()
    GENERATE_FROM_EXPLICIT_CACHE = auto()


@dataclass(frozen=True)
class CacheAction:  # noqa: D101
    strategy: CacheStrategy
    payload: PartsPayload | ExplicitCachePayload

    def __post_init__(self):  # noqa: D105
        # Validate payload matches strategy
        if self.strategy == CacheStrategy.GENERATE_FROM_EXPLICIT_CACHE:
            if not isinstance(self.payload, ExplicitCachePayload):
                raise ValueError(
                    f"Strategy {self.strategy} requires ExplicitCachePayload"
                )
        else:
            if not isinstance(self.payload, PartsPayload):
                raise ValueError(f"Strategy {self.strategy} requires PartsPayload")
