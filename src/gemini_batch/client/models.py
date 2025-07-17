from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union

from google.genai import types


@dataclass(frozen=True)
class PartsPayload:
    parts: List[types.Part]

@dataclass(frozen=True)
class ExplicitCachePayload:
    cache_name: str
    parts: List[types.Part]

class CacheStrategy(Enum):
    GENERATE_RAW = auto()
    GENERATE_WITH_OPTIMIZED_PARTS = auto()
    GENERATE_FROM_EXPLICIT_CACHE = auto()

@dataclass(frozen=True)
class CacheAction:
    strategy: CacheStrategy
    payload: Union[PartsPayload, ExplicitCachePayload]

    def __post_init__(self):
        # Validate payload matches strategy
        if self.strategy == CacheStrategy.GENERATE_FROM_EXPLICIT_CACHE:
            if not isinstance(self.payload, ExplicitCachePayload):
                raise ValueError(f"Strategy {self.strategy} requires ExplicitCachePayload")
        else:
            if not isinstance(self.payload, PartsPayload):
                raise ValueError(f"Strategy {self.strategy} requires PartsPayload")
