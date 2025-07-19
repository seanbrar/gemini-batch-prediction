from dataclasses import dataclass  # noqa: D100
from enum import Enum, auto
from typing import List, Union  # noqa: UP035

from google.genai import types


@dataclass(frozen=True)
class PartsPayload:  # noqa: D101
    parts: List[types.Part]  # noqa: UP006

@dataclass(frozen=True)
class ExplicitCachePayload:  # noqa: D101
    cache_name: str
    parts: List[types.Part]  # noqa: UP006

class CacheStrategy(Enum):  # noqa: D101
    GENERATE_RAW = auto()
    GENERATE_WITH_OPTIMIZED_PARTS = auto()
    GENERATE_FROM_EXPLICIT_CACHE = auto()

@dataclass(frozen=True)
class CacheAction:  # noqa: D101
    strategy: CacheStrategy
    payload: Union[PartsPayload, ExplicitCachePayload]  # noqa: UP007

    def __post_init__(self):  # noqa: ANN204, D105
        # Validate payload matches strategy
        if self.strategy == CacheStrategy.GENERATE_FROM_EXPLICIT_CACHE:
            if not isinstance(self.payload, ExplicitCachePayload):
                raise ValueError(f"Strategy {self.strategy} requires ExplicitCachePayload")  # noqa: E501, EM102, TRY003
        else:  # noqa: PLR5501
            if not isinstance(self.payload, PartsPayload):
                raise ValueError(f"Strategy {self.strategy} requires PartsPayload")  # noqa: EM102, TRY003
