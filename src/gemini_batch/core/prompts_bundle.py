"""Immutable container for assembled prompts with provenance."""

from __future__ import annotations

import dataclasses
import typing

from ._validation import _freeze_mapping, _is_tuple_of, _require


@dataclasses.dataclass(frozen=True, slots=True)
class PromptBundle:
    """Immutable container for assembled prompts with provenance.

    This represents the final, composed prompts after assembly from various
    sources (inline config, files, builder hooks). The bundle preserves the
    exact count of user prompts to maintain batching invariants while allowing
    system instructions to be added separately.
    """

    user: tuple[
        str, ...
    ]  # Transformed user prompts (prefix/suffix applied), count preserved
    system: str | None = None  # Optional system instruction
    hints: typing.Mapping[str, typing.Any] = dataclasses.field(
        default_factory=dict
    )  # Provenance flags (has_sources, user_from, etc.)

    def __post_init__(self) -> None:
        """Validate and freeze prompt bundle components."""
        # Ensure user prompts are strings and not empty
        _require(
            condition=_is_tuple_of(self.user, str),
            message="must be a tuple[str, ...]",
            field_name="user",
            exc=TypeError,
        )
        _require(
            condition=len(self.user) > 0,
            message="must contain at least one prompt",
            field_name="user",
        )
        # Validate that user prompts are not all empty
        non_empty_prompts = [p for p in self.user if p.strip()]
        _require(
            condition=len(non_empty_prompts) > 0,
            message="must contain at least one non-empty prompt",
            field_name="user",
        )

        _require(
            condition=self.system is None or isinstance(self.system, str),
            message="must be a str or None",
            field_name="system",
            exc=TypeError,
        )
        # System prompt should not be empty string if provided
        if self.system is not None:
            _require(
                condition=self.system.strip() != "",
                message="cannot be empty string when provided",
                field_name="system",
            )

        # Freeze hints mapping to prevent downstream mutation
        frozen = _freeze_mapping(self.hints)
        if frozen is not None:
            object.__setattr__(self, "hints", frozen)
