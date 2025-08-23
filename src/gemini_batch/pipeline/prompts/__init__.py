"""Prompt assembly system for the pipeline.

This module provides the PromptBundle data model and assembler functionality
for composing prompts from configuration, files, and advanced builder hooks.
"""

from .assembler import assemble_prompts

__all__ = ["assemble_prompts"]
