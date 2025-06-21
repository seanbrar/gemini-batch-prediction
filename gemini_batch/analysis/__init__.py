"""
Analysis module for content evaluation and efficiency metrics.

Provides tools for analyzing sources before batch processing to demonstrate
efficiency gains and provide insights into content complexity.
"""

from .content import ContentAnalyzer, SourceSummary

__all__ = [
    "ContentAnalyzer",
    "SourceSummary",
]
