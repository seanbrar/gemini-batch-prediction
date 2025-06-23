"""
Quality assessment and comparison for batch responses
"""

from typing import List, Optional


def calculate_quality_score(
    individual_answers: List[str], batch_answers: List[str]
) -> Optional[float]:
    """Calculate quality comparison between individual and batch answers"""
    # Handle case where individual answers aren't available
    if not individual_answers or not batch_answers:
        return None

    if len(individual_answers) != len(batch_answers):
        return 0.0

    quality_scores = []

    for ind, batch in zip(individual_answers, batch_answers):
        # Completeness check (both answers should be substantive)
        ind_complete = len(ind.strip()) > 10
        batch_complete = len(batch.strip()) > 10
        completeness = 1.0 if (ind_complete and batch_complete) else 0.5

        # Word overlap similarity
        ind_words = set(ind.lower().split())
        batch_words = set(batch.lower().split())

        if len(ind_words.union(batch_words)) > 0:
            overlap = len(ind_words.intersection(batch_words)) / len(
                ind_words.union(batch_words)
            )
        else:
            overlap = 0.0

        # Length similarity (with a preference for similar-length answers)
        length_ratio = min(len(ind), len(batch)) / max(len(ind), len(batch), 1)

        # Combined score
        score = (completeness * 0.5) + (overlap * 0.3) + (length_ratio * 0.2)
        quality_scores.append(score)

    return sum(quality_scores) / len(quality_scores)
