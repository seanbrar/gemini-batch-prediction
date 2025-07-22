"""Quality assessment and comparison for batch responses

This module provides metrics to compare the quality of individual vs batch
responses, helping evaluate the effectiveness of batch processing.
"""  # noqa: D415


def calculate_quality_score(
    individual_answers: list[str],
    batch_answers: list[str],
) -> float | None:
    """Calculate quality comparison between individual and batch answers

    Compares individual and batch answers using three metrics:
    - Completeness: Both answers should be substantive (>10 characters)
    - Word overlap: Jaccard similarity of word sets
    - Length similarity: Ratio of shorter to longer answer

    Returns a score between 0.0 and 1.0, where higher scores indicate
    better quality match between individual and batch responses.
    """  # noqa: D415
    # Handle case where individual answers aren't available
    if not individual_answers or not batch_answers:
        return None

    if len(individual_answers) != len(batch_answers):
        return 0.0

    quality_scores = []

    for ind, batch in zip(individual_answers, batch_answers, strict=False):
        # Completeness check: both answers should be substantive (>10 chars)
        ind_complete = len(ind.strip()) > 10
        batch_complete = len(batch.strip()) > 10
        completeness = 1.0 if (ind_complete and batch_complete) else 0.5

        # Word overlap similarity using Jaccard index
        ind_words = set(ind.lower().split())
        batch_words = set(batch.lower().split())

        if len(ind_words.union(batch_words)) > 0:
            overlap = len(ind_words.intersection(batch_words)) / len(
                ind_words.union(batch_words),
            )
        else:
            overlap = 0.0

        # Length similarity: ratio of shorter to longer answer
        length_ratio = min(len(ind), len(batch)) / max(len(ind), len(batch), 1)

        # Combined score with weighted metrics
        score = (completeness * 0.5) + (overlap * 0.3) + (length_ratio * 0.2)
        quality_scores.append(score)

    return sum(quality_scores) / len(quality_scores)
