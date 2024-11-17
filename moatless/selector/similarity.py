import difflib
from typing import Set

from moatless.file_context import FileContext


def calculate_similarity(context_a: FileContext, context_b: FileContext) -> float:
    """
    Calculates the similarity score between the two FileContext instances.

    Returns:
        float: The similarity score between 0 and 1.
    """
    # Step 1: File path similarity
    files_a = set(context_a._files.keys())
    files_b = set(context_b._files.keys())
    file_path_similarity = jaccard_similarity(files_a, files_b)

    # Step 2: Span similarity
    span_similarities = []
    for file_path in files_a.intersection(files_b):
        spans_a = context_a._files[file_path].span_ids
        spans_b = context_b._files[file_path].span_ids
        if spans_a or spans_b:
            span_similarity = jaccard_similarity(spans_a, spans_b)
            span_similarities.append(span_similarity)
        else:
            # If both have no spans, consider them fully similar
            span_similarities.append(1.0)
    if span_similarities:
        average_span_similarity = sum(span_similarities) / len(span_similarities)
    else:
        average_span_similarity = 1.0  # Default to full similarity if no spans

    # Step 3: Patch similarity
    patch_similarities = []
    for file_path in files_a.intersection(files_b):
        patch_a = context_a._files[file_path].patch or ""
        patch_b = context_b._files[file_path].patch or ""
        if patch_a or patch_b:
            patch_similarity = string_similarity(patch_a, patch_b)
            patch_similarities.append(patch_similarity)
        else:
            # If both have no patches, consider them fully similar
            patch_similarities.append(1.0)
    if patch_similarities:
        average_patch_similarity = sum(patch_similarities) / len(patch_similarities)
    else:
        average_patch_similarity = 1.0  # Default to full similarity if no patches

    # Combine the similarities with weights
    total_similarity = (
        0.4 * file_path_similarity
        + 0.2 * average_span_similarity
        + 0.4 * average_patch_similarity
    )

    return total_similarity


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """
    Calculates the Jaccard similarity between two sets.

    Returns:
        float: Jaccard similarity score.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 1.0  # Both sets are empty
    return len(intersection) / len(union)


def string_similarity(s1: str, s2: str) -> float:
    """
    Calculates the similarity between two strings using difflib.

    Returns:
        float: Similarity score between 0 and 1.
    """
    if not s1 and not s2:
        return 1.0  # Both strings are empty
    matcher = difflib.SequenceMatcher(None, s1, s2)
    return matcher.ratio()
