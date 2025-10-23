import numpy as np
from typing import List

def rocchio_refine(
    original_query_vector: np.ndarray,
    positive_doc_vectors: List[np.ndarray],
    negative_doc_vectors: List[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15
) -> np.ndarray:
    """
    Refines a query vector using the Rocchio algorithm.

    Args:
        original_query_vector: The initial query vector.
        positive_doc_vectors: A list of vectors for relevant documents.
        negative_doc_vectors: A list of vectors for non-relevant documents.
        alpha: Weight for the original query vector.
        beta: Weight for the positive (relevant) feedback vectors.
        gamma: Weight for the negative (non-relevant) feedback vectors.

    Returns:
        The refined query vector as a numpy array.
    """
    if not isinstance(original_query_vector, np.ndarray):
        raise TypeError("original_query_vector must be a numpy array")

    positive_centroid = np.zeros_like(original_query_vector)
    if positive_doc_vectors:
        positive_centroid = np.mean(positive_doc_vectors, axis=0)

    negative_centroid = np.zeros_like(original_query_vector)
    if negative_doc_vectors:
        negative_centroid = np.mean(negative_doc_vectors, axis=0)

    # Rocchio formula: q_new = alpha * q_original + beta * mean(pos_docs) - gamma * mean(neg_docs)
    refined_vector = (
        alpha * original_query_vector +
        beta * positive_centroid -
        gamma * negative_centroid
    )

    # Normalize the resulting vector to maintain unit length
    norm = np.linalg.norm(refined_vector)
    return refined_vector / norm if norm > 0 else refined_vector
