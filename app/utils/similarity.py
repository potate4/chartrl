"""Similarity computation utilities using sentence embeddings."""

from typing import List, Optional, Tuple
import numpy as np

# Lazy loading for sentence transformer
_sentence_model = None
_model_name = "all-MiniLM-L6-v2"


def get_sentence_model():
    """Get or initialize the sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(_model_name)
    return _sentence_model


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.

    Uses cosine similarity of sentence embeddings.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score in [0, 1]
    """
    if not text1 or not text2:
        return 0.0

    if text1 == text2:
        return 1.0

    model = get_sentence_model()
    embeddings = model.encode([text1, text2], convert_to_numpy=True)

    # Cosine similarity
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-8
    )

    # Normalize to [0, 1] (cosine sim is in [-1, 1])
    return float((cos_sim + 1) / 2)


def compute_pairwise_similarity(texts: List[str]) -> Tuple[float, List[List[float]]]:
    """
    Compute pairwise similarity matrix and mean.

    Args:
        texts: List of texts to compare

    Returns:
        Tuple of (mean_similarity, similarity_matrix)
    """
    n = len(texts)

    if n < 2:
        return 1.0, [[1.0]]

    # Filter empty texts
    valid_texts = [t for t in texts if t and t.strip()]
    if len(valid_texts) < 2:
        return 1.0, [[1.0]]

    model = get_sentence_model()
    embeddings = model.encode(valid_texts, convert_to_numpy=True)

    # Compute similarity matrix
    n = len(valid_texts)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i][j] = 1.0
            elif j > i:
                # Cosine similarity
                cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                # Normalize to [0, 1]
                normalized = (cos_sim + 1) / 2
                sim_matrix[i][j] = normalized
                sim_matrix[j][i] = normalized

    # Compute mean of upper triangle (excluding diagonal)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    mean_sim = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 1.0

    return mean_sim, sim_matrix.tolist()


def compute_table_similarity(table1: dict, table2: dict) -> float:
    """
    Compute similarity between two table dictionaries.

    Compares:
    - Column headers (exact match fraction)
    - Row values (fuzzy numeric matching)

    Args:
        table1: First table dict
        table2: Second table dict

    Returns:
        Similarity score in [0, 1]
    """
    if not isinstance(table1, dict) or not isinstance(table2, dict):
        return 0.0

    if not table1 or not table2:
        return 0.0

    if table1 == table2:
        return 1.0

    scores = []

    # Compare columns
    cols1 = set(str(c).lower() for c in table1.get("columns", []))
    cols2 = set(str(c).lower() for c in table2.get("columns", []))

    if cols1 or cols2:
        col_intersection = len(cols1 & cols2)
        col_union = len(cols1 | cols2)
        col_sim = col_intersection / col_union if col_union > 0 else 1.0
        scores.append(col_sim)

    # Compare rows
    rows1 = table1.get("rows", [])
    rows2 = table2.get("rows", [])

    if rows1 and rows2:
        row_sim = _compare_rows(rows1, rows2)
        scores.append(row_sim)

    return float(np.mean(scores)) if scores else 0.0


def _compare_rows(rows1: List, rows2: List) -> float:
    """Compare two lists of rows."""
    if not rows1 or not rows2:
        return 0.0

    # Flatten rows to values
    def flatten(rows):
        values = []
        for row in rows:
            if isinstance(row, list):
                values.extend(row)
            elif isinstance(row, dict):
                values.extend(row.values())
            else:
                values.append(row)
        return values

    vals1 = flatten(rows1)
    vals2 = flatten(rows2)

    if not vals1 or not vals2:
        return 0.0

    # Count matching values (with tolerance for numbers)
    matches = 0
    total = max(len(vals1), len(vals2))

    for v1 in vals1:
        for v2 in vals2:
            if _values_match(v1, v2):
                matches += 1
                break

    return matches / total if total > 0 else 0.0


def _values_match(v1, v2, tolerance: float = 0.05) -> bool:
    """Check if two values match (with numeric tolerance)."""
    # String comparison
    if str(v1).strip().lower() == str(v2).strip().lower():
        return True

    # Numeric comparison
    try:
        n1 = float(v1)
        n2 = float(v2)
        if n2 != 0:
            return abs(n1 - n2) / abs(n2) <= tolerance
        return abs(n1 - n2) <= tolerance
    except (ValueError, TypeError):
        return False


def batch_encode(texts: List[str]) -> np.ndarray:
    """
    Encode multiple texts to embeddings.

    Args:
        texts: List of texts

    Returns:
        Numpy array of embeddings (n_texts, embedding_dim)
    """
    model = get_sentence_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
