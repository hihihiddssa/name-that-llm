import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


def prompt_group_decode(texts, probabilities, threshold=0.72, duplicate_penalty=0.55):
    independent = probabilities.argmax(axis=1)
    if len(texts) < 2:
        return independent
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2, dtype=np.float32,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError as error:
        if "vocabulary" in str(error) or "documents" in str(error) or "terms remain" in str(error):
            return independent
        raise
    distances = np.clip(1.0 - (matrix @ matrix.T).toarray(), 0, 1)
    groups = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold,
    ).fit_predict(distances)
    predictions = independent.copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        count = len(rows)
        if not 2 <= count <= probabilities.shape[1]:
            continue
        negative_log = -np.log(probabilities[rows].clip(1e-7, 1))
        assignment_cost = negative_log[:, :, None] + duplicate_penalty * np.arange(count)[None, None, :]
        selected_rows, slots = linear_sum_assignment(assignment_cost.reshape(count, -1))
        predictions[rows[selected_rows]] = slots // count
    return predictions
