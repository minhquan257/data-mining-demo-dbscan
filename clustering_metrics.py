"""Các chỉ số của bảng so sánh, tính trên nhãn trước khi ghép màu."""

import numpy as np
from sklearn.metrics import silhouette_score


def calculate_metrics(points: np.ndarray, labels: np.ndarray) -> dict:
    points, labels = np.asarray(points), np.asarray(labels)
    if points.ndim != 2 or labels.ndim != 1 or len(points) != len(labels) or not len(labels):
        raise ValueError("Cần ma trận điểm không rỗng và một nhãn cho mỗi điểm.")
    valid = labels >= 0
    n_clusters = int(np.unique(labels[valid]).size)
    n_noise = int(np.count_nonzero(~valid))
    score = None
    if 2 <= n_clusters < int(valid.sum()):
        score = float(silhouette_score(points[valid], labels[valid], metric="euclidean"))
    return {"Số cụm": n_clusters, "Điểm nhiễu": n_noise,
            "Tỷ lệ nhiễu (%)": round(100 * n_noise / len(labels), 2),
            "Silhouette (bỏ noise)": score}
