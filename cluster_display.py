"""Chuẩn hoá nhãn hiển thị và phạm vi trục, không thay đổi phân hoạch."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_for_display(reference: np.ndarray, *predictions: np.ndarray) -> list[np.ndarray]:
    """Ghép màu một-một theo số điểm chung, giữ nguyên noise và thành viên cụm."""
    reference_ids = np.unique(reference[reference >= 0])
    next_label = int(reference_ids.max()) + 1 if reference_ids.size else 0
    aligned = []
    for labels in predictions:
        ids = np.unique(labels[labels >= 0])
        counts = np.array([[np.count_nonzero((labels == i) & (reference == j))
                            for j in reference_ids] for i in ids], dtype=int)
        mapping = {}
        if ids.size and reference_ids.size:
            rows, cols = linear_sum_assignment(-counts)
            mapping = {int(ids[r]): int(reference_ids[c]) for r, c in zip(rows, cols) if counts[r, c] > 0}
        result = np.full(labels.shape, -1, dtype=int)
        for label in ids:
            if int(label) not in mapping:
                mapping[int(label)] = next_label
                next_label += 1
            result[labels == label] = mapping[int(label)]
        aligned.append(result)
    return aligned


def axis_limits(points: np.ndarray) -> np.ndarray:
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    half_span = max(float((high - low).max()), 1.0) * 0.55
    return np.column_stack((center - half_span, center + half_span))
