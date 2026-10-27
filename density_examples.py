"""Dữ liệu tổng hợp nhiều mật độ; nhãn gốc chỉ dùng để đánh giá.

six_density phỏng theo ví dụ BSD-3-Clause của scikit-learn developers:
https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html
Các bộ còn lại được tạo riêng cho demo này.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


DENSITY_EXAMPLES = {
    "six_density": {
        "name": "6 vùng mật độ — ví dụ scikit-learn",
        "description": "1.500 điểm từ 6 phân bố Gaussian có độ phân tán 0.1–2.0. "
                       "Một số vùng chồng lấn, nên nhãn sinh dữ liệu không đồng nghĩa với 6 cụm tách biệt hoàn toàn.",
        "min_samples": 10, "eps": 0.12,
    },
    "close_dense_sparse": {
        "name": "Hai cụm dày gần nhau + một cụm thưa",
        "description": "720 điểm, 3 nhóm: hai nhóm rất chặt ở gần nhau và một nhóm rộng. "
                       "Tăng eps để giữ nhóm thưa có thể nối hai nhóm dày.",
        "min_samples": 10, "eps": 0.06,
    },
    "density_rings": {
        "name": "Ba vòng tròn — mật độ chênh lệch",
        "description": "900 điểm: hai vòng dày gần nhau (400 điểm/vòng) và một vòng ngoài thưa (100 điểm). "
                       "Khảo sát sự đánh đổi giữa tách hai vòng trong và nối đủ các điểm của vòng ngoài.",
        "min_samples": 5, "eps": 0.06,
    },
}


def make_density_example(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    if name == "six_density":
        centers = [(-5, -2), (4, -1), (1, -2), (-2, 3), (3, -2), (5, 6)]
        scales = [0.8, 0.1, 0.2, 0.3, 1.6, 2.0]
        groups = [np.asarray(c) + s * rng.randn(250, 2) for c, s in zip(centers, scales)]
    elif name == "close_dense_sparse":
        centers = [(-1, 0), (-0.8, 0), (2, 0)]
        groups = [np.asarray(c) + s * rng.randn(240, 2)
                  for c, s in zip(centers, [0.015, 0.015, 0.8])]
    elif name == "density_rings":
        groups = []
        for radius, count in [(1.0, 400), (1.1, 400), (3.0, 100)]:
            angle = np.linspace(0, 2 * np.pi, count, endpoint=False) + rng.normal(0, 0.002, count)
            radii = radius + rng.normal(0, 0.005, count)
            groups.append(np.column_stack((radii * np.cos(angle), radii * np.sin(angle))))
    else:
        raise ValueError(f"Bộ dữ liệu không được hỗ trợ: {name}")
    labels = np.concatenate([np.full(len(group), i, dtype=int) for i, group in enumerate(groups)])
    return np.vstack(groups), labels


def label_recovery(true_labels: np.ndarray, predicted: np.ndarray) -> float:
    """Tỷ lệ khớp tối đa khi ghép cụm một-một; noise luôn tính là sai.

    Các bộ dữ liệu sinh ở đây không chứa nhãn noise gốc. Điểm bị từ chối
    vẫn nằm trong mẫu số, tránh việc bỏ toàn bộ một nhóm nhưng đạt điểm cao.
    """
    true_labels, predicted = np.asarray(true_labels), np.asarray(predicted)
    if true_labels.ndim != 1 or true_labels.shape != predicted.shape or not true_labels.size:
        raise ValueError("Hai mảng nhãn phải cùng kích thước và không rỗng.")
    valid = predicted != -1
    if not valid.any():
        return 0.0
    _, true_ids = np.unique(true_labels, return_inverse=True)
    _, pred_ids = np.unique(predicted[valid], return_inverse=True)
    counts = np.zeros((true_ids.max() + 1, pred_ids.max() + 1), dtype=int)
    np.add.at(counts, (true_ids[valid], pred_ids), 1)
    rows, columns = linear_sum_assignment(-counts)
    return float(counts[rows, columns].sum() / len(true_labels))
