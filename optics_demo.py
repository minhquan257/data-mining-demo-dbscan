"""Tính cấu trúc OPTICS bằng scikit-learn, không tự động trích nhãn cụm."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import cluster_optics_dbscan, cluster_optics_xi, compute_optics_graph

EXTRACTION_XI = 0.10


@dataclass
class OpticsStructure:
    ordering: np.ndarray
    core_distances: np.ndarray
    reachability: np.ndarray
    predecessor: np.ndarray


def compute_structure(points: np.ndarray, min_samples: int) -> OpticsStructure:
    """Chỉ MinPts thay đổi; cố định max_eps=∞ và khoảng cách Euclidean.

    Dùng API graph để tránh bước cluster_optics_xi mặc định trong OPTICS.fit.
    Các mảng khoảng cách/predecessor có chỉ số theo hàng đầu vào, không theo
    ordering. Không sinh labels và không coi reachability=∞ là noise.
    """
    result = compute_optics_graph(
        points, min_samples=min_samples, max_eps=np.inf, metric="euclidean",
        p=2, metric_params=None, algorithm="auto", leaf_size=30, n_jobs=None,
    )
    return OpticsStructure(*result)


def extract_labels(structure: OpticsStructure, min_samples: int, *,
                   method: str = "xi", xi: float = EXTRACTION_XI,
                   eps: float = 1.5, min_cluster_size: int | float | None = None) -> np.ndarray:
    """Bước trích cụm riêng: Xi hoặc cắt ngang eps', không dựng lại ordering.

    Không dùng nhãn gốc hay nhãn thuật toán khác để quyết định phân hoạch.
    """
    if method == "dbscan":
        return cluster_optics_dbscan(
            reachability=structure.reachability, core_distances=structure.core_distances,
            ordering=structure.ordering, eps=eps,
        )
    if method != "xi":
        raise ValueError("method chỉ có thể là xi hoặc dbscan.")
    # scikit-learn divides by (1 - xi); approximate the upper endpoint.
    effective_xi = float(np.nextafter(1.0, 0.0)) if xi == 1.0 else xi
    labels, _ = cluster_optics_xi(
        reachability=structure.reachability, predecessor=structure.predecessor,
        ordering=structure.ordering, min_samples=min_samples,
        min_cluster_size=min_cluster_size, xi=effective_xi, predecessor_correction=True,
    )
    return labels
