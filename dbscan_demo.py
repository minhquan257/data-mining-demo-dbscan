"""Demo trực quan thuật toán DBSCAN.

Ví dụ:
    python dbscan_demo.py --dataset moons --eps 0.22 --min-samples 5
    python dbscan_demo.py --dataset blobs --eps 0.35 --save dbscan.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class DemoData:
    points: np.ndarray
    title: str


def embed_in_3d(points: np.ndarray, seed: int) -> np.ndarray:
    """Nhúng dữ liệu 2D thành bề mặt 3D cong, không sử dụng nhãn thật."""
    rng = np.random.default_rng(seed)
    x_values, y_values = points.T
    z_values = (
        0.45 * np.sin(2.2 * x_values)
        + 0.35 * np.cos(2.2 * y_values)
        + rng.normal(0, 0.045, size=len(points))
    )
    return np.column_stack((points, z_values))


def make_data(name: str, seed: int, dimensions: int = 3) -> DemoData:
    """Nạp/tạo một trong ba bộ dữ liệu để chạy DBSCAN ở 2D hoặc 3D."""
    if dimensions not in (2, 3):
        raise ValueError("dimensions chỉ có thể là 2 hoặc 3.")
    if name == "moons":
        # Dùng chính dữ liệu người dùng cung cấp: archive.zip/make_moons.csv.
        archive_path = Path(__file__).with_name("archive.zip")
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {archive_path.name}. Hãy đặt tệp cạnh dbscan_demo.py."
            )
        with ZipFile(archive_path) as archive:
            with archive.open("make_moons.csv") as csv_file:
                points = np.loadtxt(csv_file, delimiter=",", skiprows=1, usecols=(0, 1))
        title = "Make Moons — archive.zip/make_moons.csv"
    elif name == "circles":
        points, _ = make_circles(
            n_samples=1000, noise=0.03, random_state=seed
        )
        title = "Make Circles — make-circles.ipynb"
    else:
        points, _ = make_blobs(
            n_samples=460,
            centers=[(-2, -1), (0.5, 2), (2.5, -0.5)],
            cluster_std=[0.35, 0.6, 0.45],
            random_state=seed,
        )
        rng = np.random.default_rng(seed)
        noise = rng.uniform(low=(-4, -3.5), high=(4.5, 4), size=(40, 2))
        points = np.vstack((points, noise))
        title = "Ba cụm mật độ cao và điểm nhiễu"

    if dimensions == 3:
        points = embed_in_3d(points, seed)
        title = f"{title} (nhúng 3D)"
    else:
        title = f"{title} (2D gốc)"
    return DemoData(StandardScaler().fit_transform(points), title)


def k_distance(points: np.ndarray, min_samples: int) -> np.ndarray:
    """Khoảng cách tới láng giềng thứ k, hữu ích để chọn eps."""
    neighbors = NearestNeighbors(n_neighbors=min_samples).fit(points)
    distances, _ = neighbors.kneighbors(points)
    return np.sort(distances[:, -1])


def describe(labels: np.ndarray, points: np.ndarray) -> str:
    cluster_labels = set(labels) - {-1}
    n_clusters = len(cluster_labels)
    n_noise = int(np.sum(labels == -1))
    message = f"Số cụm: {n_clusters} | Điểm nhiễu: {n_noise}/{len(labels)}"

    mask = labels != -1
    if n_clusters >= 2 and np.unique(labels[mask]).size >= 2:
        score = silhouette_score(points[mask], labels[mask])
        message += f" | Silhouette (bỏ noise): {score:.3f}"
    return message


def plot_result(data: DemoData, eps: float, min_samples: int, save: str | None) -> None:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data.points)
    core_mask = np.zeros(labels.size, dtype=bool)
    core_mask[model.core_sample_indices_] = True

    fig = plt.figure(figsize=(18, 6))
    ax_data = fig.add_subplot(1, 3, 1, projection="3d")
    ax_clusters = fig.add_subplot(1, 3, 2, projection="3d")
    ax_kdist = fig.add_subplot(1, 3, 3)
    fig.suptitle(
        f"DBSCAN — eps={eps}, min_samples={min_samples}\n{describe(labels, data.points)}",
        fontsize=14,
        fontweight="bold",
    )

    ax_data.scatter(
        data.points[:, 0], data.points[:, 1], data.points[:, 2], s=22, c="#52616b"
    )
    ax_data.set_title(f"Dữ liệu đầu vào\n{data.title}")
    ax_data.set_xlabel("Đặc trưng 1 (đã chuẩn hoá)")
    ax_data.set_ylabel("Đặc trưng 2 (đã chuẩn hoá)")
    ax_data.set_zlabel("Đặc trưng Z")

    cmap = plt.colormaps["tab10"]
    for label in sorted(set(labels)):
        mask = labels == label
        if label == -1:
            ax_clusters.scatter(
                data.points[mask, 0],
                data.points[mask, 1],
                data.points[mask, 2],
                s=35,
                c="black",
                marker="x",
                label="Noise (-1)",
            )
            continue

        color = cmap(label % 10)
        ax_clusters.scatter(
            data.points[mask & ~core_mask, 0],
            data.points[mask & ~core_mask, 1],
            data.points[mask & ~core_mask, 2],
            s=24,
            c=[color],
            alpha=0.5,
            label=f"Cụm {label} — border",
        )
        ax_clusters.scatter(
            data.points[mask & core_mask, 0],
            data.points[mask & core_mask, 1],
            data.points[mask & core_mask, 2],
            s=46,
            c=[color],
            edgecolors="white",
            linewidths=0.7,
            label=f"Cụm {label} — core",
        )

    ax_clusters.set_title("Kết quả DBSCAN")
    ax_clusters.set_xlabel("Đặc trưng 1 (đã chuẩn hoá)")
    ax_clusters.set_ylabel("Đặc trưng 2 (đã chuẩn hoá)")
    ax_clusters.set_zlabel("Đặc trưng Z")
    ax_clusters.legend(loc="best", fontsize=8)

    distances = k_distance(data.points, min_samples)
    ax_kdist.plot(distances, color="#236192", linewidth=2)
    ax_kdist.axhline(eps, color="#c23b22", linestyle="--", label=f"eps = {eps}")
    ax_kdist.set_title(f"Biểu đồ khoảng cách láng giềng thứ {min_samples}")
    ax_kdist.set_xlabel("Điểm (đã sắp xếp)")
    ax_kdist.set_ylabel("Khoảng cách")
    ax_kdist.legend()

    for axis in (ax_data, ax_clusters):
        axis.set_box_aspect((1, 1, 0.8))
        axis.grid(alpha=0.2)
    ax_kdist.grid(alpha=0.25)
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=180, bbox_inches="tight")
        print(f"Đã lưu biểu đồ vào: {save}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo và trực quan DBSCAN")
    parser.add_argument(
        "--dataset", choices=("moons", "circles", "blobs"), default="moons"
    )
    parser.add_argument("--eps", type=float, default=0.35, help="Bán kính lân cận")
    parser.add_argument(
        "--min-samples", type=int, default=5, help="Số điểm tối thiểu để là core point"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", help="Đường dẫn PNG để lưu thay vì mở cửa sổ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eps <= 0 or args.min_samples < 2:
        raise SystemExit("eps phải > 0 và min_samples phải >= 2.")
    plot_result(make_data(args.dataset, args.seed), args.eps, args.min_samples, args.save)
