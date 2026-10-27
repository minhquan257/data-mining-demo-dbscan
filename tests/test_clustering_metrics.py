import unittest

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS

from clustering_metrics import calculate_metrics
from dbscan_demo import make_data


def manual_silhouette(points, labels):
    """Tính trực tiếp a(i), b(i) để đối chiếu độc lập với scorer scikit-learn."""
    mask = labels >= 0
    x, groups = points[mask], labels[mask]
    if not 2 <= len(np.unique(groups)) < len(groups):
        return None
    distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    scores = []
    for i, label in enumerate(groups):
        same = groups == label
        same[i] = False
        if not same.any():
            scores.append(0.)
            continue
        a = distances[i, same].mean()
        b = min(distances[i, groups == other].mean() for other in np.unique(groups) if other != label)
        scores.append((b-a)/max(a, b) if max(a, b) else 0.)
    return float(np.mean(scores))


class ClusteringMetricsTests(unittest.TestCase):
    def test_counts_exclude_noise_and_do_not_assume_contiguous_labels(self):
        points = np.array([[0.], [1.], [10.], [11.], [100.]])
        labels = np.array([3, 3, 17, 17, -1])
        actual = calculate_metrics(points, labels)
        self.assertEqual(actual["Số cụm"], 2)
        self.assertEqual(actual["Điểm nhiễu"], 1)
        self.assertEqual(actual["Tỷ lệ nhiễu (%)"], 20.)
        self.assertAlmostEqual(actual["Silhouette (bỏ noise)"], manual_silhouette(points, labels), places=12)

    def test_undefined_silhouette_cases(self):
        points = np.array([[0.], [1.], [2.]])
        for labels in [[-1, -1, -1], [0, 0, 0], [0, 1, 2], [0, 0, -1]]:
            with self.subTest(labels=labels):
                self.assertIsNone(calculate_metrics(points, np.array(labels))["Silhouette (bỏ noise)"])

    def test_singleton_cluster_contributes_zero(self):
        points, labels = np.array([[0.], [1.], [10.]]), np.array([0, 0, 1])
        self.assertAlmostEqual(calculate_metrics(points, labels)["Silhouette (bỏ noise)"], manual_silhouette(points, labels))

    def test_all_negative_labels_are_excluded(self):
        result = calculate_metrics(np.array([[0.], [1.], [2.]]), np.array([-1, -2, -3]))
        self.assertEqual(result["Số cụm"], 0)
        self.assertEqual(result["Điểm nhiễu"], 3)
        self.assertEqual(result["Tỷ lệ nhiễu (%)"], 100.)

    def test_ratio_uses_total_number_of_points(self):
        result = calculate_metrics(np.array([[0.], [1.], [2.]]), np.array([0, 0, -1]))
        self.assertEqual(result["Tỷ lệ nhiễu (%)"], 33.33)

    def test_invalid_shapes(self):
        for points, labels in [(np.empty((0, 2)), np.array([])), (np.zeros((3, 2)), np.array([0, 0]))]:
            with self.assertRaises(ValueError):
                calculate_metrics(points, labels)

    def test_screenshot_moons_regression(self):
        points = make_data("moons", 42, 2).points
        models = [DBSCAN(eps=.35, min_samples=5),
                  OPTICS(min_samples=5, max_eps=np.inf, metric="euclidean", xi=.1, min_cluster_size=.1),
                  HDBSCAN(min_samples=5, min_cluster_size=20, copy=True)]
        expected = [(1, 0, 0., None), (2, 5, .5, .3811), (2, 11, 1.1, .3815)]
        for model, (clusters, noise, ratio, silhouette) in zip(models, expected):
            with self.subTest(model=type(model).__name__):
                labels = model.fit_predict(points)
                result = calculate_metrics(points, labels)
                self.assertEqual(result["Số cụm"], clusters)
                self.assertEqual(result["Điểm nhiễu"], noise)
                self.assertEqual(result["Tỷ lệ nhiễu (%)"], ratio)
                if silhouette is None:
                    self.assertIsNone(result["Silhouette (bỏ noise)"])
                else:
                    self.assertAlmostEqual(result["Silhouette (bỏ noise)"], silhouette, places=4)
                    self.assertAlmostEqual(result["Silhouette (bỏ noise)"], manual_silhouette(points, labels), places=12)


if __name__ == "__main__":
    unittest.main()
