import unittest

import numpy as np
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors

from dbscan_demo import make_data
from density_examples import DENSITY_EXAMPLES
from optics_demo import EXTRACTION_XI, compute_structure, extract_labels


class OpticsStructureTests(unittest.TestCase):
    def test_explicit_extraction_matches_sklearn_defaults_without_mutating_structure(self):
        points = make_data("blobs", 42, 2).points
        graph = compute_structure(points, 5)
        reachability = graph.reachability.copy()
        labels = extract_labels(graph, 5)
        reference = OPTICS(min_samples=5, max_eps=np.inf, metric="euclidean",
                           xi=EXTRACTION_XI, min_cluster_size=None).fit_predict(points)
        np.testing.assert_array_equal(labels, reference)
        np.testing.assert_array_equal(graph.reachability, reachability)
        self.assertEqual(labels.shape, (len(points),))

    def test_structure_matches_sklearn_independently_of_xi_extraction(self):
        points = np.random.RandomState(42).normal(size=(60, 2))
        actual = compute_structure(points, 5)
        for xi, size in [(0.05, None), (0.15, 0.3)]:
            reference = OPTICS(min_samples=5, max_eps=np.inf, metric="euclidean",
                               xi=xi, min_cluster_size=size).fit(points)
            np.testing.assert_array_equal(actual.ordering, reference.ordering_)
            np.testing.assert_allclose(actual.core_distances, reference.core_distances_)
            np.testing.assert_allclose(actual.reachability, reference.reachability_)
            np.testing.assert_array_equal(actual.predecessor, reference.predecessor_)
        self.assertFalse(hasattr(actual, "labels_"))

    def test_core_distance_counts_the_point_itself(self):
        points = np.array([[0., 0.], [0., 0.], [1., 0.], [3., 0.], [8., 0.]])
        actual = compute_structure(points, 3)
        neighbors = NearestNeighbors(n_neighbors=3).fit(points)
        expected = neighbors.kneighbors(points)[0][:, -1]
        np.testing.assert_allclose(actual.core_distances, expected)
        self.assertTrue(np.isinf(actual.reachability[actual.ordering[0]]))
        self.assertEqual(actual.predecessor[actual.ordering[0]], -1)

    def test_all_example_presets_return_a_complete_ordering(self):
        for name, preset in DENSITY_EXAMPLES.items():
            with self.subTest(name=name):
                points = make_data(name, 42, 2).points
                actual = compute_structure(points, preset["min_samples"])
                np.testing.assert_array_equal(np.sort(actual.ordering), np.arange(len(points)))
                self.assertTrue(np.isfinite(actual.core_distances).all())
                self.assertFalse(np.isnan(actual.reachability).any())
                self.assertTrue(((actual.predecessor >= -1) & (actual.predecessor < len(points))).all())

    def test_identical_points_have_zero_distances_not_noise(self):
        actual = compute_structure(np.zeros((5, 2)), 2)
        np.testing.assert_array_equal(actual.core_distances, 0)
        self.assertEqual(np.isinf(actual.reachability).sum(), 1)
        np.testing.assert_array_equal(actual.reachability[actual.ordering[1:]], 0)


if __name__ == "__main__":
    unittest.main()
