import unittest

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from optics_demo import compute_structure, extract_labels
from slide_simulation import DBSCAN_POINTS, OPTICS_POINTS, dbscan_steps, optics_steps, density_hierarchy


class SlideSimulationTests(unittest.TestCase):
    def test_dbscan_matches_slide_and_sklearn(self):
        frames = dbscan_steps(DBSCAN_POINTS, .8, 4)
        final = frames[-1]
        reference = DBSCAN(eps=.8, min_samples=4).fit(DBSCAN_POINTS)
        np.testing.assert_array_equal(final["labels"], reference.labels_)
        self.assertEqual(int(final["core"].sum()), 6)
        self.assertEqual(int(((final["labels"] >= 0) & ~final["core"]).sum()), 8)
        self.assertEqual(int((final["labels"] == -1).sum()), 1)
        self.assertEqual(len(frames), 17)  # initial + 15 visits + final
        self.assertEqual(frames[1]["current"], 0)
        self.assertEqual(frames[1]["added"], [1, 2, 3, 4])

    def test_dbscan_noise_can_be_reassigned_to_border(self):
        points = np.array([[0, 0], [.8, 0], [1, 0], [1.1, .1], [1, -.1]], dtype=float)
        frames = dbscan_steps(points, .9, 4)
        self.assertEqual(frames[1]["labels"][0], -1)
        np.testing.assert_array_equal(frames[-1]["labels"], DBSCAN(eps=.9, min_samples=4).fit_predict(points))
        self.assertGreaterEqual(frames[-1]["labels"][0], 0)

    def test_optics_queue_trace_matches_sklearn(self):
        for minpts in [2, 4, 6, 8]:
            with self.subTest(minpts=minpts):
                graph = compute_structure(OPTICS_POINTS, minpts)
                frames = optics_steps(OPTICS_POINTS, minpts)
                np.testing.assert_array_equal(frames[-1]["order"], graph.ordering)
                np.testing.assert_allclose(frames[-1]["reach"], graph.reachability)
                np.testing.assert_array_equal(frames[-1]["predecessor"], graph.predecessor)
                for frame in frames:
                    self.assertTrue(all(not frame["visited"][q] for q in frame["queue"]))
                    pairs = [(frame["reach"][q], q) for q in frame["queue"]]
                    self.assertEqual(pairs, sorted(pairs))

    def test_optics_reachability_values_match_slide(self):
        graph = compute_structure(OPTICS_POINTS, 4)
        expected = [np.inf, .57, .38, .38, .43, 1., .56, .4, .4, .41, 2.19, 1.38, 1.38, 1.38, 1.38, 1.38]
        np.testing.assert_array_equal(np.round(graph.reachability[graph.ordering], 2), expected)

    def test_two_cuts_and_xi_match_slide(self):
        graph = compute_structure(OPTICS_POINTS, 4)
        large = extract_labels(graph, 4, method="dbscan", eps=1.5)
        small = extract_labels(graph, 4, method="dbscan", eps=.9)
        xi = extract_labels(graph, 4, xi=.1)
        np.testing.assert_array_equal(large, [0]*10 + [1]*6)
        np.testing.assert_array_equal(small, [0]*5 + [1]*5 + [-1]*6)
        np.testing.assert_array_equal(xi, [0]*5 + [1]*5 + [2]*6)

    def test_hierarchy_eom_and_stability_match_slide(self):
        hierarchy = density_hierarchy(OPTICS_POINTS, 4, 4)
        self.assertAlmostEqual(hierarchy.nodes[1]["stability"], 5.43, places=2)
        self.assertAlmostEqual(hierarchy.nodes[3]["stability"] + hierarchy.nodes[4]["stability"], 7.52, places=2)
        self.assertEqual(hierarchy.selected, {2, 3, 4})
        expected = HDBSCAN(min_samples=4, min_cluster_size=4, copy=True).fit_predict(OPTICS_POINTS)
        self.assertEqual(adjusted_rand_score(hierarchy.labels, expected), 1.)

    def test_hierarchy_matches_sklearn_across_lesson_parameters(self):
        for minpts in [2, 4, 8]:
            for size in [2, 4, 8]:
                with self.subTest(minpts=minpts, size=size):
                    hierarchy = density_hierarchy(OPTICS_POINTS, minpts, size)
                    expected = HDBSCAN(min_samples=minpts, min_cluster_size=size, copy=True).fit_predict(OPTICS_POINTS)
                    self.assertEqual(adjusted_rand_score(hierarchy.labels, expected), 1.)

    def test_hierarchy_cuts_show_split_and_dissolution(self):
        hierarchy = density_hierarchy(OPTICS_POINTS, 4, 4)
        self.assertEqual(len(set(hierarchy.cut(2.5, 4)) - {-1}), 1)
        self.assertEqual(len(set(hierarchy.cut(2., 4)) - {-1}), 2)
        small = hierarchy.cut(.8, 4)
        self.assertEqual(len(set(small) - {-1}), 2)
        np.testing.assert_array_equal(small[10:], -1)

    def test_silhouette_sample_matches_slide(self):
        points = np.array([(0, 0), (2, 0), (0, -2), (3, 4), (4, 3), (6, 8), (8, 6)], dtype=float)
        self.assertAlmostEqual(silhouette_samples(points, [0, 0, 0, 1, 1, 2, 2])[0], .6)


if __name__ == "__main__":
    unittest.main()
