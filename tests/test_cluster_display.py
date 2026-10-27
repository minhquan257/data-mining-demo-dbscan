import unittest

import numpy as np

from cluster_display import align_for_display, axis_limits


class ClusterDisplayTests(unittest.TestCase):
    def test_permuted_clusters_align_by_members_not_numeric_label(self):
        reference = np.array([0, 0, 1, 1, -1])
        other = np.array([7, 7, 3, 3, -1])
        actual, = align_for_display(reference, other)
        np.testing.assert_array_equal(actual, reference)

    def test_split_merge_and_unmatched_groups_preserve_partition_and_noise(self):
        reference = np.array([0, 0, 1, 1, -1, -1, -1])
        candidates = [np.array([5, 5, 5, 5, 6, 6, -1]),
                      np.array([3, 4, 1, 1, -1, 7, 7])]
        for before, after in zip(candidates, align_for_display(reference, *candidates)):
            np.testing.assert_array_equal(before == -1, after == -1)
            np.testing.assert_array_equal(before[:, None] == before, after[:, None] == after)

    def test_all_noise_reference_does_not_discard_other_clusters(self):
        actual, = align_for_display(np.array([-1, -1, -1]), np.array([4, 4, -1]))
        self.assertEqual(actual[0], actual[1])
        self.assertGreaterEqual(actual[0], 0)
        self.assertEqual(actual[2], -1)

    def test_axis_limits_cover_points_with_equal_spans(self):
        for points in [np.array([[0., 1., -3.], [2., 7., 0.]]), np.zeros((4, 2))]:
            limits = axis_limits(points)
            self.assertTrue((points >= limits[:, 0]).all())
            self.assertTrue((points <= limits[:, 1]).all())
            spans = limits[:, 1] - limits[:, 0]
            np.testing.assert_allclose(spans, spans[0])


if __name__ == "__main__":
    unittest.main()
