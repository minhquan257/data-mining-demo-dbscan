import unittest

import numpy as np

from dbscan_demo import make_data
from density_examples import DENSITY_EXAMPLES, label_recovery


class LabelRecoveryTests(unittest.TestCase):
    def test_permuted_labels_are_equivalent(self):
        self.assertEqual(label_recovery([0, 0, 1, 1], [7, 7, 2, 2]), 1.0)

    def test_rejected_group_is_not_a_correct_cluster(self):
        self.assertEqual(label_recovery([0, 0, 1, 1], [7, 7, -1, -1]), 0.5)
        self.assertEqual(label_recovery([0, 1], [-1, -1]), 0.0)

    def test_merges_and_splits_are_penalized(self):
        self.assertEqual(label_recovery([0, 0, 1, 1], [7, 7, 7, 7]), 0.5)
        self.assertEqual(label_recovery([0, 0, 0, 0], [0, 0, 1, 1]), 0.5)

    def test_invalid_shapes(self):
        for actual, predicted in [([], []), ([0, 1], [0]), ([[0, 1]], [[0, 1]])]:
            with self.assertRaises(ValueError):
                label_recovery(actual, predicted)


class DensityExampleTests(unittest.TestCase):
    def test_reproducible_aligned_labels_in_both_dimensions(self):
        for name in DENSITY_EXAMPLES:
            with self.subTest(name=name):
                original = make_data(name, 42, 2)
                repeated = make_data(name, 42, 2)
                spatial = make_data(name, 42, 3)
                np.testing.assert_array_equal(original.points, repeated.points)
                np.testing.assert_array_equal(original.true_labels, spatial.true_labels)
                self.assertEqual(spatial.points.shape, (len(original.points), 3))
                self.assertTrue(np.isfinite(spatial.points).all())
                self.assertEqual(len(original.points), len(original.true_labels))

    def test_original_datasets_have_no_synthetic_evaluation_labels(self):
        for name in ["moons", "circles", "blobs", "variable_density"]:
            with self.subTest(name=name):
                self.assertIsNone(make_data(name, 42, 2).true_labels)


if __name__ == "__main__":
    unittest.main()
