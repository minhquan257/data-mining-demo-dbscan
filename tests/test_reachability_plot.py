import unittest

import numpy as np

from optics_demo import OpticsStructure
from reachability_plot import reachability_bars


class ReachabilityPlotTests(unittest.TestCase):
    def setUp(self):
        self.graph = OpticsStructure(
            np.array([2, 0, 3, 1]), np.array([.1, .2, .3, .4]),
            np.array([np.inf, .2, .5, .9]), np.array([-1, 0, 1, 2]),
        )
        self.labels = np.array([0, 0, 1, -1])
        self.colors = {0: "#00B8D9", 1: "#FF5630"}

    def test_columns_preserve_each_finite_value_and_original_index(self):
        figure = reachability_bars(self.graph, self.labels, self.colors)
        self.assertTrue(all(trace.type == "bar" for trace in figure.data))
        actual = {int(x): (float(y), int(original)) for trace in figure.data
                  for x, y, original in zip(trace.x, trace.y, trace.customdata)}
        self.assertEqual(actual, {0: (.5, 2), 2: (.9, 3), 3: (.2, 1)})
        self.assertFalse(figure.layout.xaxis.rangeslider.visible)
        self.assertEqual(list(figure.layout.xaxis.range), [-.5, 3.5])

    def test_noise_is_black_and_cluster_colors_match_scatter(self):
        figure = reachability_bars(self.graph, self.labels, self.colors)
        traces = {trace.name: trace for trace in figure.data}
        self.assertEqual(traces["Noise (-1)"].marker.color, "#222222")
        self.assertEqual(list(traces["Noise (-1)"].x), [2])
        self.assertEqual(traces["Cụm 0"].marker.color, self.colors[0])
        self.assertEqual(traces["Cụm 1"].marker.color, self.colors[1])

    def test_infinity_is_a_paper_annotation_not_a_finite_column(self):
        figure = reachability_bars(self.graph, self.labels, self.colors)
        infinity = [a for a in figure.layout.annotations if a.text == "∞"]
        self.assertEqual(len(infinity), 1)
        self.assertEqual(infinity[0].yref, "paper")
        self.assertEqual(infinity[0].x, 1)

    def test_cut_does_not_change_columns_or_add_core_line(self):
        plain = reachability_bars(self.graph, self.labels, self.colors)
        detailed = reachability_bars(self.graph, self.labels, self.colors, cut_eps=.7)
        bars = [t for t in detailed.data if t.type == "bar"]
        self.assertEqual([t.to_plotly_json() for t in bars], [t.to_plotly_json() for t in plain.data])
        self.assertTrue(all(t.type == "bar" for t in detailed.data))
        self.assertTrue(any(shape.y0 == .7 and shape.y1 == .7 for shape in detailed.layout.shapes))


if __name__ == "__main__":
    unittest.main()
