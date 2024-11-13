import unittest
from draw_plots import PlotDrawer

data = "D:\Work\AI ML Team\data.json"
class TestPlotDrawer(unittest.TestCase):
    def test_draw_plots(self):
        drawer = PlotDrawer(data)
        plot_paths = drawer.draw_plots()
        self.assertTrue(len(plot_paths) > 0)

if __name__ == "__main__":
    unittest.main()
