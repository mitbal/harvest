import importlib.util
from pathlib import Path

# harvest.plot is stubbed by conftest; load the real file under another name.
_PLOT_PATH = Path(__file__).parents[2] / 'harvest' / 'plot.py'
_spec = importlib.util.spec_from_file_location('harvest_plot_treemap_test', _PLOT_PATH)
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)


def test_treemap_bin_colors_match_ramp_endpoints_and_midpoint():
    ramp = plot.TREEMAP_COLORMAPS['red_green']
    colors = plot.treemap_bin_colors('red_green', 9)
    assert len(colors) == 9
    assert colors[0] == ramp[0].upper()
    assert colors[-1] == ramp[-1].upper()
    assert colors[4] == ramp[3].upper()  # middle bin is the neutral midpoint


def test_treemap_bin_colors_single_bin():
    assert plot.treemap_bin_colors('green_shade', 1) == [plot.TREEMAP_COLORMAPS['green_shade'][0].upper()]
