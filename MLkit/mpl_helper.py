import matplotlib.axes
import matplotlib.pyplot as plt
from MLkit.dataset import CategoricalDataSet
from matplotlib.figure import Figure
from typing import Optional, Tuple, List

AxesSubplot = matplotlib.axes.Axes


def _scatter_2D(_data: CategoricalDataSet,
                _ax: AxesSubplot,
                sample: Optional[int] = 200) -> AxesSubplot:
    if sample is not None:
        _data_in = _data.sample(sample)
    else:
        _data_in = _data
    _x_in = _data_in.x
    _ax.scatter(_x_in[:, 0], _x_in[:, 1], c=_data_in.c)
    _ax.set_title(_data_in.name)
    return _ax


def visualize_2D(_data: CategoricalDataSet,
                 size: Tuple[int, int] = (6, 6),
                 sample: Optional[int] = 200) -> Figure:
    _fig, _ax = plt.subplots(nrows=1,
                             ncols=1,
                             figsize=size)
    _scatter_2D(_data, _ax, sample)
    return _fig


def visualize_2D_grid(_datas: List[CategoricalDataSet],
                      ncols: int = 1,
                      size: Tuple[int, int] = (6, 6),
                      sample: Optional[int] = 200,
                      share_axis_range:bool = False) -> Figure:
    nrows = len(_datas) // ncols
    _fig, _axs = plt.subplots(nrows=nrows,
                              ncols=ncols,
                              figsize=(size[0] * ncols, size[1] * nrows))
    if share_axis_range:
        x1_max = x1_min = x2_max = x2_min = None
        for _ax, _data in zip(_axs, _datas):
            _scatter_2D(_data, _ax, sample)
            _x1_min, _x1_max = _ax.get_xlim()
            _x2_min, _x2_max = _ax.get_ylim()
            if x1_max is None:
                x1_max = _x1_max
                x2_max = _x2_max
                x1_min = _x1_min
                x2_min = _x2_min
            else:
                x1_max = max(_x1_max, x1_max)
                x2_max = max(_x2_max, x2_max)
                x1_min = min(_x1_min, x1_min)
                x2_min = min(_x2_min, x2_min)
        for _ax in _axs:
            _ax.set_xlim(x1_min, x1_max)
            _ax.set_ylim(x2_min, x1_max)
    else:
        for _ax, _data in zip(_axs, _datas):
            _scatter_2D(_data, _ax, sample)

    return _fig
