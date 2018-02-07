import numpy as np
import matplotlib as mpl
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt
from MLkit.dataset import CategoricalDataSet
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from typing import Optional, Tuple, List, Callable, Union

AxesSubplot = Axes
Vector = Union[np.ndarray, List, Tuple]

mpl.rcParams["font.sans-serif"] = ["Fira Sans", "Candara",
                                   "Calibri", "Segoe", "Segoe UI",
                                   "Optima", "Arial"]
mpl.rcParams["font.family"] = "sans-serif"


def visualize_matrix(mat: np.ndarray) -> Tuple[Figure, AxesSubplot, AxesImage]:
    f, _ax = plt.subplots(figsize=(5, 5), dpi=600)
    _ax.spines['left'].set_visible(False)
    _ax.spines['bottom'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.spines['top'].set_visible(False)
    pt = _ax.matshow(mat, cmap="Greys")
    cax = f.colorbar(pt, shrink=0.8, drawedges=False)
    cax.outline.set_visible(False)
    return f, _ax, pt


def _scatter_2D(_data: CategoricalDataSet,
                _ax: AxesSubplot,
                sample: Optional[int] = 200) -> Tuple[AxesSubplot, PathCollection]:
    if sample is not None:
        _data_in = _data.sample(sample, name_mod=None)
    else:
        _data_in = _data
    _x_in = _data_in.x
    sc_handle = _ax.scatter(_x_in[:, 0], _x_in[:, 1], c=_data_in.c)
    _ax.set_title(_data_in.name)
    return _ax, sc_handle


def visualize_data_2D(_data: CategoricalDataSet,
                      size: Tuple[int, int] = (6, 6),
                      sample: Optional[int] = 200) -> Figure:
    _fig, _ax = plt.subplots(nrows=1,
                             ncols=1,
                             figsize=size)
    _scatter_2D(_data, _ax, sample)
    return _fig


def watch_data_2D(init_data: CategoricalDataSet,
                  size: Tuple[int, int] = (6, 6)
                  ) -> Tuple[AxesSubplot, Callable[[CategoricalDataSet], None]]:
    _fig, _ax = plt.subplots(nrows=1,
                             ncols=1,
                             figsize=size)
    _, handle = _scatter_2D(init_data, _ax, sample=None)
    _fig.canvas.draw()

    def update(new_data: CategoricalDataSet):
        _x_in = new_data.x
        handle.set_offsets(_x_in[:, 0:2])
        _fig.canvas.draw()

    return _ax, update


def visualize_data_2D_grid(_datas: List[CategoricalDataSet],
                           ncols: int = 1,
                           size: Tuple[int, int] = (6, 6),
                           sample: Optional[int] = 200,
                           share_axis_range: bool = False) -> Figure:
    nrows = -(-len(_datas) // ncols)  # Sneaky ceil()
    _fig, _axs = plt.subplots(nrows=nrows,
                              ncols=ncols,
                              figsize=(size[0] * ncols, size[1] * nrows))
    _axs = _axs.ravel()
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


def bounded_f_line(f: Callable[[float], float],
                   inv_f: Callable[[float], float],
                   _ax: AxesSubplot, **kwargs) -> Optional[Line2D]:
    e = 0.000001
    x_min, x_max = _ax.get_xbound()
    y_min, y_max = _ax.get_ybound()
    fx_min = f(x_min)
    fx_max = f(x_max)
    fy_min = inv_f(y_min)
    fy_max = inv_f(y_max)
    x_range = [x_min, x_max]
    y_range = [fx_min, fx_max]
    in_range = True
    if not (y_min - e <= fx_min <= y_max + e):
        if x_max + e >= fy_max >= fy_min >= x_min - e:
            x_range[0], y_range[0] = fy_min, y_min
        elif x_max + e >= fy_min >= fy_max >= x_min - e:
            x_range[0], y_range[0] = fy_max, y_max
        else:
            in_range = False
    if not (y_min - e <= fx_max <= y_max + e):
        if x_max + e >= fy_max >= fy_min >= x_min - e:
            x_range[1], y_range[1] = fy_max, y_max
        elif x_max + e >= fy_min >= fy_max >= x_min - e:
            x_range[1], y_range[1] = fy_min, y_min
        else:
            in_range = False

    if in_range:
        line = Line2D(x_range, y_range, **kwargs)
        _ax.add_line(line)
        return line
    else:
        return None


def bounded_line(p1: Vector, p2: Vector,
                 _ax: AxesSubplot,
                 **kwargs) -> Optional[Line2D]:
    if p2[0] == p1[0]:
        return _ax.axvline(x=p1[0])
    elif p2[1] == p1[1]:
        return _ax.axhline(y=p1[1])
    else:
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])

        def f(x):
            return p1[1] + k * (x - p1[0])

        def inv_f(y):
            return p1[0] + 1 / k * (y - p1[1])

        return bounded_f_line(f, inv_f, _ax, **kwargs)
