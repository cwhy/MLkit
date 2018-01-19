import numpy as np
from MLkit.dataset import CategoricalDataSet
from MLkit.mpl_helper import watch_data_2D, bounded_line
import matplotlib.pyplot as plt

test_watch = False
if test_watch:
    z = lambda: CategoricalDataSet(np.random.normal(size=(100, 2)),
                                   np.random.binomial(1, 0.2, (100, 1)))


    def shuffle_set(_ds):
        new_set = CategoricalDataSet(_ds.x, _ds.y,
                                     y_encoding=_ds.y_encoding,
                                     shuffle=True)
        return new_set


    data = z()
    T = np.random.normal(size=(2, 2))

    plt.ion()
    fig, update_fig = watch_data_2D(data)
    plt.draw()
    for _ in range(10):
        print("????")
        plt.pause(1)
        update_fig(shuffle_set(data))
        print(shuffle_set(data).y.T)

test_bounded_line = True
if test_bounded_line:
    plt.ion()
    fig, ax = plt.subplots()
    x = np.linspace(0, 10)
    y = x ** 2
    plt.plot(x, y)
    plt.show()
    l = bounded_line([1, 20], [6, 70], ax)


    def test(p1, p2):
        l = bounded_line(p1, p2, ax)
        fig.canvas.draw()
        return l

    l = test([6, 40], [8, 20])
    l = test([7, 20], [6, 70])
    l = test([6, 20], [6, 70])
    l = test([6, 20], [8, 20])
    l = test([6, 70], [8, 20])
