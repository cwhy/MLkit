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
    x = np.linspace(0, 1)
    y = x ** 2
    plt.plot(x, y)
    plt.show()
    l = bounded_line([1, 20], [6, 70], ax)


    def update_line(p1, p2, _l):
        if _l is not None:
            _l.remove()
        new_l = bounded_line(p1, p2, ax)
        fig.canvas.draw()
        return new_l


    for _ in range(10):
        p1 = np.random.rand(2)
        p2 = np.random.rand(2)
        l = update_line(p1, p2, l)
        if l is None:
            print("Outside")
        plt.pause(0.2)


    def put_line(p1, p2):
        bounded_line(p1, p2, ax)
        fig.canvas.draw()


    for _ in range(20):
        p1 = np.random.rand(2)
        p2 = np.random.rand(2)
        put_line(p1, p2)
    ax.set_ylim((ax.get_ylim()[0] - 1, ax.get_ylim()[1] + 1))
    ax.set_xlim((ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1))
    plt.pause(100)
