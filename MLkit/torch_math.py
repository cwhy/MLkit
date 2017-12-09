import torch as th


class Add1(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, axis):
        s = list(X.shape)
        s[axis] += 1
        X1 = th.cuda.FloatTensor(*s)
        self.register_buffer("X1", X1)
        th.ones(*s, out=X1)
        slices = tuple(slice(None, -1, None)
                       if i == axis
                       else slice(None, None, None)
                       for i, _ in enumerate(s))
        X1.__setitem__(slices, X)
        return X1


add1 = Add1()
