import torch
import time

from MLkit import torch_math
x = torch.rand(50, 50)
y = torch.zeros(50, 50)
x = x.cuda()
y = y.cuda()

z = torch_math.add1.cuda()(x, 1)


