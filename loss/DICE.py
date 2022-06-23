import torch.nn as nn
import torch
from torch.autograd import Variable


class DICE():
    def __init__(self):
        pass

    def __call__(self, x, y, weight=None):
        if weight is not None:
            intersection = torch.sum(x * y * weight, dim=[-1, -2])
            loss = (2. * intersection) / (torch.sum(x * weight, dim=[-1, -2]) + torch.sum(y * weight, dim=[-1, -2]))
        else:
            intersection = torch.sum(x * y, dim=[-1, -2])
            loss = (2. * intersection) / (torch.sum(x, dim=[-1, -2]) + torch.sum(y, dim=[-1, -2]))
        return 1 - torch.mean(loss)


if __name__ == '__main__':
    ps = DICE()
    x = Variable(torch.zeros(10, 1, 64, 128))
    y = Variable(torch.ones(10, 1, 64, 128))
    a = ps(x, y)
    print(a)
