import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class WBCE():
    def __init__(self):
        self.bce = torch.nn.BCELoss(reduction='none')

    def __call__(self, x, y, weight=None):
        loss = self.bce(x, y)
        if weight is not None:
            loss = loss * weight
            loss = torch.mean(loss, dim=[2, 3]) / torch.mean(weight, dim=[2, 3])
            loss = torch.mean(loss)
        else:
            loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    ps = WBCE()
    nce = torch.nn.BCELoss()
    x = Variable(torch.ones(10, 1, 64, 128))
    y = Variable(torch.zeros(10, 1, 64, 128))
    a = ps(x, y)
    print(a)
    b = F.binary_cross_entropy(x, y)
    print(b)
