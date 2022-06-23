import torch.nn as nn
import torch
from torch.autograd import Variable


class IOU_Loss(nn.Module):
    def __init__(self):
        super(IOU_Loss, self).__init__()

    def forward(self, x, y, weight=None):
        if weight is not None:
            x *= weight
            y *= weight
        Iand = torch.sum(x * y, dim=[2, 3])
        Ior = torch.sum(x + y, dim=[2, 3]) - Iand
        IoU = Iand / Ior
        IoU = torch.mean(IoU)
        return 1 - IoU


if __name__ == '__main__':
    ps = IOU_Loss()
    x = Variable(torch.ones(10, 1, 64, 128))
    y = Variable(torch.ones(10, 1, 64, 128))
    a = ps(x, y)
    print(a)
