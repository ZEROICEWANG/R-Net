import torch.nn as nn
import numpy as np



class HA_r(nn.Module):
    def __init__(self, channels):
        super(HA_r, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, attention, x):
        x = self.conv(x * attention) + x
        return x


if __name__ == '__main__':
    gaussian_kernel = np.float32(gkern(31, 4))
    gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
    print('get')
