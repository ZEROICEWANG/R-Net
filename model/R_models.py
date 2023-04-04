import torch
import torch.nn as nn
from model.HolisticAttention import HA_r
from model.ResNet import B2_ResNet
import torchvision.models as models
import numpy as np
from model.dense_aggregation import dense_aggregation, Self_Attention
import time
import os
