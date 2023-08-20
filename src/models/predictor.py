import torch
from torch import nn as nn
import argparse
import torch.nn.functional as F
import torch.optim as optim
import time

class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)