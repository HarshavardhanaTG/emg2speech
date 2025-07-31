"""
A simple Euclidean RNN which takes vectorized covariance matrices as input.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RnnNet(nn.Module):
    def __init__(self, classifications, hidden, device, numLayers = 1):
        super(RnnNet, self).__init__()
        self.classifications = classifications
        self.hidden = hidden * (hidden - 1) // 2 + hidden
        self.device = device

        self.RNN = nn.GRU(input_size = 31 * 31, hidden_size = self.hidden, num_layers = numLayers, bidirectional = True, batch_first = True)

    
        self.cls = nn.Sequential(nn.Linear(self.hidden * 2, self.classifications))

    def forward(self, x, lengths):
        b, s, n , _ = x.shape
        x = x.reshape(b, s, -1)
        x = pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted = False)
        x, _ = self.RNN(x)
        x, _ = pad_packed_sequence(x, batch_first = True)
        x = self.cls(x)
        x = torch.nn.functional.log_softmax(x, dim = -1)
        
        return x.permute(1, 0, 2)
    
