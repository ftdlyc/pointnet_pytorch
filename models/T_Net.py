import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TNet(nn.Module):

    def __init__(self, point_nums, K, initial_weights=True):
        super(TNet, self).__init__()

        self.point_nums = point_nums
        self.K = K
        self.mlp1 = nn.Sequential(
            nn.Conv1d(K, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.max_pool = nn.MaxPool1d(point_nums, stride=1)
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, K * K)
        )

        if initial_weights:
            self.initialize_weights()

    def forward(self, x):
        t = self.mlp1(x)
        t = self.max_pool(t)
        t = t.view(t.size(0), -1)
        t = self.mlp2(t)
        t = t.view(x.size(0), self.K, self.K)
        self.trans = t

        x = x.transpose(1, 2)
        x = torch.bmm(x, t)
        x = x.transpose(2, 1)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.mlp2[-1].bias.data = torch.Tensor(np.identity(self.K).flatten())
