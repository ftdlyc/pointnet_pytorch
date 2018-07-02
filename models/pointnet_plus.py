import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from T_Net import TNet
from utils.sa_utils import farthest_point_sampling, group_points, ball_query


class PointNetPlusSAModule(nn.Module):

    def __init__(self, in_channels, mlp_specs, sample_point_nums, radius, group_point_nums, use_xyz=True):
        super(PointNetPlusSAModule, self).__init__()
        assert (type(mlp_specs) == type(group_point_nums) == type(radius) == list)
        assert (len(mlp_specs) == len(radius) == len(group_point_nums))
        assert (type(mlp_specs[0]) == list)

        if use_xyz:
            self.in_channels = in_channels + 3
        else:
            self.in_channels = in_channels
        self.sample_point_nums = sample_point_nums
        self.group_point_nums = group_point_nums
        self.radius = radius
        self.use_xyz = use_xyz
        self.mlps = nn.ModuleList()

        for i in range(0, len(radius)):
            layers = []
            in_channels = self.in_channels
            for out_channels in mlp_specs[i]:
                layers.append(nn.Conv2d(in_channels, out_channels, (1, 1)))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(True))
                in_channels = out_channels
            self.mlps.append(nn.Sequential(*layers))

    def forward(self, pc, features):
        new_pc = farthest_point_sampling(pc, self.sample_point_nums)
        new_feature_lists = []

        for i in range(0, len(self.radius)):
            group_idxs = ball_query(pc, new_pc, self.radius[i], self.group_point_nums[i])
            if features is None:
                xyz_features = group_points(pc, group_idxs.detach())
                new_features = xyz_features - new_pc.unsqueeze(-1)
            else:
                group_features = group_points(features, group_idxs.detach())
                if self.use_xyz:
                    xyz_features = group_points(pc, group_idxs.detach())
                    xyz_features -= new_pc.unsqueeze(-1)
                    new_features = torch.cat([xyz_features, group_features], dim=1)
                else:
                    new_features = group_features
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, (1, new_features.size(3)), stride=1)
            new_features = new_features.squeeze(-1)
            new_feature_lists.append(new_features)

        return new_pc, torch.cat(new_feature_lists, dim=1)


class PointNetPlusClassify(nn.Module):

    def __init__(self, class_nums, initial_weights=True, device_id=0):
        super(PointNetPlusClassify, self).__init__()

        self.device_id = device_id

        self.sa_layer1 = PointNetPlusSAModule(0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                              512, [0.1, 0.2, 0.4], [32, 64, 128], True)
        self.sa_layer2 = PointNetPlusSAModule(64 + 128 + 128, [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                              128, [0.2, 0.4, 0.8], [16, 32, 64], True)
        self.mlp_global = nn.Sequential(
            nn.Conv1d(128 + 256 + 256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.classify = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )

        if initial_weights:
            self.initialize_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters() )
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda()

    def forward(self, pc):
        pc, feature = self.sa_layer1(pc, None)
        pc, feature = self.sa_layer2(pc, feature)
        feature = self.mlp_global(feature)
        feature = F.max_pool1d(feature, feature.size(2), stride=1).squeeze(-1)
        outputs = self.classify(feature)

        return outputs

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

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch):
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 4 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 4))
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)

                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        return correct / total
