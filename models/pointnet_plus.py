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
from utils.sa_utils import farthest_point_sampling, group_points, ball_query, interpolate


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


class PointNetPlusFPModule(nn.Module):

    def __init__(self, in_channels, mlp_specs, nn_points=3):
        super(PointNetPlusFPModule, self).__init__()
        self.nn_points = nn_points
        self.in_channels = in_channels
        self.mlp_specs = mlp_specs

        layers = []
        for out_channels in mlp_specs:
            layers.append(nn.Conv1d(in_channels, out_channels, 1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(True))
            in_channels = out_channels
        self.mlps = nn.Sequential(*layers)

    def forward(self, unknown_pc, known_pc, unknow_features, known_features):
        interpolated_features = interpolate(known_features, unknown_pc.detach(), known_pc.detach(), self.nn_points)
        if unknow_features is not None:
            new_features = torch.cat([interpolated_features, unknow_features], dim=1)
        else:
            new_features = interpolated_features
        new_features = self.mlps(new_features)

        return new_features


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
        self.optimizer = optim.Adam(self.parameters())
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda()

    def forward(self, pc):
        pc, features = self.sa_layer1(pc, None)
        pc, features = self.sa_layer2(pc, features)
        features = self.mlp_global(features)
        features = F.max_pool1d(features, features.size(2), stride=1).squeeze(-1)
        outputs = self.classify(features)

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


class PointNetPlusSegment(nn.Module):

    def __init__(self, class_nums, category_nums, initial_weights=True, device_id=0):
        super(PointNetPlusSegment, self).__init__()

        self.class_nums = class_nums
        self.category_nums = category_nums
        self.device_id = device_id

        self.sa_layer1 = PointNetPlusSAModule(0, [[32, 32, 64], [64, 64, 128], [64, 96, 128]],
                                              512, [0.04, 0.08, 0.16], [32, 64, 128], True)
        self.sa_layer2 = PointNetPlusSAModule(64 + 128 + 128, [[64, 64, 128], [128, 128, 256], [128, 128, 256]],
                                              128, [0.08, 0.16, 0.32], [16, 32, 64], True)
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
        self.mlp_local = nn.Sequential(
            nn.Conv1d(1024 + 128 + 256 + 256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
        )
        self.fp_layers1 = PointNetPlusFPModule(256 + 64 + 128 + 128, [256, 256])
        self.fp_layers2 = PointNetPlusFPModule(256 + 3 + category_nums, [128, 128])
        self.mlp_seg = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, class_nums, 1),
        )

        if initial_weights:
            self.initialize_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda()

    def forward(self, pc, labels):
        pc_sa1, features_sa1 = self.sa_layer1(pc, None)
        pc_sa2, features_sa2 = self.sa_layer2(pc_sa1, features_sa1)
        global_features = self.mlp_global(features_sa2)
        global_features = F.max_pool1d(global_features, global_features.size(2), stride=1)
        global_features = global_features.repeat([1, 1, features_sa2.size(2)])
        local_featrures = self.mlp_local(torch.cat([features_sa2, global_features], dim=1))
        features_fp1 = self.fp_layers1(pc_sa1, pc_sa2, features_sa1, local_featrures)
        index = labels.unsqueeze(1).repeat([1, pc.size(2)]).unsqueeze(1)
        one_hot = torch.zeros([pc.size(0), self.category_nums, pc.size(2)]).cuda(self.device_id)
        one_hot = one_hot.scatter_(1, index, 1)
        features_fp2 = self.fp_layers2(pc, pc_sa1, torch.cat([pc, one_hot], dim=1), features_fp1)
        out = self.mlp_seg(features_fp2)

        return out

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

        for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            labels = labels.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs, labels)
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
            for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)

                outputs = self(inputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        return correct / total
