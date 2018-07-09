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


class PointNetClassify(nn.Module):

    def __init__(self, class_nums, use_cuda=None, device_id=0, initial_weights=True):
        super(PointNetClassify, self).__init__()

        self.class_nums = class_nums
        self.use_cuda = use_cuda
        self.device_id = device_id

        self.trans1 = TNet(3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.trans2 = TNet(64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(256, class_nums)
        )

        if initial_weights:
            self.initialize_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.t_reg_weight = 0.001
        self.optimizer = optim.Adam(self.parameters())
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        if use_cuda:
            self.cuda(device_id)

    def forward(self, x):
        x = self.trans1(x)
        x = self.mlp1(x)
        x = self.trans2(x)
        x = self.mlp2(x)
        x = F.max_pool1d(x, x.size(2), stride=1)
        x = x.view(x.size(0), -1)
        x = self.mlp3(x)

        return x

    def initialize_weights(self):
        for name, module in self.named_children():
            if name not in ['trans1', 'trans2']:
                for m in module:
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
        t = self.trans2.trans
        eye = torch.eye(64)
        if self.use_cuda:
            eye = eye.cuda(self.device_id)
        t_reg = torch.pow(
            torch.norm(
                eye.sub(torch.matmul(t, t.transpose(1, 2)))
            ), 2)
        return self.criterion(outputs, targets) + self.t_reg_weight * t_reg

    def fit(self, dataloader, epoch):
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if self.use_cuda:
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
                if self.use_cuda:
                    inputs = inputs.cuda(self.device_id)
                    targets = targets.cuda(self.device_id)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        return correct / total


class PointNetSegment(nn.Module):

    def __init__(self, class_nums, category_nums, use_cuda=None, device_id=None, initial_weights=True):
        super(PointNetSegment, self).__init__()

        self.class_nums = class_nums
        self.category_nums = category_nums
        self.use_cuda = use_cuda
        self.device_id = device_id

        self.trans1 = TNet(3)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.trans2 = TNet(64)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088 + category_nums, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, class_nums, 1)
        )

        if initial_weights:
            self.initialize_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.t_reg_weight = 0.001
        self.optimizer = optim.Adam(self.parameters())
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        if use_cuda:
            self.cuda(device_id)

    def forward(self, x, labels):
        x = self.trans1(x)
        x = self.mlp1(x)
        x = self.trans2(x)
        local_feature = x
        x = self.mlp2(x)
        x = F.max_pool1d(x, x.size(2), stride=1)
        gobal_feature = x.repeat([1, 1, local_feature.size(2)])
        index = labels.unsqueeze(1).repeat([1, local_feature.size(2)]).unsqueeze(1)
        one_hot = torch.zeros([local_feature.size(0), self.category_nums, local_feature.size(2)])
        if self.use_cuda:
            one_hot = one_hot.cuda(self.device_id)
        one_hot = one_hot.scatter_(1, index, 1)
        x = torch.cat([local_feature, gobal_feature, one_hot], 1)
        x = self.mlp3(x)

        return x

    def initialize_weights(self):
        for name, module in self.named_children():
            if name not in ['trans1', 'trans2']:
                for m in module:
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
        t = self.trans2.trans
        eye = torch.eye(64)
        if self.use_cuda:
            eye = eye.cuda(self.device_id)
        t_reg = torch.pow(
            torch.norm(
                eye.sub(torch.matmul(t, t.transpose(1, 2)))
            ), 2)
        return self.criterion(outputs, targets) + self.t_reg_weight * t_reg

    def fit(self, dataloader, epoch):
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, targets, labels) in enumerate(dataloader):
            if self.use_cuda:
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
                if self.use_cuda:
                    inputs = inputs.cuda(self.device_id)
                    targets = targets.cuda(self.device_id)
                    labels = labels.cuda(self.device_id)

                outputs = self(inputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

        return correct / total
