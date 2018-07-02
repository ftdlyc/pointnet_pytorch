import torch
from models.pointnet import PointNetClassify
from data.modelnet import ModelNetDataset

trainset = ModelNetDataset('/opt/modelnet40_normal_resampled', train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = ModelNetDataset('/opt/modelnet40_normal_resampled', train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)

net = PointNetClassify(trainset.point_nums, trainset.class_nums, use_cuda=True)
for epcho in range(1, 60):
    net.fit(trainloader, epcho)
net.score(testloader)
