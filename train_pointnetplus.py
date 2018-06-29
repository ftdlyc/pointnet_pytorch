import torch
from models.pointnet_plus import PointNetPlusClassify
from data.modelnet import ModelNetDataset

trainset = ModelNetDataset('/opt/modelnet40_normal_resampled', train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
testset = ModelNetDataset('/opt/modelnet40_normal_resampled', train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

net = PointNetPlusClassify(trainset.class_nums)
for epcho in range(1, 2):
    net.fit(trainloader, epcho)
net.score(testloader)
