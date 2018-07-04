import torch
from models.pointnet_plus import PointNetPlusSegment
from data.shapenet import ShapeNetDataset

trainset = ShapeNetDataset('/opt/shapenetcore_partanno_segmentation_benchmark_v0', split='train&val')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
testset = ShapeNetDataset('/opt/shapenetcore_partanno_segmentation_benchmark_v0', split='test')
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True, num_workers=2)

net = PointNetPlusSegment(trainset.class_nums, trainset.category_nums)
for epcho in range(1, 60):
    net.fit(trainloader, epcho)
net.score(testloader)
