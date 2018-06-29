import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

extension_ = load(name='extension_',
                  sources=['utils/wrap.cpp', 'utils/sampling.cu', 'utils/group_points.cu'],
                  extra_include_paths=['/usr/local/cuda/include', os.path.join(os.getcwd(), 'utils', 'include')],
                  verbose=False)


class GroupPoints(Function):
    """
    simpling points

    :param features: (B, C', N)
           xyz: (B, C, N)
           new_xyz: (B, C, M)
           radius: R
           group_point_nums: K
    :return: out: (B, C, M)
    """
    @staticmethod
    def forward(ctx, features, pc, new_pc, radius, group_point_nums):
        xyz = pc.transpose(1, 2).contiguous()
        new_xyz = new_pc.transpose(1, 2).contiguous()
        group_idxs = extension_.ball_query_wraper(xyz, new_xyz, radius, group_point_nums)
        out = extension_.group_points_wrapper(features, group_idxs)

        ctx.save_for_backward(features, group_idxs)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, group_idxs = ctx.saved_tensors
        n = features.size(2)
        grad_inputs = extension_.group_points_grad_wrapper(grad_out.data.contiguous(), group_idxs, n)

        return grad_inputs, None, None, None, None


group_points = GroupPoints.apply


class FarthestPointSampling(Function):
    """
    simpling points

    :param pc: (B, C, N)
           sample_point_nums: M
    :return: out: (B, C, M)
    """

    @staticmethod
    def forward(ctx, pc, sample_point_nums):
        xyz = pc.transpose(1, 2).contiguous()
        idxs = extension_.farthest_point_sampling_wraper(xyz, sample_point_nums)
        out = extension_.gather_points_wrapper(pc, idxs)

        ctx.save_for_backward(idxs, pc)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        idxs, pc = ctx.saved_tensors
        print(grad_outputs.size())
        grad_inputs = extension_.gather_points_grad_wrapper(grad_outputs.data.contiguous(), idxs, xyz.size(2))

        return grad_inputs, None


farthest_point_sampling = FarthestPointSampling.apply
