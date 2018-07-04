import os
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

extension_ = load(name='extension_',
                  sources=['utils/wrap.cpp', 'utils/sampling.cu', 'utils/group_points.cu', 'utils/interpolate.cu'],
                  extra_include_paths=['/usr/local/cuda/include', os.path.join(os.getcwd(), 'utils', 'include')],
                  verbose=False)


class BallQuery(Function):
    """


    :param pc: (B, C, N)
           new_pc: (B, C, M)
           radius: R
           group_point_nums: K
    :return: group_idxs: (B, M, K)
    """

    @staticmethod
    def forward(ctx, pc, new_pc, radius, group_point_nums):
        xyz = pc.transpose(1, 2).contiguous()
        new_xyz = new_pc.transpose(1, 2).contiguous()
        group_idxs = extension_.ball_query_wrapper(xyz, new_xyz, radius, group_point_nums)

        return group_idxs

    @staticmethod
    def backward(ctx, grad_out=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupPoints(Function):
    """
    simpling points

    :param features: (B, C, N)
           group_idxs: (B, M, K)
    :return: out: (B, C, M, K)
    """

    @staticmethod
    def forward(ctx, features, group_idxs):
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
        idxs = extension_.farthest_point_sampling_wrapper(xyz, sample_point_nums)
        out = extension_.gather_points_wrapper(pc, idxs)

        ctx.save_for_backward(idxs, pc)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        idxs, pc = ctx.saved_tensors
        grad_inputs = extension_.gather_points_grad_wrapper(grad_outputs.data.contiguous(), idxs, pc.size(2))

        return grad_inputs, None


farthest_point_sampling = FarthestPointSampling.apply


class Interpolate(Function):
    """
    interploate points

    :param features: (B, C, M)
           unknown_pc: (B, 3, N)
           known_pc: (B, 3, M)
           nn_points: K
    :return: out: (B, C, N)
    """

    @staticmethod
    def forward(ctx, features, unknown_pc, known_pc, nn_points):
        unknown = unknown_pc.transpose(1, 2).contiguous()
        known = known_pc.transpose(1, 2).contiguous()
        idxs, dists = extension_.knn_wrapper(unknown, known, nn_points)
        dists_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dists_recip, dim=2, keepdim=True)
        weights = dists_recip / norm
        out = extension_.interpolate_wrapper(features, idxs, weights)
        ctx.save_for_backward(known, idxs, weights)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        known, idxs, weights = ctx.saved_tensors
        grad_inputs = extension_.interpolate_grad_wrapper(grad_out.contiguous(), idxs, weights, known.size(1))
        return grad_inputs, None, None, None


interpolate = Interpolate.apply
