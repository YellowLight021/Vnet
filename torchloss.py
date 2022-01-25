import torch
from torch.autograd.function import Function
from itertools import repeat
import numpy as np

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

class DiceLoss(Function):
    @staticmethod
    def forward(ctx,input, target):

        eps = 0.000001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
#       print(input)
        intersect = torch.dot(result_, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result_)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
        #     union, intersect, target_sum, result_sum, 2*IoU))
        out = 2*IoU
        ctx.save_for_backward(input, target,intersect,union)
        return out

    @staticmethod
    def backward(ctx,grad_output):
        input, target,intersect,union= ctx.saved_tensors
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input , None

def dice_loss(input, target):
    return DiceLoss()(input, target)

def dice_error(input, target):
    eps = 0.000001
    # eps=0.00001
    _, result_ = input.max(1)
    result_ = torch.squeeze(result_)
    if input.is_cuda:
        result = torch.cuda.FloatTensor(result_.size())
        target_ = torch.cuda.FloatTensor(target.size())
    else:
        result = torch.FloatTensor(result_.size())
        target_ = torch.FloatTensor(target.size())
    result.copy_(result_.data)
    target_.copy_(target.data)
    target = target_
    # print('reusult.shape:{}'.format(result.shape))
    # print('target.shape:{}'.format(target.shape))
    intersect = torch.dot(result, target)

    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum + 2*eps
    # print('intersect:{}'.format(intersect))
    # intersect = torch.max([eps, intersect])


    # the target volume can be empty - so we still want to
    # end up with a score of 1 if the result is 0/0
    IoU = intersect / union
#    print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#        union, intersect, target_sum, result_sum, 2*IoU))
    return 2*IoU
