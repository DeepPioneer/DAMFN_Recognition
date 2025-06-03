from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F
"""
Adapted from https://github.com/fungtion/DSN/blob/master/functions.py
"""
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


import torch
import torch.nn as nn

class CMD(nn.Module):
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=5):
        mx1 = torch.mean(x1, dim=0)
        mx2 = torch.mean(x2, dim=0)

        sx1 = x1 - mx1
        sx2 = x2 - mx2

        # First-order moment (mean)
        loss = self.matchnorm(mx1, mx2)

        # Higher-order moments
        for k in range(2, n_moments + 1):
            loss = loss + self.scm(sx1, sx2, k)
        return loss

    def matchnorm(self, x1, x2):
        eps = 1e-6
        power = (x1 - x2) ** 2
        summed = torch.sum(power)
        return torch.sqrt(summed + eps)  # avoids sqrt(0) and keeps it differentiable

    def scm(self, sx1, sx2, k):
        moment1 = torch.mean(sx1 ** k, dim=0)
        moment2 = torch.mean(sx2 ** k, dim=0)
        return self.matchnorm(moment1, moment2)


# Distillation loss
class DistillationLoss(nn.Module):
    def __init__(self, T=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.T = T  # Temperature for softening the probabilities
        self.alpha = alpha  # Balance between soft and hard targets

    def forward(self, student_output, teacher_output, student_labels):
        # Soft Target Loss
        soft_loss = F.kl_div(F.log_softmax(student_output / self.T, dim=1),
                             F.softmax(teacher_output / self.T, dim=1), reduction='batchmean') * (self.T ** 2)

        # Hard Target Loss (cross-entropy)
        hard_loss = F.cross_entropy(student_output, student_labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss