#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import image_func, helpers


def get_mask_edges(data):
    data_x = abs(data[:, :, :-1] - data[:, :, 1:])
    data_y = abs(data[:, :, :, :-1] - data[:, :, :, 1:])
    data_z = abs(data[..., :-1] - data[..., 1:])
    data_x = F.pad(data_x, (0, 0, 0, 0, 0, 1), mode="replicate")
    data_y = F.pad(data_y, (0, 0, 0, 1, 0, 0), mode="replicate")
    data_z = F.pad(data_z, (0, 1, 0, 0, 0, 0), mode="replicate")
    data_edge = data_x + data_y + data_z
    data_edge[data_edge > 0] = 1
    # [1,1,x,y,z]
    return 1 - data_edge


def diceloss(target, predict):
    # value is 1, 2, 3
    # dice=2|X ∩ Y|/(|X|+|Y|)
    dice_sum = 0
    for label_value in [1, 2, 3]:
        target01 = torch.zeros(target.shape).to(target.device)
        predict01 = torch.zeros(predict.shape).to(predict.device)
        target01[target == label_value] = 1
        predict01[predict == label_value] = 1
        dice_sum += (
            2
            * (torch.sum(target01 * predict01))
            / (torch.sum(target01) + torch.sum(predict01))
        )
    dice_sum /= 3
    return -dice_sum


# *************************#
#  groupwise metrics        #
# *************************#
class SEGREG(nn.Module):
    """
    loss for a sequence
    """

    def __init__(self, lamb_reg, lamb_dice, dsim_name, reg_name, mask_name="mse"):
        """

        :param lamb:
        :param dsim_kargs:
        :param reg_kargs:
        :param frame: lag:Lagrangian, eul: Eulerian
        """
        super().__init__()
        self.lamb_reg = lamb_reg
        self.lamb_dice = lamb_dice
        self.dsim_module = helpers.str_to_class("modules.losses", dsim_name)()
        self.reg_module = helpers.str_to_class("modules.losses", reg_name)()
        self.mask_name = mask_name

    def forward(
        self,
        batch_disp,
        batch_regular_grid,
        batch_fix,
        batch_mov,
        gt_fixed,
        gt_moving,
        fixed_idx,
    ):
        loss_dsim = 0
        loss_reg = 0
        weight = 1

        batch_grid = batch_regular_grid.repeat(len(batch_disp), 1, 1, 1, 1)

        gt_fixed = gt_fixed.to(batch_disp.device)
        gt_moving = gt_moving.to(batch_disp.device)
        data_edge = get_mask_edges(gt_moving)  # 选择浮动图像对应的mask。
        # data_edge = get_mask_edges(gt_fixed) # 选择固定图像对应的mask。

        batch_warped = image_func.grid_sample_without_grid(
            batch_mov, batch_disp, batch_grid
        )
        loss_dsim += weight * self.dsim_module(batch_warped, batch_fix)
        loss_reg += weight * self.reg_module(batch_disp, data_edge)

        # dice loss. batch_disp的长度和batch_mov的长度是一样的，而fixed_idx是image list的长度，在开头多了一个moving image。
        fixed_disp = batch_disp[int(fixed_idx) - 1].unsqueeze(0)  # 选择某一个时间点对应的位移场。
        # print("shape: ", es_disp.shape, gt_es.shape, gt_ed.shape)
        gt_warped = (
            image_func.grid_sample_without_grid(
                gt_moving.to(batch_disp.device),
                fixed_disp,
                batch_regular_grid,
                interp_mode="nearest",
            )
            .int()
            .squeeze(0)
            .squeeze(0)
        )
        # gt_es_onehot = F.one_hot(warped_gt_es.to(torch.int64), num_classes=4)
        # gt_ed_onehot = F.one_hot(gt_ed.to(torch.int64), num_classes=4) # [1, 1, 128, 128, 16, 4]
        # loss_dice = torch.mean(torch.abs(gt_es_onehot-gt_ed_onehot))
        if self.mask_name == "mse":
            abs_results = gt_warped - gt_fixed
            abs_results[abs_results != 0] = 1
            loss_dice = torch.mean(abs_results)
        elif self.mask_name == "dice":
            loss_dice = diceloss(gt_warped, gt_fixed)

        # loss = loss_dsim + self.lamb_reg * loss_reg + (1e-4)*loss_seg
        loss = loss_dsim + self.lamb_reg * loss_reg + self.lamb_dice * loss_dice
        # print(loss, loss_dsim, loss_reg, loss_dice)
        # print("loss %f, %f, %f, %f, %f" %(loss_dsim, loss_reg, loss_dice, gt_moving.sum(), data_edge.sum()))

        return loss, torch.stack(
            [loss_dsim.detach(), loss_reg.detach(), loss_dice.detach()]
        )


# *************************#
#  pairwise metrics        #
# *************************#


class PAIR(nn.Module):
    """
    loss for a sequence
    """

    def __init__(self, lamb_reg, dsim_name, reg_name):
        super(PAIR, self).__init__()
        self.lamb_reg = lamb_reg
        self.dsim_module = helpers.str_to_class("modules.losses", dsim_name)()
        self.reg_module = helpers.str_to_class("modules.losses", reg_name)()

    def forward(self, disp_seq, batch_regular_grid, fix, mov, loss_kl=0):
        """

        :param disp_seq: [batch, 3, x, y, z]
        :param batch_grid: [1, x, y, z, 3]
        :param fix: [batch, 1, x, y, z]
        :param mov: [batch, 1, x, y, z]
        :return:
        """
        loss_dsim = 0
        loss_reg = 0
        if loss_kl == 0:
            loss_kl = torch.tensor([0.0])

        warped = image_func.grid_sample_without_grid(mov, disp_seq, batch_regular_grid)
        loss_dsim += self.dsim_module(warped, fix)
        loss_reg += self.reg_module(disp_seq)
        loss = loss_dsim + self.lamb_reg * loss_reg
        return loss, torch.stack([loss_dsim.detach(), loss_reg.detach()])


# *************************#
#  dissimilarity metrics   #
# *************************#


class MSE(nn.Module):
    def __init__(self, win=None, eps=1e-5):
        super(MSE, self).__init__()
        self.win = win
        self.eps = eps

    def forward(self, I, J):
        ndims = I.ndimension() - 2
        assert ndims in [1, 2, 3], (
            "volumes should be 1 to 3 dimensions. found: %d" % ndims
        )

        # compute CC squares
        IJ = (I - J) ** 2

        return torch.mean(IJ)


# *************************#
#  regularization metrics  #
# *************************#


class Grad(nn.Module):
    """
    N-D gradient loss
    """

    def __init__(self, penalty="l2"):
        super(Grad, self).__init__()
        self.penalty = penalty

    def _diffs(self, y):
        # [batch, 3, x, y, z] or [batch, 2, x, y]
        self.dim = len(y.shape) - 2
        if self.dim == 2:
            graident_x = y[:, :, 1:] - y[:, :, :-1]
            graident_x = F.pad(graident_x, (0, 0, 0, 1), mode="replicate").contiguous()
            graident_y = y[..., 1:] - y[..., :-1]
            graident_y = F.pad(graident_y, (0, 1, 0, 0), mode="replicate").contiguous()
            return [graident_x, graident_y]
        elif self.dim == 3:
            graident_x = y[:, :, 1:] - y[:, :, :-1]
            graident_x = F.pad(
                graident_x, (0, 0, 0, 0, 0, 1), mode="replicate"
            ).contiguous()
            graident_y = y[:, :, :, 1:] - y[:, :, :, :-1]
            graident_y = F.pad(
                graident_y, (0, 0, 0, 1, 0, 0), mode="replicate"
            ).contiguous()
            graident_z = y[..., 1:] - y[..., :-1]
            graident_z = F.pad(
                graident_z, (0, 1, 0, 0, 0, 0), mode="replicate"
            ).contiguous()
            return [graident_x, graident_y, graident_z]

    def forward(self, pred, mask_edges=None):
        self.ndims = pred.ndimension() - 2
        df = []
        # f is [3,x,y,z]
        for f in self._diffs(pred):
            if self.penalty == "l1":
                if mask_edges is None:
                    df.append((f).abs().mean() / self.ndims)
                else:
                    df.append((mask_edges * f).abs().mean() / self.ndims)
            else:
                assert self.penalty == "l2", (
                    "penalty can only be l1 or l2. Got: %s" % self.penalty
                )
                if mask_edges is None:
                    df.append((f * f).mean() / self.ndims)
                else:
                    df.append((f * f * mask_edges).mean() / self.ndims)
        return sum(df)  # [delta_x, delta_y, delta_z]


class Laplacian(nn.Module):
    """
    N-D gradient loss
    """

    def __init__(self, penalty="l2"):
        super().__init__()
        self.penalty = penalty

    def _diffs(self, y):
        # [batch, 3, x, y, z] or [batch, 2, x, y]
        self.dim = len(y.shape) - 2
        if self.dim == 2:
            gradient2_x = y[:, :, 2:] + y[:, :, :-2] - 2 * y[:, :, 1:-1]
            gradient2_y = y[..., 2:] + y[..., :-2] - 2 * y[..., 1:-1]
            gradient2_x = F.pad(
                gradient2_x, (0, 0, 1, 1), mode="replicate"
            ).contiguous()
            gradient2_y = F.pad(
                gradient2_y, (1, 1, 0, 0), mode="replicate"
            ).contiguous()
            return [gradient2_x, gradient2_y]

        elif self.dim == 3:
            gradient2_x = y[:, :, 2:] + y[:, :, :-2] - 2 * y[:, :, 1:-1]
            gradient2_y = y[:, :, :, 2:] + y[:, :, :, :-2] - 2 * y[:, :, :, 1:-1]
            gradient2_z = y[..., 2:] + y[..., :-2] - 2 * y[..., 1:-1]
            gradient2_x = F.pad(
                gradient2_x, (0, 0, 0, 0, 1, 1), mode="replicate"
            ).contiguous()
            gradient2_y = F.pad(
                gradient2_y, (0, 0, 1, 1, 0, 0), mode="replicate"
            ).contiguous()
            gradient2_z = F.pad(
                gradient2_z, (1, 1, 0, 0, 0, 0), mode="replicate"
            ).contiguous()
            return [gradient2_x, gradient2_y, gradient2_z]

    def forward(self, pred, mask_edges=None):
        self.ndims = pred.ndimension() - 2
        df = []
        # f is [3,x,y,z]
        for f in self._diffs(pred):
            if self.penalty == "l1":
                if mask_edges is None:
                    df.append((mask_edges * f).abs().mean() / self.ndims)
                else:
                    df.append((f).abs().mean() / self.ndims)
            else:
                if mask_edges is None:
                    df.append((mask_edges * f * f).mean() / self.ndims)
                else:
                    df.append((f * f).mean() / self.ndims)

        return sum(df)  # [delta_x, delta_y, delta_z]
