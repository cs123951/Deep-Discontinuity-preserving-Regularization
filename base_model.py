import torch
import torch.nn as nn

import image_func


# 主要是为了定义loss
class SeqModel(nn.Module):
    def __init__(self, device_ids, input_device, output_device, img_size, n_steps):
        super().__init__()
        self.device_ids = device_ids
        self.input_device = input_device
        self.output_device = output_device
        self.img_size = img_size
        self.n_steps = n_steps
        self.dim = len(img_size)
        self.batch_regular_grid = image_func.create_batch_regular_grid(
            1, img_size, device=self.output_device
        )

    def get_class(self):
        return "SeqModel"

    def loss(self, loss_function, disp_b3xyz, batch_fix, batch_mov):
        if loss_function is None:
            return torch.tensor(0.0, device=self.output_device), [
                torch.tensor(0.0, device=self.output_device)
            ]
        loss, loss_part = loss_function(
            disp_b3xyz, self.batch_regular_grid, batch_fix, batch_mov
        )
        return loss, loss_part


class SegModel(nn.Module):
    def __init__(self, device_ids, input_device, output_device, img_size, n_steps):
        super().__init__()
        self.device_ids = device_ids
        self.input_device = input_device
        self.output_device = output_device
        self.img_size = img_size
        self.n_steps = n_steps
        self.dim = len(img_size)
        self.batch_regular_grid = image_func.create_batch_regular_grid(
            1, img_size, device=self.output_device
        )

    def get_class(self):
        return "SegModel"

    def loss(
        self, loss_function, disp_b3xyz, batch_fix, batch_mov, gt_ed, gt_es, es_idx
    ):
        if loss_function is None:
            return torch.tensor(0.0, device=self.output_device), [
                torch.tensor(0.0, device=self.output_device)
            ]

        loss, loss_part = loss_function(
            disp_b3xyz,
            self.batch_regular_grid,
            batch_fix,
            batch_mov,
            gt_ed,
            gt_es,
            es_idx,
        )
        return loss, loss_part
