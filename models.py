import torch
import numpy as np
import torch.nn as nn

import image_func
from base_model import SeqModel, SegModel
from layers import Conv, conv_disp, RNN_block
from transformer import (
    get_3DReg_config,
    Transformer,
    DecoderCup,
    RegistrationHead,
)


# formal CRNet
class CRNet(SeqModel):
    def __init__(
        self,
        device_ids,
        input_device,
        output_device,
        img_size,
        num_layers,
        enc_nf,
        hidden_nf,
        n_steps=0,
    ):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps)

        self.rnn_device = output_device
        self.disp_device = output_device

        self.scale_factor = 4
        RNN_size = [img_size[i] // self.scale_factor for i in range(self.dim)]

        self.cell_params = {}

        self.scale_factor = [self.scale_factor] * self.dim

        self.enc_features1 = Conv(2, enc_nf[0], down=True, dim=self.dim, bn=True).to(
            self.input_device
        )
        self.enc_features2 = Conv(
            enc_nf[0], enc_nf[1], down=True, dim=self.dim, bn=True
        ).to(self.input_device)

        self.multiRNN = RNN_block(
            img_size=RNN_size,
            input_dim=enc_nf[-1],
            hidden_nf=hidden_nf,
            num_layers=num_layers,
        ).to(self.rnn_device)

        self.dec_features2 = Conv(
            enc_nf[1] * 2, enc_nf[1], up=True, dim=self.dim, bn=True
        ).to(self.input_device)
        self.dec_features1 = Conv(
            enc_nf[1] + enc_nf[0], enc_nf[1], up=True, dim=self.dim, bn=True
        ).to(self.input_device)

        self.outconv3 = conv_disp(enc_nf[1], kernel_size=3, dim=self.dim).to(
            self.disp_device
        )

    def forward(self, data, loss_function=None):
        """
        img_tb1xyz: seq * [batch, channel, x, y, z]
        """
        img_tb1xyz = [data[i].to(self.input_device) for i in range(len(data))]
        batch_fix = torch.cat(img_tb1xyz[1:], dim=0)
        batch_mov = img_tb1xyz[0].repeat(len(batch_fix), 1, 1, 1, 1)
        batch_seq = torch.cat([batch_fix, batch_mov], dim=1)  # [batch/t, 2, x,y,z]
        num_pairs = len(img_tb1xyz) - 1

        # split_list = [len(img_tb1xyz[i]) for i in range(num_pairs)]
        # features: list of t * [batch_t, channel, x//4, y//4, z//4]

        enc_feat1 = self.enc_features1(batch_seq)
        enc_feat2 = self.enc_features2(enc_feat1)  # batch, c, x, y, z
        rnn_feat = [enc_feat2[t].unsqueeze(0) for t in range(num_pairs)]
        rnn_feat = self.multiRNN(rnn_feat)  # t*[batch_t, channel, x, y, z]

        feat_bcxyz = torch.cat(
            [rnn_feat[i] for i in range(len(rnn_feat))], dim=0
        )  # channel is enc_nf[1]
        dec_feat2 = torch.cat([enc_feat2, feat_bcxyz], dim=1)  # channel is enc_nf[1]*2
        dec_feat1 = self.dec_features2(dec_feat2)  # enc_nf[1]
        dec_feat1 = torch.cat([enc_feat1, dec_feat1], dim=1)  # enc_nf[0]+enc_nf[1]
        feat_bcxyz = self.dec_features1(dec_feat1)

        disp_b3xyz = self.outconv3(feat_bcxyz)

        # batch只为1
        batch_fix = torch.cat(img_tb1xyz[1:], dim=0)
        batch_mov = img_tb1xyz[0].repeat(len(batch_fix), 1, 1, 1, 1)

        loss, loss_part = self.loss(
            loss_function, disp_b3xyz, batch_fix, batch_mov
        )  # 这里的loss是pair loss，所以要传递batch
        # (seq_len-1)*[batch, 3, x, y, z]
        return disp_b3xyz, loss, loss_part


class SegCRNet(SegModel):
    def __init__(
        self,
        device_ids,
        input_device,
        output_device,
        img_size,
        num_layers,
        enc_nf,
        hidden_nf,
        n_steps=0,
    ):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps)

        self.rnn_device = output_device
        self.disp_device = output_device

        self.scale_factor = 4
        RNN_size = [img_size[i] // self.scale_factor for i in range(self.dim)]

        self.scale_factor = [self.scale_factor] * self.dim

        self.enc_features1 = Conv(2, enc_nf[0], down=True, dim=self.dim, bn=True).to(
            self.input_device
        )
        self.enc_features2 = Conv(
            enc_nf[0], enc_nf[1], down=True, dim=self.dim, bn=True
        ).to(self.input_device)

        self.multiRNN = RNN_block(
            img_size=RNN_size,
            input_dim=enc_nf[-1],
            hidden_nf=hidden_nf,
            num_layers=num_layers,
        ).to(self.rnn_device)

        self.dec_features2 = Conv(
            enc_nf[1] * 2, enc_nf[1], up=True, dim=self.dim, bn=True
        ).to(self.input_device)
        self.dec_features1 = Conv(
            enc_nf[1] + enc_nf[0], enc_nf[1], up=True, dim=self.dim, bn=True
        ).to(self.input_device)

        self.outconv3 = conv_disp(enc_nf[1], kernel_size=3, dim=self.dim).to(
            self.disp_device
        )

    def forward(self, data, loss_function=None):
        """
        img_tb1xyz: seq * [batch, channel, x, y, z]
        """
        img_tb1xyz, gt_fixed, gt_moving, fixed_idx = data[0], data[1], data[2], data[3]
        img_tb1xyz = [
            img_tb1xyz[i].to(self.input_device) for i in range(len(img_tb1xyz))
        ]
        batch_fix = torch.cat(img_tb1xyz[1:], dim=0)
        batch_mov = img_tb1xyz[0].repeat(len(batch_fix), 1, 1, 1, 1)
        batch_seq = torch.cat([batch_fix, batch_mov], dim=1)  # [batch/t, 2, x,y,z]
        num_pairs = len(img_tb1xyz) - 1

        # split_list = [len(img_tb1xyz[i]) for i in range(num_pairs)]
        # features: list of t * [batch_t, channel, x//4, y//4, z//4]

        enc_feat1 = self.enc_features1(batch_seq)
        enc_feat2 = self.enc_features2(enc_feat1)  # batch, c, x, y, z
        rnn_feat = [enc_feat2[t].unsqueeze(0) for t in range(num_pairs)]
        rnn_feat = self.multiRNN(rnn_feat)  # t*[batch_t, channel, x, y, z]

        feat_bcxyz = torch.cat(
            [rnn_feat[i] for i in range(len(rnn_feat))], dim=0
        )  # channel is enc_nf[1]
        dec_feat2 = torch.cat([enc_feat2, feat_bcxyz], dim=1)  # channel is enc_nf[1]*2
        dec_feat1 = self.dec_features2(dec_feat2)  # enc_nf[1]
        dec_feat1 = torch.cat([enc_feat1, dec_feat1], dim=1)  # enc_nf[0]+enc_nf[1]
        feat_bcxyz = self.dec_features1(dec_feat1)

        disp_b3xyz = self.outconv3(feat_bcxyz)

        # batch只为1
        batch_fix = torch.cat(img_tb1xyz[1:], dim=0)
        batch_mov = img_tb1xyz[0].repeat(len(batch_fix), 1, 1, 1, 1)

        loss, loss_part = self.loss(
            loss_function,
            disp_b3xyz,
            batch_fix,
            batch_mov,
            gt_fixed,
            gt_moving,
            fixed_idx,
        )
        # (seq_len-1)*[batch, 3, x, y, z]
        return disp_b3xyz, loss, loss_part


# https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/networks.py
class unet_core(nn.Module):
    def __init__(self, img_size, enc_nf=[16, 32, 32], dec_nf=[32, 32, 32, 32, 16, 16]):
        super().__init__()

        # fixed setting
        max_pool = 2
        nb_conv_per_level = 1
        in_channels = 2  # the number of images

        self.img_size = img_size
        self.dim = len(img_size)
        self.half_res = half_res = False  # Skip the last decoder upsampling.

        nb_dec_convs = len(enc_nf)  # 3
        final_convs = dec_nf[nb_dec_convs:]  # the last 3 elements of dec_nf
        dec_nf = dec_nf[:nb_dec_convs]  # [32, 32, 32]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1  # 4

        max_pool = [max_pool] * self.dim
        max_pool = tuple(max_pool)
        MaxPooling = getattr(nn, "MaxPool%dd" % self.dim)
        self.pooling = [MaxPooling(max_pool) for s in range(nb_dec_convs)]
        self.upsampling = [
            nn.Upsample(scale_factor=max_pool, mode="nearest")
            for s in range(nb_dec_convs)
        ]

        prev_nf = in_channels
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        # 3 down loops
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for ith_conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + ith_conv]
                convs.append(Conv(prev_nf, nf, dim=self.dim))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        encoder_nfs = np.flip(encoder_nfs)
        # 3 up levels
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for ith_conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + ith_conv]
                convs.append(Conv(prev_nf, nf, dim=self.dim))
                prev_nf = nf
            self.decoder.append(convs)
            # level < 2
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(Conv(prev_nf, nf, dim=self.dim))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VoxelMorph(SeqModel):
    def __init__(self, device_ids, input_device, output_device, img_size, n_steps=0):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps)

        self.unet_model = unet_core(img_size).to(input_device)

        self.flow = conv_disp(16, 3).to(output_device)

    def forward(self, data, loss_function=None):
        img_tb1xyz = [data[i].to(self.input_device) for i in range(len(data))]

        batch_fix, batch_mov = image_func.pick_imgpair_first_mov_others_fix(img_tb1xyz)

        # change role
        batch_fix = batch_fix.to(self.input_device)
        batch_mov = batch_mov.to(self.input_device)

        x = torch.cat([batch_fix, batch_mov], 1)  # [batch, 2, x, y, z]

        # concatenate inputs and propagate unet
        x = self.unet_model(x)
        flow_field = self.flow(x.to(self.output_device))

        # resize flow for integration
        disp_b3xyz = flow_field  # # b,3,x,y,z
        if self.n_steps > 0:
            disp_b3xyz = image_func.integrate_displacement(
                disp_b3xyz, grid=self.batch_regular_grid, n_steps=self.n_steps
            )

        if self.output_device != self.input_device:
            batch_fix = batch_fix.to(self.output_device)
            batch_mov = batch_mov.to(self.output_device)

        if loss_function is None:
            loss = 0
            loss_part = 0
        else:
            loss, loss_part = self.loss(loss_function, disp_b3xyz, batch_fix, batch_mov)

        return disp_b3xyz, loss, loss_part


class SegVoxelMorph(SegModel):
    def __init__(self, device_ids, input_device, output_device, img_size, n_steps=0):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps)

        self.unet_model = unet_core(img_size).to(input_device)

        self.flow = conv_disp(16, 3).to(output_device)

    def forward(self, data, loss_function=None):
        img_tb1xyz, gt_fixed, gt_moving, fixed_idx = data[0], data[1], data[2], data[3]
        img_tb1xyz = [
            img_tb1xyz[i].to(self.input_device) for i in range(len(img_tb1xyz))
        ]

        batch_fix, batch_mov = image_func.pick_imgpair_first_mov_others_fix(img_tb1xyz)
        batch_fix = batch_fix.to(self.input_device)
        batch_mov = batch_mov.to(self.input_device)

        x = torch.cat([batch_fix, batch_mov], 1)  # [batch, 2, x, y, z]

        # concatenate inputs and propagate unet
        x = self.unet_model(x)
        flow_field = self.flow(x.to(self.output_device))

        # resize flow for integration
        disp_b3xyz = flow_field
        if self.n_steps > 0:
            disp_b3xyz = image_func.integrate_displacement(
                disp_b3xyz, grid=self.batch_regular_grid, n_steps=self.n_steps
            )

        if loss_function is None:
            loss = 0
            loss_part = 0
        else:
            # loss, loss_part = self.loss(loss_function, disp_b3xyz)
            loss, loss_part = self.loss(
                loss_function,
                disp_b3xyz,
                batch_fix,
                batch_mov,
                gt_fixed,
                gt_moving,
                fixed_idx,
            )

        return disp_b3xyz, loss, loss_part


class ViTVNet(SeqModel):
    def __init__(self, device_ids, input_device, output_device, img_size, n_steps=0):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps)

        config = get_3DReg_config()

        self.transformer = Transformer(config, img_size, False).to(self.input_device)
        self.decoder = DecoderCup(config, img_size).to(self.input_device)
        self.reg_head = RegistrationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config["n_dims"],
            kernel_size=3,
        ).to(self.input_device)

    def forward(self, data, loss_function=None):
        img_tb1xyz = [data[i].to(self.input_device) for i in range(len(data))]
        batch_fix, batch_mov = image_func.pick_imgpair_first_mov_others_fix(img_tb1xyz)

        batch_fix = batch_fix.to(self.input_device)
        batch_mov = batch_mov.to(self.input_device)

        x = torch.cat([batch_fix, batch_mov], 1)  # [batch, 2, x, y, z]

        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        disp_b3xyz = self.reg_head(x)

        if self.n_steps > 0:
            disp_b3xyz = image_func.integrate_displacement(
                disp_b3xyz, grid=self.batch_regular_grid, n_steps=self.n_steps
            )

        if self.output_device != self.input_device:
            batch_fix = batch_fix.to(self.output_device)
            batch_mov = batch_mov.to(self.output_device)

        if loss_function is None:
            loss = 0
            loss_part = 0
        else:
            loss, loss_part = self.loss(loss_function, disp_b3xyz, batch_fix, batch_mov)

        return disp_b3xyz, loss, loss_part


class SegViTVNet(SegModel):
    def __init__(
        self,
        device_ids,
        input_device,
        output_device,
        img_size,
        n_steps=3,
        depthwise=False,
    ):
        super().__init__(device_ids, input_device, output_device, img_size, n_steps)

        config = get_3DReg_config()

        self.transformer = Transformer(config, img_size, False).to(self.input_device)
        self.decoder = DecoderCup(config, img_size).to(self.input_device)
        self.reg_head = RegistrationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=config["n_dims"],
            kernel_size=3,
        ).to(self.input_device)

    def forward(self, data, loss_function=None):
        img_tb1xyz, gt_fixed, gt_moving, fixed_idx = data[0], data[1], data[2], data[3]
        img_tb1xyz = [
            img_tb1xyz[i].to(self.input_device) for i in range(len(img_tb1xyz))
        ]
        batch_fix, batch_mov = image_func.pick_imgpair_first_mov_others_fix(img_tb1xyz)

        batch_fix = batch_fix.to(self.input_device)
        batch_mov = batch_mov.to(self.input_device)

        x = torch.cat([batch_fix, batch_mov], 1)  # [batch, 2, x, y, z]

        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        disp_b3xyz = self.reg_head(x)

        if self.n_steps > 0:
            disp_b3xyz = image_func.integrate_displacement(
                disp_b3xyz, grid=self.batch_regular_grid, n_steps=self.n_steps
            )

        if self.output_device != self.input_device:
            batch_fix = batch_fix.to(self.output_device)
            batch_mov = batch_mov.to(self.output_device)

        if loss_function is None:
            loss = 0
            loss_part = 0
        else:
            loss, loss_part = self.loss(
                loss_function,
                disp_b3xyz,
                batch_fix,
                batch_mov,
                gt_fixed,
                gt_moving,
                fixed_idx,
            )
        return disp_b3xyz, loss, loss_part
