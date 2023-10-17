# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn import Dropout, Softmax, Linear, Conv3d, LayerNorm
from torch.nn.modules.utils import _triple
from torch.distributions.normal import Normal

# logger = logging.getLogger(__name__)

import ml_collections


def get_3DReg_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (4, 4, 4)})
    config.patches.grid = (8, 8, 8)
    # config.hidden_size = 252  # hidden_size是number_heads的倍数
    config.hidden_size = 126
    config.transformer = ml_collections.ConfigDict()
    # config.transformer.mlp_dim = 3072 # 和hidden_size组成两个全连接层
    config.transformer.mlp_dim = 512
    config.transformer.num_heads = 6  # num_heads=12
    config.transformer.num_layers = 6  # num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.patch_size = 8

    # config.conv_first_channel = 512
    config.conv_first_channel = 256
    config.encoder_channels = (16, 32, 32)
    config.down_factor = 2
    config.down_num = 1  # 注意，这里是1
    config.pool_num = 1
    config.decoder_channels = (48, 32, 16)
    config.skip_channels = (32, 32, 16)
    config.n_dims = 3
    config.n_skip = 4
    return config


# ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
# ATTENTION_K = "MultiHeadDotProductAttention_1/key"
# ATTENTION_V = "MultiHeadDotProductAttention_1/value"
# ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
# FC_0 = "MlpBlock_3/Dense_0"
# FC_1 = "MlpBlock_3/Dense_1"
# ATTENTION_NORM = "LayerNorm_0"
# MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer[
            "num_heads"
        ]  # this is defined as 12
        self.attention_head_size = int(
            config.hidden_size / self.num_attention_heads
        )  # hidden is 252, 252/12=21
        self.all_head_size = (
            self.num_attention_heads * self.attention_head_size
        )  # config.hidden_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)  # [b, n_patch, num_heads, head_size]
        return x.permute(0, 2, 1, 3)  # [b, num_heads, n_patch, head_size]

    def forward(self, hidden_states):
        mixed_query_layer = self.query(
            hidden_states
        )  # [b, n_patch, hidden]->[b, n_patch, hidden]
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(
            mixed_query_layer
        )  # [b, num_heads, n_patch, head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # [b, num_heads, n_patch, n_patch]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(
            attention_scores
        )  # [b, num_heads, 1, n_patch] 存疑
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(
            attention_probs, value_layer
        )  # [b, num_heads, 1, head_size]
        context_layer = context_layer.permute(
            0, 2, 1, 3
        ).contiguous()  # [b, 1, num_heads, head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)  # [b, n_patch, hidden]
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config, img_size):
        super(Embeddings, self).__init__()
        self.config = config
        self.down_factor = config.down_factor
        self.down_num = config.down_num
        patch_size = _triple(config.patches["size"])  # [4,4,4]
        # n_patches: 64*64*56/(2^2)^3/(8*8*7)=(16*16*14)/(8*8*7)=8
        n_patches = int(
            (img_size[0] / self.down_factor**self.down_num // patch_size[0])
            * (img_size[1] / self.down_factor**self.down_num // patch_size[1])
            * (img_size[2] / self.down_factor**self.down_num // patch_size[2])
        )

        self.hybrid_model = CNNEncoder(config, n_channels=2)
        in_channels = config["encoder_channels"][-1]  # [32]
        # hidden_size is 126, 注意下面的kernel_size=patch_size，
        # 虽然stride也是patch size，这就导致了图像会变小，
        # 输出的大小为原始图像的大小/patch_size，因为stride是patch_size
        self.patch_embeddings = Conv3d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x, features = self.hybrid_model(
            x
        )  # x is [b, encoder_channels[-1]==32, x/4, y/4, z/4]

        # 接下来开始Transformer，此时x的大小为[b,ch,x/4,y/4,z/4]

        # [b,ch,x/4,y/4,z/4] -> [B, hidden, x/4/patch_size, y/4/patch_size, z/4/patch_size]
        # 把图像x分为 （x/4/patch_size * y/4/patch_size * z/4/patch_size）个 （patch_size**3）大小的块
        # 每个块的维度为hidden_size
        x = self.patch_embeddings(x)

        # [b, hidden, x/4/patch_size,y/4/patch_size,z/4/patch_size] --> [b, hidden, n_patch]
        # n_patch = x/4/patch_size * y/4/patch_size * z/4/patch_size
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # [b, n_patch, hidden]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


# 这个Block是Transformer的Encoder，所以其实他没有Decoder
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        # 这里x的大小为 [b,n_patch, hidden]
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        # num_layers is 12
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        # (B, xyz/64, hidden)
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        # my understanding: features is a list
        # my understanding: embedding_output includes information about position
        embedding_output, features = self.embeddings(
            input_ids
        )  # embedding_output is [b, n_patch, hidden]
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv3dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        self.down_factor = config.down_factor  # 2
        self.down_num = config.down_num
        head_channels = config.conv_first_channel  # 512
        self.img_size = img_size
        self.conv_more = Conv3dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.patch_size = _triple(config.patches["size"])
        skip_channels = self.config.skip_channels
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        # x: (B, n_patch, hidden)
        # [16, 32, 32] is set by encoder_channels
        # features: [(n, 32, x/8, y/8, z/8)] * self.down_num
        # + [(b, 32, x/4, y/4, z/4), (b, 32, x/2, y/2, z/2), (b, 16, x, y, z)]
        (
            B,
            n_patch,
            hidden,
        ) = (
            hidden_states.size()
        )  # reshape from (B, n_patch, hidden) to (B, h, w,l, hidden)

        # 在 CNNencoder 里面，他对图像进行两次下采样，又进行两次maxpool，这得到了features，
        # 此时最前面的features是原始图像边长大小的 1/4*1/4 倍
        # 而这里的x，在CNNencoder那一步，只是原图像的1/4，接着，由于原文的patch_size设置的是[8,8,8]，
        # 它成为原图像的1/4，接着，由于transformer，它成为原图像的1/4*1/patch_size，
        # 所以这里的patch_size必需得是8，才能使得接下来进行连接操作。
        l, h, w = (
            (
                self.img_size[0]
                // self.down_factor**self.down_num
                // self.patch_size[0]
            ),
            (
                self.img_size[1]
                // self.down_factor**self.down_num
                // self.patch_size[1]
            ),
            (
                self.img_size[2]
                // self.down_factor**self.down_num
                // self.patch_size[2]
            ),
        )
        x = hidden_states.permute(0, 2, 1)  # [b, hidden, n_patch]
        x = x.contiguous().view(B, hidden, l, h, w)
        x = self.conv_more(x)  # [b, hidden, h, w, l] -> [b, head_channels, h, w, l]
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
                # print(skip.shape)
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class CNNEncoder(nn.Module):
    def __init__(self, config, n_channels=2):
        super(CNNEncoder, self).__init__()
        self.n_channels = n_channels
        decoder_channels = config.decoder_channels
        encoder_channels = config.encoder_channels
        self.down_num = config.down_num  # in config, this is set as 1.
        self.pool_num = config.pool_num  # 1
        self.inc = DoubleConv(n_channels, encoder_channels[0])
        self.down1 = Down(encoder_channels[0], encoder_channels[1])
        if self.down_num == 2:
            self.down2 = Down(encoder_channels[1], encoder_channels[2])
        self.width = encoder_channels[-1]

    def forward(self, x):
        features = []
        x = self.inc(x)
        features.append(x)
        x = self.down1(x)
        features.append(x)
        feats = x
        if self.down_num == 2:
            x = self.down2(x)
            features.append(x)
            feats = x
        feats_down = feats
        for i in range(self.pool_num):
            feats_down = nn.MaxPool3d(2)(feats_down)
            features.append(feats_down)
        return feats, features[::-1]


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)


# class ViTVNet(nn.Module):
#     def __init__(self, config, img_size=(64, 256, 256), int_steps=7, vis=False):
#         super(ViTVNet, self).__init__()
#         self.transformer = Transformer(config, img_size, vis)
#         self.decoder = DecoderCup(config, img_size)
#         self.reg_head = RegistrationHead(
#             in_channels=config.decoder_channels[-1],
#             out_channels=config['n_dims'],
#             kernel_size=3,
#         )
#         self.spatial_trans = SpatialTransformer(img_size)
#         self.config = config
#         #self.integrate = VecInt(img_size, int_steps)
#     def forward(self, x):

#         source = x[:,0:1,:,:]

#         # x: (B, n_patch, hidden)
#         # [16, 32, 32] is set by encoder_channels
#         # features: [(n, 32, x/8, y/8, z/8)] * self.down_num
#         # + [(b, 32, x/4, y/4, z/4), (b, 32, x/2, y/2, z/2), (b, 16, x, y, z)]
#         x, attn_weights, features = self.transformer(x)
#         x = self.decoder(x, features)
#         flow = self.reg_head(x)
#         #flow = self.integrate(flow)
#         # out = self.spatial_trans(source, flow)
#         return flow


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


# CONFIGS = {
#     'ViT-V-Net': configs.get_3DReg_config(),
# }
