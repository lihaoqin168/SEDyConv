# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os.path
from typing import Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
from networks.blocks.SEDyUnetr_block import UnetrPrUpBlockDyQKV5, UnetrUpBlockPlusDyQKV5
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock


class SEDyUnetr(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        dyAttBlocks: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        reduction: float = 0.25,
        kernel_num: int = 1,
        dy_flg: bool = False,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.= number of classes
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = SEDyUnetr(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = SEDyUnetr(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.count_num = 0
        self.dyAttBlocks = dyAttBlocks
        self.dy_flg = dy_flg
        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        from networks.vit_m import ViT
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=False,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlockDyQKV5(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            reduction=reduction,
            kernel_num=kernel_num,
            dyAttBlocks=None,
        )
        self.encoder3 = UnetrPrUpBlockDyQKV5(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            reduction=reduction,
            kernel_num=kernel_num,
            dyAttBlocks=None,
        )
        self.encoder4 = UnetrPrUpBlockDyQKV5(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            reduction=reduction,
            kernel_num=kernel_num,
            dyAttBlocks=None,
        )
        self.decoder5 = UnetrUpBlockPlusDyQKV5(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            reduction=reduction,
            kernel_num=kernel_num,
            dyAttBlocks=dyAttBlocks,
        )
        self.decoder4 = UnetrUpBlockPlusDyQKV5(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            reduction=reduction,
            kernel_num=kernel_num,
            dyAttBlocks=dyAttBlocks,
        )
        self.decoder3 = UnetrUpBlockPlusDyQKV5(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            reduction=reduction,
            kernel_num=kernel_num,
            dyAttBlocks=dyAttBlocks,
        )
        self.decoder2 = UnetrUpBlockPlusDyQKV5(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
            reduction=reduction,
            kernel_num=kernel_num,
            dyAttBlocks=dyAttBlocks,
        )

        self.outA = UnetOutBlock(spatial_dims=3,
                                 in_channels=feature_size,
                                 out_channels=self.num_classes,
                                 )

        self.encoder_GAP = nn.Sequential(
            nn.InstanceNorm3d(hidden_size, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((1,1,1)),
            torch.nn.Conv3d(768, 320, 1, bias=False),
            torch.nn.BatchNorm3d(320),
            torch.nn.ReLU(),
            torch.nn.Conv3d(320, 4, 1, bias=False),
        )

    def _get_final_layer(self, in_shape: Sequence[int]):
        linear = nn.Linear(int(np.product(in_shape)), int(np.product(self.out_shape)))
        return nn.Sequential(nn.Flatten(), linear)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights, just4backbone=False):
        with torch.no_grad():
            # res_weight = weights
            for i in weights:
                print(i)
                if not just4backbone:
            #copy weight from encoder,decoder
                    if str(i).startswith("encoder") or str(i).startswith("decoder"):# or str(i).startswith("out"):
                        print("copy weight--->", i)
                        a = self
                        for attr_name in i.split("."):
                            try:
                                a = getattr(a,attr_name)
                            except:
                                continue
                        a.copy_(weights[i])
            #copy weights from vit patch embadding
            self.vit.patch_embedding.position_embeddings.copy_(
                weights['vit.patch_embedding.position_embeddings']
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights['vit.patch_embedding.patch_embeddings.1.weight']
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights['vit.patch_embedding.patch_embeddings.1.bias']
            )

            # copy weights from  encoding blocks
            for idx in range(self.vit.blocks.__len__()):
                block = self.vit.blocks.__getitem__(idx)
                print("transformer block"+str(idx))
                self.block_loadFrom(block, weights, idx)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["vit.norm.weight"])
            self.vit.norm.bias.copy_(weights["vit.norm.bias"])

    def block_loadFrom(self, block, weights, n_block):
        root =  f"vit.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.out_proj.weight",
            "attn.out_proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.linear1.weight",
            "mlp.linear1.bias",
            "mlp.linear2.weight",
            "mlp.linear2.bias",
        ]
        with torch.no_grad():
            block.norm1.weight.copy_(weights[root + block_names[0]])
            block.norm1.bias.copy_(weights[root + block_names[1]])
            block.attn.qkv.weight.copy_(weights[root + block_names[4]])
            block.attn.out_proj.weight.copy_(weights[root + block_names[6]])
            block.attn.out_proj.bias.copy_(weights[root + block_names[7]])
            block.norm2.weight.copy_(weights[root + block_names[8]])
            block.norm2.bias.copy_(weights[root + block_names[9]])
            block.mlp.linear1.weight.copy_(weights[root + block_names[10]])
            block.mlp.linear1.bias.copy_(weights[root + block_names[11]])
            block.mlp.linear2.weight.copy_(weights[root + block_names[12]])
            block.mlp.linear2.bias.copy_(weights[root + block_names[13]])
        return block

    # temperature aneal
    def net_update_temperature(self, temperature):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temperature)

    #self.dyAttBlocks, default [[1,1,1,1],[]...]
    def net_update_dyAttBlocks(self):
        for m in self.modules():
            if hasattr(m, "update_dyAttBlocks"):
                m.update_dyAttBlocks(self.dyAttBlocks)

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        if self.dy_flg:
            self.dyAttBlocks = self.encoder_GAP(dec4).squeeze()
            if x.shape[0] == 1:
                self.dyAttBlocks = torch.unsqueeze(self.dyAttBlocks, dim=0)
            # self.net_update_dyAttBlocks() # Only use at predict process !!!!!!!!!!!!!!!
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logitsA = self.outA(out)
        return logitsA

    def getModleName(self):
        return "SEDyUnetr"


if __name__ == '__main__':
    image = torch.randn(4, 1, 96, 96, 96)
    model = SEDyUnetr(
        in_channels=1,
        out_channels=14,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        reduction=0.25,
        dyAttBlocks=torch.tensor([[1, 1, 1, 1] for i in range(image.shape[0])]),
        kernel_num=4,
        dy_flg=True,
    )
    model.eval()
    print(model)
    res1 = model(image)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)