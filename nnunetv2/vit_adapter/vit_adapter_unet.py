import torch.nn as nn
from torch.nn import functional as F
import torch

from nnunetv2.vit_adapter.segmentation.mmseg_custom.models.backbones.vit_adapter import ViTAdapter
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks

class ViTAdapterUNet(nn.Module):
    def __init__(
            self,
            num_classes,
            img_size=512,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            drop_path_rate=0.0,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=12,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
            window_attn=[False] * 12,
            window_size=[None] * 12,
            enable_deep_supervision = True
        ):
        super().__init__()

        self.deep_supervision = enable_deep_supervision

        self.vit_adapter = ViTAdapter(
            img_size=img_size,
            pretrain_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            cffn_ratio=cffn_ratio,
            deform_ratio=deform_ratio,
            interaction_indexes=interaction_indexes,
            window_attn=window_attn,
            window_size=window_size
        )

        self.encoder1 = StackedResidualBlocks(
            n_blocks=2,
            conv_op=nn.Conv2d,
            input_channels=3,
            output_channels=32,
            kernel_size=3,
            initial_stride=1,
            conv_bias=True,
            norm_op=nn.BatchNorm2d,
            norm_op_kwargs = {'eps': 1e-05, 'affine': True},
            nonlin = nn.ReLU,
            nonlin_kwargs = {'inplace': True}
        )

        self.encoder2 = StackedResidualBlocks(
            n_blocks=2,
            conv_op=nn.Conv2d,
            input_channels=32,
            output_channels=64,
            kernel_size=3,
            initial_stride=2,
            conv_bias=True,
            norm_op=nn.BatchNorm2d,
            norm_op_kwargs = {'eps': 1e-05, 'affine': True},
            nonlin = nn.ReLU,
            nonlin_kwargs = {'inplace': True}
        )

        self.encoder3 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=128,
            kernel_size=1,
            stride=1,
            bias=True
        )

        self.encoder4 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=256,
            kernel_size=1,
            stride=1,
            bias=True
        )

        self.encoder5 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=256,
            kernel_size=1,
            stride=1,
            bias=True
        )

        self.encoder6 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=256,
            kernel_size=1,
            stride=1,
            bias=True
        )

        num_channels = [256, 256, 128, 64, 32]
        transpconvs = []
        for i in range(5):
            transpconvs.append(
                nn.ConvTranspose2d(
                    in_channels= num_channels[i] * 2 if num_channels[i] * 2 <= 256 else 256,
                    out_channels= num_channels[i],
                    kernel_size=2,
                    stride=2,
                    bias=True
                )
            )
        self.transpconvs = nn.ModuleList(transpconvs)

        stages = []
        for i in range(5):
            stages.append(
                StackedResidualBlocks(
                    n_blocks=2,
                    conv_op=nn.Conv2d,
                    input_channels = num_channels[i] * 2,
                    output_channels = num_channels[i],
                    kernel_size=3,
                    initial_stride=1,
                    conv_bias=True,
                    norm_op=nn.BatchNorm2d,
                    norm_op_kwargs = {'eps': 1e-05, 'affine': True},
                    nonlin = nn.ReLU,
                    nonlin_kwargs = {'inplace': True}
                )
            )
        self.stages = nn.ModuleList(stages)

        self.seg_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=num_channels[i],
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                ) for i in range(5)
            ]
        )
    
    def forward(self, x_in):
        if x_in.shape[1] <= 3:
            x_in = x_in.repeat(1, 3, 1, 1)
        skips = self.forward_encoder(x_in)

        return self.forward_decoder(skips)
    
    def forward_encoder(self, x_in):
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(enc1)
        x3, x4, x5, x6 = self.vit_adapter(x_in)
        enc3 = self.encoder3(x3)
        enc4 = self.encoder4(x4)
        enc5 = self.encoder5(x5)
        enc6 = self.encoder6(x6)

        skips = [enc1, enc2, enc3, enc4, enc5, enc6]

        return skips
    
    def forward_decoder(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

