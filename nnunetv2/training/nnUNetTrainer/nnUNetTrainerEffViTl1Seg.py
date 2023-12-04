import torch
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

import sys
sys.path.append('efficientvit')
from nnunetv2.nets.efficientvit.models.efficientvit.backbone import efficientvit_backbone_l1
from nnunetv2.nets.efficientvit.models.efficientvit import EfficientViTSeg
from nnunetv2.nets.efficientvit.models.efficientvit.seg import SegHead, EfficientViTSeg

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision

from torch.optim import AdamW
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch import nn
from torch.nn import functional as F

class EfficientViTSegWrapper(nn.Module):
    def __init__(self, efficient_vit_seg):
        super().__init__()
        self.efficient_vit_seg = efficient_vit_seg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_res_seg = self.efficient_vit_seg(x)
        ori_res_seg = F.interpolate(
            low_res_seg,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        return ori_res_seg

class nnUNetTrainerEffViTl1Seg(nnUNetTrainerNoDeepSupervision):
    def __init__(
            self,
            plans: dict,
            configuration: str,
            fold: int,
            dataset_json: dict,
            unpack_dataset: bool = True,
            device: torch.device = torch.device('cuda')
        ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision = False):



        label_manager = plans_manager.get_label_manager(dataset_json)

        backbone = efficientvit_backbone_l1(in_channels=num_input_channels)
        head = SegHead(
            fid_list=["stage4", "stage3", "stage2"],
            in_channel_list=[512, 256, 128],
            stride_list=[32, 16, 8],
            head_stride=8,
            head_width=128,
            head_depth=3,
            expand_ratio=4,
            middle_op="fmbconv",
            final_expand=8,
            act_func="gelu",
            n_classes = label_manager.num_segmentation_heads
        )

        effvit_l1_seg = EfficientViTSeg(backbone, head)
        model = EfficientViTSegWrapper(effvit_l1_seg)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")

        return model
    
    #def configure_optimizers(self):
    #    optimizer = AdamW(
    #            self.network.parameters(),
    #            lr=self.initial_lr,
    #            weight_decay=1e-5,
    #            betas=(0.9, 0.999),
    #            amsgrad=False
    #        )

    #    lr_scheduler = PolyLRScheduler(
    #        optimizer,
    #        self.initial_lr,
    #        self.num_epochs,
    #        exponent=1.0
    #    )
    
    #    return optimizer, lr_scheduler
    