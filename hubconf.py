dependencies = ['torch', 'torchvision']

import sys
import os

# Add BoQ's src directory directly to path
boq_root = os.path.dirname(__file__)  # Root of the cloned repo
sys.path.append(os.path.join(boq_root, "src"))  

import torch
from backbones import ResNet, DinoV2
from boqpp import BoQPlusPlus
from typing import Literal, Dict

class VPRModel(torch.nn.Module):
    def __init__(self, 
                 backbone,
                 aggregator):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        
    def forward(self, x, mode:Literal['train', 'test', 'simultaneous']):
        # Make sure the input lives on the same device as the model's weights,
        # so a CPU image can be fed to a GPU model (and vice-versa) seamlessly.
        x = x.to(next(self.parameters()).device)
        x = self.backbone(x)
        x = self.aggregator(x, mode=mode)
        return x


AVAILABLE_BACKBONES = {
    # this list will be extended
    # "resnet18": [8192 , 4096],
    "resnet50": [16384],
    "dinov2": [12288],
}

MODEL_URLS = {
    "resnet50_16384": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/resnet50_16384.pth",
    "dinov2_12288": "https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth",
    # "resnet50_4096": "",
}

def _build_vpr_model(backbone_name, output_dim, aggregator_cls, **kwargs):
    if backbone_name not in AVAILABLE_BACKBONES:
        raise ValueError(f"backbone_name should be one of {list(AVAILABLE_BACKBONES.keys())}")
    try:
        output_dim = int(output_dim)
    except:
        raise ValueError(f"output_dim should be an integer, not a {type(output_dim)}")
    if output_dim not in AVAILABLE_BACKBONES[backbone_name]:
        raise ValueError(f"output_dim should be one of {AVAILABLE_BACKBONES[backbone_name]}")

    if "dinov2" in backbone_name:
        # load the backbone
        backbone = DinoV2()
        # load the aggregator
        aggregator = aggregator_cls(
            in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
            proj_channels=384,
            **kwargs
        )

    elif "resnet" in backbone_name:
        backbone = ResNet(
                backbone_name=backbone_name,
                crop_last_block=True,
            )
        aggregator = aggregator_cls(
                in_channels=backbone.out_channels,  # make sure the backbone has out_channels attribute
                proj_channels=512,
                **kwargs
            )

    vpr_model = VPRModel(
            backbone=backbone,
            aggregator=aggregator
        )

    boq_state_dict = torch.hub.load_state_dict_from_url(
        MODEL_URLS[f"{backbone_name}_{output_dim}"],
        map_location=torch.device('cpu')
    )

    # Both aggregators reuse the BoQ backbone plus the aggregator's proj_c,
    # norm_input and per-block transformer encoder. The BoQ-specific query /
    # cross-attn / fc params are dropped, and the aggregator's own non-learnable
    # state (SEER buffers / BoE banks) is left at init.
    own_state_dict = vpr_model.state_dict()
    reusable = {k: v for k, v in boq_state_dict.items()
                if k in own_state_dict and own_state_dict[k].shape == v.shape}
    missing = set(own_state_dict) - set(reusable)
    dropped = set(boq_state_dict) - set(reusable)
    print(f"[{aggregator_cls.__name__}] loaded {len(reusable)} tensors from BoQ checkpoint; "
          f"dropped {len(dropped)} unused BoQ params; "
          f"{len(missing)} {aggregator_cls.__name__} params left at init.")

    vpr_model.load_state_dict(reusable, strict=False)

    return vpr_model


def get_trained_boq(backbone_name="resnet50", output_dim=16384, device=None, **kwargs):
    # device=None -> use CUDA when available, otherwise CPU. Pass an explicit
    # device (e.g. "cpu" / "cuda" / torch.device(...)) to override.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_vpr_model(backbone_name, output_dim, BoQPlusPlus, **kwargs)
    return model.to(device)