"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
from .blocks import FeatureFusionBlock, Interpolate, _make_encoder
from .base_model import BaseModel


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnext101
        """

        print("Loading weights: ", path)
        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = _make_encoder(backbone="resnext101_wsl", features=features, use_pretrained=use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)
            
    def get_midas_encoder(self):        
        return self.pretrained
    
    def get_midas_decoder(self):
        return self.scratch

    def forward_encoder(self, x):
        """Forward pass for midas' encoder.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: resnext-101 last layer
        """
        encoder = self.get_midas_encoder()
        
        layer_1 = encoder.layer1(x)
        layer_2 = encoder.layer2(layer_1)
        layer_3 = encoder.layer3(layer_2)
        layer_4 = encoder.layer4(layer_3)
        
        return [layer_1, layer_2, layer_3, layer_4]
    
    def forward_decoder(self, encoder_layered_outputs):
        
        layer_1_rn = self.scratch.layer1_rn(encoder_layered_outputs[0])
        layer_2_rn = self.scratch.layer2_rn(encoder_layered_outputs[1])
        layer_3_rn = self.scratch.layer3_rn(encoder_layered_outputs[2])
        layer_4_rn = self.scratch.layer4_rn(encoder_layered_outputs[3])

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return torch.squeeze(out, dim=1)