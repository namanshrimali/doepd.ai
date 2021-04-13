import torch
import torch.nn as nn
from models.midas.midas_net import MidasNet
from models.yolo.yolo_decoder import YoloDecoder

class DoepdNet(torch.nn.Module):
    midas_encoder_layered_output = []
    
    def __init__(self, train_mode):
        super(DoepdNet, self).__init__()
        self.train_mode = train_mode

        if not (self.train_mode == "yolo" or self.train_mode == "midas"):
            raise NotImplementedError(f'Current implementation does not support {train_mode} training mode')
        
        self.midas_net = MidasNet("weights/model-f6b98070.pt")
        
        midas_encoder_filters = [3, 256, 512, 1024, 2048] # output filters from each layer of resnext 101
                    
        # Each of the three layers in yolo takes input from last 3 layers of midas    
        self.yolo_decoder = YoloDecoder(midas_encoder_filters)
        
        # Freeze training for midas (encoder & decoder)
        for param in self.midas_net.parameters():
            param.requires_grad = False
             
    
    
    def forward(self, x):
        encoder_layered_outputs = self.midas_net.forward_encoder(x)
        
        if self.train_mode == 'yolo':
            return self.yolo_decoder.forward(encoder_layered_outputs)
        elif self.train_mode == 'midas':
            return self.midas_net.forward_decoder(encoder_layered_outputs)
    
        
          
        