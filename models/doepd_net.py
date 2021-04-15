import torch
import torch.nn as nn
from .midas.midas_net import MidasNet
from .yolo.yolo_decoder import YoloDecoder
from .yolo.yolo_decoder import load_yolo_decoder_weights
class DoepdNet(torch.nn.Module):
    midas_encoder_layered_output = []
    
    def __init__(self, train_mode, yolo_weights='weights/best.pt', midas_weights = "weights/model-f6b98070.pt", image_size=384):
        super(DoepdNet, self).__init__()
        self.train_mode = train_mode

        if not (self.train_mode == "yolo" or self.train_mode == "midas"):
            raise NotImplementedError(f'Current implementation does not support {train_mode} training mode')
        
        self.midas_net = MidasNet(midas_weights)
        
        midas_encoder_filters = [256, 256, 512, 512, 1024] # output filters from each layer of resnext 101
                    
        # Each of the three layers in yolo takes input from last 3 layers of midas    
        self.yolo_decoder = YoloDecoder(midas_encoder_filters, (image_size, image_size))
        self.yolo_layers = self.yolo_decoder.yolo_layers
        
        load_yolo_decoder_weights(self.yolo_decoder, yolo_weights)
        
        
        self.midas_layer_2_to_yolo_small_obj = nn.Conv2d(in_channels= 512, out_channels = 256, kernel_size = 1, padding = 0)
        self.midas_layer_3_to_yolo_med_obj = nn.Conv2d(in_channels= 1024, out_channels = 512, kernel_size = 1, padding = 0)
        self.midas_layer_4_to_yolo_med_obj = nn.Conv2d(in_channels= 2048, out_channels = 512, kernel_size = 1, padding = 0)
        self.midas_layer_4_to_yolo_large_obj = nn.Conv2d(in_channels= 2048, out_channels = 1024, kernel_size = 1, padding = 0)
        
        # Freeze training for midas (encoder & decoder)
        for param in self.midas_net.parameters():
            param.requires_grad = False
             
    
    
    def forward(self, x):
        encoder_layered_outputs = self.midas_net.forward_encoder(x)
        
        if self.train_mode == 'yolo':
            yolo_small = self.midas_layer_2_to_yolo_small_obj(encoder_layered_outputs[1]) # midas resnext 101 layer 2
            yolo_med = self.midas_layer_3_to_yolo_med_obj(encoder_layered_outputs[2]) # midas resnext 101 layer 3
            yolo_med_before_upsample = self.midas_layer_4_to_yolo_med_obj(encoder_layered_outputs[3]) # midas resnext 101 layer 4
            yolo_large = self.midas_layer_4_to_yolo_large_obj(encoder_layered_outputs[3]) # midas resnext 101 layer 4
            
            return self.yolo_decoder.forward([yolo_small, yolo_med_before_upsample, yolo_med, yolo_large])
        elif self.train_mode == 'midas':
            return self.midas_net.forward_decoder(encoder_layered_outputs)
    
        
          
        