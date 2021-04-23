import torch
import torch.nn as nn
from .midas.midas_net import MidasNet
from utils.config import PlaneConfig
class DoepdNet(torch.nn.Module):
    midas_encoder_layered_output = []
    
    def __init__(self, train_mode, midas_weights = "weights/model-f6b98070.pt", image_size=384):
        super(DoepdNet, self).__init__()
        self.train_mode = train_mode

        self.midas_net = MidasNet(midas_weights)
        
        midas_encoder_filters = [256, 256, 512, 512, 1024] # output filters from each layer of resnext 101
        self.yolo_decoder = None
        self.yolo_layers = None
        self.midas_layer_2_to_yolo_small_obj = None
        self.midas_layer_3_to_yolo_med_obj = None
        self.midas_layer_4_to_yolo_med_obj = None
        self.midas_layer_4_to_yolo_large_obj = None
        self.plane_rcnn_decoder = None
        
        if self.train_mode == 'yolo':
            from .yolo.yolo_decoder import YoloDecoder
            from .yolo.yolo_decoder import load_yolo_decoder_weights
            
            # Each of the three layers in yolo takes input from last 3 layers of midas
            self.yolo_decoder = YoloDecoder(midas_encoder_filters, (image_size, image_size))
            self.yolo_layers = self.yolo_decoder.yolo_layers
            self.midas_layer_2_to_yolo_small_obj = nn.Conv2d(in_channels= 512, out_channels = 256, kernel_size = 1, padding = 0)
            self.midas_layer_3_to_yolo_med_obj = nn.Conv2d(in_channels= 1024, out_channels = 512, kernel_size = 1, padding = 0)
            self.midas_layer_4_to_yolo_med_obj = nn.Conv2d(in_channels= 2048, out_channels = 512, kernel_size = 1, padding = 0)
            self.midas_layer_4_to_yolo_large_obj = nn.Conv2d(in_channels= 2048, out_channels = 1024, kernel_size = 1, padding = 0)
        
        elif self.train_mode == 'planercnn':
            from .planercnn.planercnn_decoder import MaskRCNN
            import sys
            
            sys.argv=['']
            del sys
            
            from utils.options import parse_args
            args = parse_args()
            config = PlaneConfig(args)
            self.plane_rcnn_decoder = MaskRCNN(config)
            
        # Freeze training for midas (encoder & decoder)
        for param in self.midas_net.parameters():
            param.requires_grad = False
    
    def forward(self, x, plane_rcnn_image_meta = None, augment=False, mode='inference_detection', use_nms=2, use_refinement=True):
    
        encoder_layered_outputs = self.midas_net.forward_encoder(x)
        
        if self.train_mode == 'yolo':
            yolo_small = self.midas_layer_2_to_yolo_small_obj(encoder_layered_outputs[1]) # midas resnext 101 layer 2
            yolo_med = self.midas_layer_3_to_yolo_med_obj(encoder_layered_outputs[2]) # midas resnext 101 layer 3
            yolo_med_before_upsample = self.midas_layer_4_to_yolo_med_obj(encoder_layered_outputs[3]) # midas resnext 101 layer 4
            yolo_large = self.midas_layer_4_to_yolo_large_obj(encoder_layered_outputs[3]) # midas resnext 101 layer 4
            
            return self.yolo_decoder.forward([yolo_small, yolo_med_before_upsample, yolo_med, yolo_large], augment=augment)
        elif self.train_mode == 'midas':
            return self.midas_net.forward_decoder(encoder_layered_outputs)
        elif self.train_mode == 'planercnn':
            return self.plane_rcnn_decoder.predict(x, plane_rcnn_image_meta, 'inference', encoder_layered_outputs = encoder_layered_outputs)
            
    
def load_doepd_weights(self, device='cpu', scratch=False, train_mode = False, load_mode='all'):
    yolo_weights = []
    chkpt = None
        
    if not scratch:
        # loading yolo weights
            
        if self.train_mode == 'yolo':
            yolo_weight_file =  None
            # loading yolo weights from last/best based on train_mode. Will update to add planercnn weights
            if train_mode:
                yolo_weight_file = 'weights/doepd_yolo_last.pt'
            else:
                yolo_weight_file = 'weights/doepd_yolo_best.pt'
            
            chkpt = torch.load(yolo_weight_file, map_location = "cpu")    
            num_items = 0
            
            for k, v in chkpt['model'].items():
                if num_items>=666 and num_items<756:
                    if not k.endswith('num_batches_tracked'):
                        yolo_weights.append(v.detach().numpy())
                num_items = num_items + 1 
            
            load_yolo_decoder_weights(self.yolo_decoder, yolo_weights)
                    
            self.midas_layer_2_to_yolo_small_obj.weight = torch.nn.Parameter(chkpt['model']['midas_layer_2_to_yolo_small_obj.weight'])
            self.midas_layer_2_to_yolo_small_obj.bias = torch.nn.Parameter(chkpt['model']['midas_layer_2_to_yolo_small_obj.bias'])
            self.midas_layer_3_to_yolo_med_obj.weight = torch.nn.Parameter(chkpt['model']['midas_layer_3_to_yolo_med_obj.weight'])
            self.midas_layer_3_to_yolo_med_obj.bias = torch.nn.Parameter(chkpt['model']['midas_layer_3_to_yolo_med_obj.bias'])
            self.midas_layer_4_to_yolo_med_obj.weight = torch.nn.Parameter(chkpt['model']['midas_layer_4_to_yolo_med_obj.weight'])
            self.midas_layer_4_to_yolo_med_obj.bias = torch.nn.Parameter(chkpt['model']['midas_layer_4_to_yolo_med_obj.bias'])
            self.midas_layer_4_to_yolo_large_obj.weight = torch.nn.Parameter(chkpt['model']['midas_layer_4_to_yolo_large_obj.weight'])
            self.midas_layer_4_to_yolo_large_obj.bias = torch.nn.Parameter(chkpt['model']['midas_layer_4_to_yolo_large_obj.bias'])
            
        elif self.train_mode == 'planercnn':
            planer_cnn_file = 'weights/planer_checkpoint.pth'
            chkpt = torch.load(planer_cnn_file, map_location = "cpu")    
            plane_rcnn_weights = []
            num_items = 0
            
            for k, v in chkpt.items():
                if num_items>=728:
                    # eg k = depth.deconv1.2.running_var
                    # we need plane_rcnn_decoder.depth.deconv1.2.running_var
                    self.plane_rcnn_decoder.state_dict()[f'plane_rcnn_decoder.{k}'] = v.data
                num_items = num_items + 1 
            
            
            
    else:
         # loading yolo_best weights : got from 300 epochs trained in Assignment 13
            
        yolo_weight_file='weights/yolo_old_300.pt'
            
        chkpt = torch.load(yolo_weight_file, map_location = device)
            
        num_items=0
        for k, v in chkpt['model'].items():
            if num_items >= 354:
                if not k.endswith('num_batches_tracked'):
                    if v.shape[0]!=255:
                        yolo_weights.append(v.detach().numpy())
            num_items = num_items + 1
        
        load_yolo_decoder_weights(self.yolo_decoder, yolo_weights)
    
    return chkpt