# yolo opts -> opt.img_size,  opt.output, opt.source, opt.device, opt.conf_thres, opt.iou_thres, opt.fourcc, opt.names

import argparse


def parse_args():
    """
    Parse input arguments
    """
    # TODO remove number of training/testing images argument from train_planercnn
    parser = argparse.ArgumentParser(description='DoepdNet inference options')
    parser.add_argument('--names', type=str, default='data/assignment13/custom.names', help='*.names path')
    parser.add_argument('--source', type=str, default='data/customdata/images', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--run-mode', default='all', help='choose load mode for doepd_net from "planercnn", "yolo", "midas", "all". Defaults to "all"')

    args = parser.parse_args()
    return args