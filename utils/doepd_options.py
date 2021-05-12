import argparse


def parse_args():
    """
    Parse input arguments
    """
    # TODO remove number of training/testing images argument from train_planercnn
    parser = argparse.ArgumentParser(description='DoepdNet')
    parser.add_argument('-i', '--input_path', default='input', help='folder with input images')
    parser.add_argument('-o', '--output_path', default='output', help='folder for output images')
    parser.add_argument('--dataFolder', dest='dataFolder', help='data folder', default='/content/doepd.ai/data/ScanNet', type=str)
    parser.add_argument('--anchorFolder', dest='anchorFolder',
                        help='anchor folder',
                        default='/content/doepd.ai/models/planercnn/anchors/', type=str)
    parser.add_argument('--LR', dest='LR', help='learning rate', default=1e-5, type=float)
    parser.add_argument('--epochs', type=int, help="Number of epochs for training" , default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', nargs='+', type=int, default=[64], help='[min_train, max-train, test] img sizes')
    parser.add_argument('--resume', action='store_true', help='resume training from doepd_yolo_last.pt')
    parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
    parser.add_argument('--save-dir', type=str, default='/content/drive/MyDrive/doepd/weights', help='*.cfg path')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--gpu', dest='gpu', help='gpu', default=1, type=int)
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')

    args = parser.parse_args()
    return args