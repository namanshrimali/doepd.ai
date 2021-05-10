import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='DoepdNet')
    parser.add_argument('-i', '--input_path', default='input', help='folder with input images')
    parser.add_argument('-o', '--output_path', default='output', help='folder for output images')
    parser.add_argument('--gpu', dest='gpu', help='gpu', default=1, type=int)
    parser.add_argument('--dataFolder', dest='dataFolder', help='data folder', default='/content/doepd.ai/data/ScanNet', type=str)
    parser.add_argument('--anchorFolder', dest='anchorFolder',
                        help='anchor folder',
                        default='/content/doepd.ai/models/planercnn/anchors/', type=str)
    parser.add_argument('--numTrainingImages', dest='numTrainingImages',
                        help='the number of images to train',
                        default=2000, type=int)
    parser.add_argument('--numTestingImages', dest='numTestingImages',
                        help='the number of images to test/predict',
                        default=200, type=int)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=1e-5, type=float)
    parser.add_argument('--numEpochs', dest='numEpochs',
                        help='the number of epochs',
                        default=1000, type=int)
    
    args = parser.parse_args()
    return args