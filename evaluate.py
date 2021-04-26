"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import sys
import os
import cv2
import copy
import glob

from models.planercnn.planercnn_decoder import *
from models.planercnn.refinement_net import RefineModel
from models.planercnn.modules import *
from datasets.plane_stereo_dataset import PlaneDataset
from datasets.inference_dataset import InferenceDataset
from datasets.nyu_dataset import NYUDataset
from utils.planer_cnn_utils import *
from utils.visualize_utils import *
from utils.evaluate_utils import *
from utils.plane_utils import *
from utils.options import parse_args
from utils.config import InferenceConfig
from models.doepd_net import DoepdNet, load_doepd_weights

class PlaneRCNNDetector():
    def __init__(self, options, config, modelType, checkpoint_dir='weights'):
        self.options = options
        self.config = config
        self.modelType = modelType
        self.model = DoepdNet(train_mode='planercnn')
        self.model.cuda()
        self.model.eval()

        if modelType == 'final':
            checkpoint_dir = checkpoint_dir if checkpoint_dir != '' else 'checkpoint/planercnn_' + options.anchorType
            pass

        if options.suffix != '':
            checkpoint_dir += '_' + options.suffix
            pass

        ## Indicates that the refinement network is trained separately        
        separate = modelType == 'refine'

        # if not separate:
        #     if options.startEpoch >= 0:
        #         self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint_' + str(options.startEpoch) + '.pth'))
        #     else:
                
        #         self.model.load_state_dict(torch.load(checkpoint_dir + '/checkpoint.pth'))
        #         pass
        #     pass
        
        load_doepd_weights(self.model, device="cuda")

        if 'refine' in modelType or 'final' in modelType:
            self.refine_model = RefineModel(options)

            self.refine_model.cuda()
            self.refine_model.eval()
            if not separate:
                state_dict = torch.load(checkpoint_dir + '/checkpoint_refine.pth')
                self.refine_model.load_state_dict(state_dict)
                pass
            else:
                # self.model.load_state_dict(torch.load('checkpoint/pair_' + options.anchorType + '_pair/checkpoint.pth'))
                self.refine_model.load_state_dict(torch.load('checkpoint/instance_normal_refine_mask_softmax_valid/checkpoint_refine.pth'))
                pass
            pass

        return

    def detect(self, sample):

        input_pair = []
        detection_pair = []
        camera = sample[30][0].cuda()
        for indexOffset in [0, ]:
            images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()
            # rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = self.model.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], mode='inference_detection', use_nms=2, use_refinement=True)
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks, detection_gt_parameters, detection_gt_masks, rpn_rois, roi_features, roi_indices, depth_np_pred = self.model.forward(
                images,
                [image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters, camera], 
                mode='inference_detection', 
                use_nms=2, 
                use_refinement=True)

            if len(detections) > 0:
                detections, detection_masks = unmoldDetections(self.config, camera, detections, detection_masks, depth_np_pred, debug=False)
                pass

            XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
            detection_mask = detection_mask.unsqueeze(0)

            input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera})

            if 'nyu_dorn_only' in self.options.dataset:
                XYZ_pred[1:2] = sample[27].cuda()
                pass

            detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'depth_np': depth_np_pred, 'plane_XYZ': plane_XYZ})
            continue

        if ('refine' in self.modelType or 'refine' in self.options.suffix):
            pose = sample[26][0].cuda()
            pose = torch.cat([pose[0:3], pose[3:6] * pose[6]], dim=0)
            pose_gt = torch.cat([pose[0:1], -pose[2:3], pose[1:2], pose[3:4], -pose[5:6], pose[4:5]], dim=0).unsqueeze(0)
            camera = camera.unsqueeze(0)

            for c in range(1):
                detection_dict, input_dict = detection_pair[c], input_pair[c]

                new_input_dict = {k: v for k, v in input_dict.items()}
                new_input_dict['image'] = (input_dict['image'] + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
                new_input_dict['image_2'] = (sample[13].cuda() + self.config.MEAN_PIXEL_TENSOR.view((-1, 1, 1))) / 255.0 - 0.5
                detections = detection_dict['detection']
                detection_masks = detection_dict['masks']
                depth_np = detection_dict['depth_np']
                image = new_input_dict['image']
                image_2 = new_input_dict['image_2']
                depth_gt = new_input_dict['depth'].unsqueeze(1)

                masks_inp = torch.cat([detection_masks.unsqueeze(1), detection_dict['plane_XYZ']], dim=1)

                segmentation = new_input_dict['segmentation']

                detection_masks = torch.nn.functional.interpolate(detection_masks[:, 80:560].unsqueeze(1), size=(192, 256), mode='nearest').squeeze(1)
                image = torch.nn.functional.interpolate(image[:, :, 80:560], size=(192, 256), mode='bilinear')
                image_2 = torch.nn.functional.interpolate(image_2[:, :, 80:560], size=(192, 256), mode='bilinear')
                masks_inp = torch.nn.functional.interpolate(masks_inp[:, :, 80:560], size=(192, 256), mode='bilinear')
                depth_np = torch.nn.functional.interpolate(depth_np[:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
                plane_depth = torch.nn.functional.interpolate(detection_dict['depth'][:, 80:560].unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1)
                segmentation = torch.nn.functional.interpolate(segmentation[:, 80:560].unsqueeze(1).float(), size=(192, 256), mode='nearest').squeeze().long()

                new_input_dict['image'] = image
                new_input_dict['image_2'] = image_2

                results = self.refine_model(image, image_2, camera, masks_inp, detection_dict['detection'][:, 6:9], plane_depth, depth_np)

                masks = results[-1]['mask'].squeeze(1)

                all_masks = torch.softmax(masks, dim=0)

                masks_small = all_masks[1:]
                all_masks = torch.nn.functional.interpolate(all_masks.unsqueeze(1), size=(480, 640), mode='bilinear').squeeze(1)
                all_masks = (all_masks.max(0, keepdim=True)[1] == torch.arange(len(all_masks)).cuda().long().view((-1, 1, 1))).float()
                masks = all_masks[1:]
                detection_masks = torch.zeros(detection_dict['masks'].shape).cuda()
                detection_masks[:, 80:560] = masks


                detection_dict['masks'] = detection_masks
                detection_dict['depth_ori'] = detection_dict['depth'].clone()
                detection_dict['mask'][:, 80:560] = (masks.max(0, keepdim=True)[0] > (1 - masks.sum(0, keepdim=True))).float()

                if self.options.modelType == 'fitting':
                    masks_cropped = masks_small
                    ranges = self.config.getRanges(camera).transpose(1, 2).transpose(0, 1)
                    XYZ = torch.nn.functional.interpolate(ranges.unsqueeze(1), size=(192, 256), mode='bilinear').squeeze(1) * results[-1]['depth'].squeeze(1)
                    detection_areas = masks_cropped.sum(-1).sum(-1)
                    A = masks_cropped.unsqueeze(1) * XYZ
                    b = masks_cropped
                    Ab = (A * b.unsqueeze(1)).sum(-1).sum(-1)
                    AA = (A.unsqueeze(2) * A.unsqueeze(1)).sum(-1).sum(-1)
                    plane_parameters = torch.stack([torch.matmul(torch.inverse(AA[planeIndex]), Ab[planeIndex]) if detection_areas[planeIndex] else detection_dict['detection'][planeIndex, 6:9] for planeIndex in range(len(AA))], dim=0)
                    plane_offsets = torch.norm(plane_parameters, dim=-1, keepdim=True)
                    plane_parameters = plane_parameters / torch.clamp(torch.pow(plane_offsets, 2), 1e-4)
                    detection_dict['detection'][:, 6:9] = plane_parameters

                    XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(self.config, camera, detection_dict['detection'], detection_masks, detection_dict['depth'], return_individual=True)
                    detection_dict['depth'] = XYZ_pred[1:2]
                    pass
                continue
            pass
        return detection_pair

def evaluate(options):
    config = InferenceConfig(options)
    config.FITTING_TYPE = options.numAnchorPlanes
    if 'inference' in options.dataset:
        image_list = glob.glob(options.customDataFolder + '/*.png') + glob.glob(options.customDataFolder + '/*.jpg')
        if os.path.exists(options.customDataFolder + '/camera.txt'):
            camera = np.zeros(6)
            with open(options.customDataFolder + '/camera.txt', 'r') as f:
                for line in f:
                    values = [float(token.strip()) for token in line.split(' ') if token.strip() != '']
                    for c in range(6):
                        camera[c] = values[c]
                        continue
                    break
                pass
        else:
            camera = [filename.replace('.png', '.txt').replace('.jpg', '.txt') for filename in image_list]
            pass
        dataset = InferenceDataset(options, config, image_list=image_list, camera=camera)
        pass

    print('the number of images', len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    epoch_losses = []
    data_iterator = tqdm(dataloader, total=len(dataset))

    specified_suffix = options.suffix
    with torch.no_grad():
        detectors = []
        for method in options.methods:
            if method == 'f':
                options.suffix = specified_suffix if specified_suffix != '' else ''
                detectors.append(('final', PlaneRCNNDetector(options, config, modelType='final')))
                pass
            continue
        pass

    if not options.debug:
        for method_name in [detector[0] for detector in detectors]:
            os.system('rm ' + options.test_dir + '/*_' + method_name + '.png')
            continue
        pass

    all_statistics = []
    for name, detector in detectors:
        statistics = [[], [], [], []]
        for sampleIndex, sample in enumerate(data_iterator):
            if options.testingIndex >= 0 and sampleIndex != options.testingIndex:
                if sampleIndex > options.testingIndex:
                    break
                continue
            input_pair = []
            camera = sample[30][0].cuda()
            for indexOffset in [0, ]:
                images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_parameters, gt_depth, extrinsics, planes, gt_segmentation = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 7].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda()
                masks = (gt_segmentation == torch.arange(gt_segmentation.max() + 1).cuda().view(-1, 1, 1)).float()
                input_pair.append({'image': images, 'depth': gt_depth, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'camera': camera, 'plane': planes[0], 'masks': masks, 'mask': gt_masks})
                continue

            if sampleIndex >= options.numTestingImages:
                break

            with torch.no_grad():
                detection_pair = detector.detect(sample)
                
                pass

            if options.dataset == 'rob':
                depth = detection_pair[0]['depth'].squeeze().detach().cpu().numpy()
                os.system('rm ' + image_list[sampleIndex].replace('color', 'depth'))
                depth_rounded = np.round(depth * 256)
                depth_rounded[np.logical_or(depth_rounded < 0, depth_rounded >= 256 * 256)] = 0
                cv2.imwrite(image_list[sampleIndex].replace('color', 'depth').replace('jpg', 'png'), depth_rounded.astype(np.uint16))
                continue


            if 'inference' not in options.dataset:
                for c in range(len(input_pair)):
                    evaluateBatchDetection(options, config, input_pair[c], detection_pair[c], statistics=statistics, printInfo=options.debug, evaluate_plane=options.dataset == '')
                    continue
            else:
                for c in range(len(detection_pair)):
                    np.save(options.test_dir + '/' + str(sampleIndex % 500) + '_plane_parameters_' + str(c) + '.npy', detection_pair[c]['detection'][:, 6:9].cpu())
                    np.save(options.test_dir + '/' + str(sampleIndex % 500) + '_plane_masks_' + str(c) + '.npy', detection_pair[c]['masks'][:, 80:560].cpu())
                    continue
                pass
                            
            if sampleIndex < 30 or options.debug or options.dataset != '':
                visualizeBatchPair(options, config, input_pair, detection_pair, indexOffset=sampleIndex % 500, suffix='_' + name + options.modelType, write_ply=options.testingIndex >= 0, write_new_view=options.testingIndex >= 0 and 'occlusion' in options.suffix)
                pass
            if sampleIndex >= options.numTestingImages:
                break
            continue
        if 'inference' not in options.dataset:
            options.keyname = name
            printStatisticsDetection(options, statistics)
            all_statistics.append(statistics)
            pass
        continue
    if 'inference' not in options.dataset:
        if options.debug and len(detectors) > 1:
            all_statistics = np.concatenate([np.arange(len(all_statistics[0][0])).reshape((-1, 1)), ] + [np.array(statistics[3]) for statistics in all_statistics], axis=-1)
            print(all_statistics.astype(np.int32))
            pass
        if options.testingIndex == -1:
            np.save('logs/all_statistics.npy', all_statistics)
            pass
        pass
    return

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == '':
        args.keyname = 'evaluate'
    else:
        args.keyname = args.dataset
        pass
    args.test_dir = 'test/' + args.keyname

    if args.testingIndex >= 0:
        args.debug = True
        pass
    if args.debug:
        args.test_dir += '_debug'
        args.printInfo = True
        pass

    ## Write html for visualization
    if False:
        if False:
            info_list = ['image_0', 'segmentation_0', 'segmentation_0_warping', 'depth_0', 'depth_0_warping']
            writeHTML(args.test_dir, info_list, numImages=100, convertToImage=False, filename='index', image_width=256)
            pass
        if False:
            info_list = ['image_0', 'segmentation_0', 'detection_0_planenet', 'detection_0_warping', 'detection_0_refine']
            writeHTML(args.test_dir, info_list, numImages=20, convertToImage=True, filename='comparison_segmentation')
            pass
        if False:
            info_list = ['image_0', 'segmentation_0', 'segmentation_0_manhattan_gt', 'segmentation_0_planenet', 'segmentation_0_warping']
            writeHTML(args.test_dir, info_list, numImages=30, convertToImage=False, filename='comparison_segmentation')
            pass
        exit(1)
        pass

    if not os.path.exists(args.test_dir):
        os.system("mkdir -p %s"%args.test_dir)
        pass

    if args.debug and args.dataset == '':
        os.system('rm ' + args.test_dir + '/*')
        pass

    evaluate(args)
