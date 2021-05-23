import os
import shutil
import utils.torch_utils as torch_utils
from models.doepd_net import *
from utils.datasets import LoadImages
from utils.utils import load_classes, non_max_suppression, scale_coords, plot_one_box
import random
import time
from pathlib import Path
import cv2
import utils.midas_utils as utils


ONNX_EXPORT = False

def inference(options):
    img_size = (320, 192) if ONNX_EXPORT else options.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source = options.output, options.source
    
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else options.device)

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    
    # Initialize model
    model = DoepdNet(run_mode = options.run_mode, image_size = img_size)
    
    load_doepd_weights(model, device=device, train_mode=False)
    # Eval mode
    model.to(device).eval()
    
    dataset = LoadImages(source, img_size=img_size)
    names = load_classes(options.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    start_time = time.time()
    _ = model(torch.zeros((1, 3, img_size, img_size), device=device)) if device.type != 'cpu' else None  # run once
    
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img = img.unsqueeze(0)
        # Inference
        time_before_inference = torch_utils.time_synchronized()
        doepd_prediction = model(img)
        time_after_inference = torch_utils.time_synchronized()
        if options.run_mode == 'yolo' or options.run_mode == 'all':
            yolo_prediction = doepd_prediction[0][0]
            # non max supression for yolo output
            yolo_prediction = non_max_suppression(yolo_prediction, options.conf_thres, options.iou_thres, multi_label=False)
                # Process detections
            for _, det in enumerate(yolo_prediction):  # detections per image
                p, s, im0 = path, '', im0s
                save_path = os.path.join(out, os.path.splitext(os.path.basename(p))[0] + "_yolo.png")
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                # Print time (inference + NMS)
                print('%sDone.' % s)
                # Save results (image with detections)
                cv2.imwrite(save_path, im0)
        
        if options.run_mode == 'midas' or options.run_mode == 'all':
            midas_prediction = doepd_prediction[1]
            midas_prediction = (
                torch.nn.functional.interpolate(
                midas_prediction.unsqueeze(1),
                size=im0s.shape[:2],
                mode="bicubic",
                align_corners=False)
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
        # output
            depth_filename = os.path.join(
                out, os.path.splitext(os.path.basename(path))[0] + "_depth"
            )

            utils.write_depth(depth_filename, midas_prediction, bits=2)
            print('%gx%g Depth Done' % midas_prediction.shape[:2])

        print(f"Inferencing completed in {(time_after_inference - time_before_inference):.3f}s")
        print('Results saved to %s' % out)

    print('Done. (%.3fs)' % (time.time() - start_time))
        
if __name__ == '__main__':
    from utils.inference_options import parse_args
    options = parse_args()
    print(options)
    inference(options = options)