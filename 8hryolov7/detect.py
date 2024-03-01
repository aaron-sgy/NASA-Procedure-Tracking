import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from utils.general import bbox_iou

import math

def get_box_center(x1, y1, x2, y2):
    return (x1+x2)/2, (y1+y2)/2

# point1 = (x1, y1)
# point2 = (x2, y2)
def get_euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# box_coordinates in format [x1, y1, x2, y2]
def are_they_the_same_detections(box_a_det, box_b_det, threshold=0):
    center_a = get_box_center(*box_a_det[:4])
    center_b = get_box_center(*box_b_det[:4])

    distance = get_euclidean_distance(center_a, center_b)

    if box_a_det[5] != box_b_det[5]:
        return False, distance

    if distance > threshold:
        return False, distance
    
    return True, distance

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    previous_overlapping_pairs_iou = -1
    previous_overlapping_interest = []

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # parameters for detections loop
        current_frame_iou = -1
        overlapping_detections = []
 
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # det[i] gives [x1, y1, x2, y2, confidence, class_id]
            if len(det):
                # Calculate IoU for each pair of bounding boxes
                for a in range(len(det)):
                    for b in range(a + 1, len(det)):
                        iou = bbox_iou(det[a][:4], det[b][:4])
                        if iou > 0:  
                            overlapping_detections.append((det[a], det[b], iou))

                overlapping_pairs_count = len(overlapping_detections)

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write Results
                for *xyxy, conf, cls in reversed(det):
                    xyxy_tensor = torch.tensor(xyxy)
                    overlap_label = None
                    for box_a, box_b, iou in overlapping_detections:
                        if bbox_iou(xyxy_tensor, torch.tensor(box_a[:4])) > 0 or bbox_iou(xyxy_tensor, torch.tensor(box_b[:4])) > 0:
                            overlap_label = f'Overlap: {iou:.2f} | {names[int(cls)]} {conf:.2f}'
                            break

                    if overlap_label:
                        plot_one_box(xyxy, im0, label=overlap_label, color=[255, 0, 0], line_thickness=1)  # Red color for overlap
                    else:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # check if we should update increasing or decreasing. Should only update when they are the same detections
            # if new they are not the same detections, should we reset prev_iou?
            # we need a new function that takes factors into account:
             #   correct two classes interactions
              #  iou threshold

            same_detections = False
            if overlapping_pairs_count == 1:
                current_frame_iou = overlapping_detections[0][2]

                if len(previous_overlapping_interest) > 0:
                    curr_detA = overlapping_detections[0][0]
                    curr_detB = overlapping_detections[0][1]

                    prev_detA = previous_overlapping_interest[0][0]
                    prev_detB = previous_overlapping_interest[0][1]

                    boolean1 = are_they_the_same_detections(curr_detA, prev_detA, 100)
                    boolean2 = are_they_the_same_detections(curr_detB, prev_detB, 100)

                    boolean3 = are_they_the_same_detections(curr_detA, prev_detB, 100)
                    boolean4 = are_they_the_same_detections(curr_detB, prev_detA, 100)
                    
                    if (boolean1[0] and boolean2[0]):
                        print("Yes they are the same detections " + str(boolean1[1]) + " and " + str(boolean2[1]))
                        same_detections = True

                    elif (boolean3[0] and boolean4[0]):
                       print("Yes they are the same detections " + str(boolean3[1]) + " and " + str(boolean4[1]))
                       same_detections = True

                    else:
                        print("No")
                        same_detections = False
                    
                previous_overlapping_interest = overlapping_detections
                    
            # detach complete
            if overlapping_pairs_count == 0 and previous_overlapping_pairs_iou <= 0.1 and previous_overlapping_pairs_iou > 0:
                cv2.putText(im0, f'Detachment has been complete for pair {str(names[int(previous_overlapping_interest[0][0][5])])} and {str(names[int(previous_overlapping_interest[0][1][5])])}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            elif webcam and overlapping_pairs_count == 1:
                cv2.putText(im0, f'There are {overlapping_pairs_count} pairs of overlapping bounding boxes. Condition Met! Updating.', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if current_frame_iou > previous_overlapping_pairs_iou and same_detections:
                    cv2.putText(im0, f'increasing', 
                        (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                elif current_frame_iou < previous_overlapping_pairs_iou and same_detections:
                    cv2.putText(im0, f'decreasing', 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                elif not same_detections:
                    cv2.putText(im0, f'New Procedure Started', 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    print("new procedure detected")

                # elif current_frame_iou == previous_overlapping_pairs_iou:
                #     cv2.putText(im0, f'unchanged?', 
                #         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                previous_overlapping_pairs_iou = current_frame_iou

            elif webcam and overlapping_pairs_count > 1:
                cv2.putText(im0, f'There are {overlapping_pairs_count} pairs of overlapping bounding boxes', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()