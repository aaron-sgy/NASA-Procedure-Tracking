# Original Imports
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

# Decision Logic Imports
from utils.general import bbox_iou
import math

# GUI Code Imports
import threading
import tkinter as tk
from CustomScrollBar import ScrollBar
from shared import *
from Step import Step
from PIL import Image, ImageTk
# already imported:
# - import cv2
# - import time

#**************************************
#
# Variable Initialization
#
#**************************************

current_step = 0
procedure = []
gui = None
flag = False

#**************************************
#
# Detect
#
#**************************************

def detect(save_img=False):
    # GUI variables being used
    global gui, flag 

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
        overlapping_pairs_count = -1

        # dummy variable for GUI
        temp = [None, None]
 
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
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            
                    if save_img or view_img:
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
                # gui.mark_step_done(DONE)

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

            temp[i] = im0

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            if not flag:
                flag = True # start GUI
                time.sleep(2) # let GUI start up

            # Stream results
            if view_img:
                # only stack videos on webcams two inputs
                if source.endswith('.txt'):
                    final = cv2.vconcat(temp)
                else:
                    final = temp[0]
                gui.set_frame(final)
                
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


#**************************************
#
# GUI
#
#**************************************

class DisplayGUI:
    def __init__(self, app):
        """
        Initializations of the GUI components for the loading screen, starts loading thread
        :param app: a TK root object
        """
        self.app = app
        self.app.title("Project Pete")
        
        # "responsive" sizing
        self.min_width = int(self.app.winfo_screenwidth() * 0.85)
        self.min_height = int(self.app.winfo_screenheight() * 0.7)
        self.app.minsize(width=self.min_width, height=self.min_height)
        
        logo = ImageTk.PhotoImage(Image.open('pete.png').resize((445,200)))
        self.logo_label = tk.Label(self.app,)
        self.logo_label.pack(padx=(100, 300))
        self.logo_label.config(image=logo)
        self.logo_label.image = logo
        
        # TODO: get loading gif to work properly
        # self.loading_label = tk.Label(self.app,)
        # self.loading_label.pack(padx=(100, 200))
        
        # loading_thread = threading.Thread(target=self._update_loading_gif,args=[])
        # loading_thread.daemon = True
        # loading_thread.start()
        
        
        self.loading_label = tk.Label(self.app, text="Please wait, loading model")
        self.loading_label.pack(padx=(100, 100))
        
        loading_model_thread = threading.Thread(target=self._check_loaded_model,args=[])
        loading_model_thread.daemon = True
        loading_model_thread.start()
    
    def procedure_tracking_setup(self, app):
        """
        Initializations of the GUI components for the actual procedure tracking system.
        :param app: a TK root object
        """
        global procedure, current_step

        # Video Frame ========================================================
        # Create a frame for the live stream
        self.left_frame = tk.Frame(self.app, bg=dark_theme_background, width=0.7 * self.min_width)
        self.left_frame.pack(side="left", fill="both", expand=False)

        # Create a label to display stream
        self.livestream = tk.Label(self.left_frame, )
        self.livestream.pack(padx=(80, 50), pady=(40, 0))

        # Logs ========================================================
        lw = int(self.left_frame.winfo_screenwidth() * 0.5)

        # Performance
        self.performance = tk.Frame(self.left_frame, bg=dark_theme_background, width=lw)
        self.performance.pack(side="left", fill="both", expand=False)
        self.performance_header = tk.Label(self.performance, text="Performance", bg=dark_theme_background,
                                           font=("Arial", 24, 'bold'))
        self.performance_header.pack(padx=(80, 0), pady=(10, 0))

        # Create a label to display the runtime
        self.runtime_label = tk.Label(self.performance, bg=dark_theme_background, text="Runtime: 0 seconds")
        self.runtime_label.pack(padx=(100, 0))

        # Create a thread for updating the runtime label
        runtime_thread = threading.Thread(target=self._update_runtime)
        runtime_thread.daemon = True
        runtime_thread.start()

        # Data
        self.data = tk.Frame(self.left_frame, width=lw, bg=dark_theme_background)
        self.data.pack(side="left", fill="both", expand=True)
        self.data_header = tk.Label(self.data, text="Data", bg=dark_theme_background, anchor='w',
                                    justify="left", font=("Arial", 24, 'bold'))
        self.data_header.pack(pady=(10, 0))

        # Procedure List ===================================================
        # Create a frame for the steps list (30% of total width)
        self.right_frame = tk.Frame(app, width=0.3 * self.min_width, bg=dark_theme_background)
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Scrollable list view
        self.canvas = tk.Canvas(self.right_frame, bg=dark_theme_background, borderwidth=0, highlightthickness=0)
        self.procedure_list = tk.Frame(self.canvas, bg=dark_theme_background, borderwidth=0, highlightthickness=0)
        self.scrollbar = ScrollBar(self.procedure_list, orient="vertical", command=self.canvas.yview)
        self.procedure_list.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="top", fill="both", expand=True, pady=(40, 0))
        self.scrollbar.pack(side="left", fill="y", expand=False)
        self.procedure_list.pack(side="right", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.procedure_list, anchor="nw", tags="canvas_frame")

        procedure = self.get_procedure()
        # initialize lists in GUI
        for step in procedure:
            if step.status == IN_PROGRESS: current_step = step.index
            step.build(self.procedure_list)

        # Override button
        self.override = tk.Label(self.right_frame, fg='white', bg=override_button_color, text="Override - mark done",
                                 borderwidth=5)
        self.override.pack(fill="x", expand=False, padx=(10, 25), pady=(20, 10))
        self.override.bind("<ButtonRelease-1>", self.override_mark_done)

        # Decision logic ===================================
        # Separate thread for communicating with other functions to get decision result
        logic_thread = threading.Thread(target=decision_logic)
        logic_thread.daemon = True
        logic_thread.start()

    def get_procedure(self):
        """
        Gets steps in a procedure.
        Step is a defined class. More stuff can be added in there to help with validation.
        TODO: actual steps lol
        """
        global procedure, current_step

        # clear out and initialize procedure + step count
        procedure = []
        current_step = 0

        # dummy steps
        # TODO: define steps & their individual criteria
        for i in range(10):
            if i < 2:
                t = i
            elif i == 2:
                t = 2
            else:
                t = 3
            s = Step(index=i + 1, title=f"Step  {i + 1}", description="[wrench type], [other relevant info]",
                     status=list(colors.keys())[t])
            procedure.append(s)
        return procedure

    def mark_step_done(self, done_type):
        """
        Mark step as done and go to next step (if exists)
        :param done_type: DONE_OV (overriden), DONE (regular auto-approved)
        """
        global current_step, procedure

        isLastStep = current_step == len(procedure)
        procedure[current_step - 1].update_status(done_type, isFocus=isLastStep)

        self.canvas.yview_moveto(1.0)
        if isLastStep: return

        current_step += 1
        procedure[current_step - 1].update_status(IN_PROGRESS)

    def override_mark_done(self, e):
        """
        Overrides logic decision (mark as complete - OV)
        """
        self.mark_step_done(DONE_OV)
    
    def set_frame(self, frame):
        """
        Updates detection preview on the left
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (750, 450))
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.livestream.config(image=photo)
        self.livestream.image = photo

    def _check_loaded_model(self):
        """
        Checks if model has finished loading, when finished, it will initialize the procedure tracking GUI and start detection
        """
        detect_thread = threading.Thread(target=detect,args=[])
        detect_thread.daemon = True
        detect_thread.start()
        
        while not flag:
            time.sleep(1)
        
        self.logo_label.destroy()
        self.loading_label.destroy()
        
        self.procedure_tracking_setup(self.app)
    
    def _update_loading_gif(self):
        i = 0
        while not flag:
            i = i + 1
            i= i % 15
            
            loading = ImageTk.PhotoImage(tk.PhotoImage(file='loading.gif',format='gif -index %i' %i))
            self.loading_label.config(image=loading)
            self.loading_label.image = loading
            time.sleep(0.1)
    
    def _update_runtime(self):
        """
        Helper function to update runtime clock
        """
        start_time = time.time()
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Format elapsed time into hh:mm:ss
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)

            runtime_str = f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}"
            self.runtime_label.config(text=runtime_str)

            time.sleep(1)  # Update the label every 1 second


# class DisplayGUI:
#     def __init__(self, app):
#         """
#         Initializations of the GUI components.
#         :param app: a TK root object
#         """
#         global procedure, current_step

#         self.app = app
#         self.app.title("Project Pete")
#         # "responsive" sizing
#         min_width = int(self.app.winfo_screenwidth() * 0.85)
#         min_height = int(self.app.winfo_screenheight() * 0.7)
#         self.app.minsize(width=min_width, height=min_height)

#         # Video Frame ========================================================
#         # Create a frame for the live stream
#         self.left_frame = tk.Frame(self.app, bg=dark_theme_background, width=0.7 * min_width)
#         self.left_frame.pack(side="left", fill="both", expand=False)

#         # Create a label to display stream
#         self.livestream = tk.Label(self.left_frame, )
#         self.livestream.pack(padx=(80, 50), pady=(40, 0))

#         # Separate thread for live stream to not cause lagging
#         update_thread = threading.Thread(target=self._update_frame)
#         update_thread.daemon = True
#         update_thread.start()

#         # Logs ========================================================
#         lw = int(self.left_frame.winfo_screenwidth() * 0.5)

#         # Performance
#         self.performance = tk.Frame(self.left_frame, bg=dark_theme_background, width=lw)
#         self.performance.pack(side="left", fill="both", expand=False)
#         self.performance_header = tk.Label(self.performance, text="Performance", bg=dark_theme_background,
#                                            font=("Arial", 24, 'bold'))
#         self.performance_header.pack(padx=(80, 0), pady=(10, 0))

#         # Create a label to display the runtime
#         self.runtime_label = tk.Label(self.performance, bg=dark_theme_background, text="Runtime: 0 seconds")
#         self.runtime_label.pack(padx=(100, 0))

#         # Create a thread for updating the runtime label
#         runtime_thread = threading.Thread(target=self._update_runtime)
#         runtime_thread.daemon = True
#         runtime_thread.start()

#         # Data
#         self.data = tk.Frame(self.left_frame, width=lw, bg=dark_theme_background)
#         self.data.pack(side="left", fill="both", expand=True)
#         self.data_header = tk.Label(self.data, text="Data", bg=dark_theme_background, anchor='w',
#                                     justify="left", font=("Arial", 24, 'bold'))
#         self.data_header.pack(pady=(10, 0))

#         # Procedure List ===================================================
#         # Create a frame for the steps list (30% of total width)
#         self.right_frame = tk.Frame(app, width=0.3 * min_width, bg=dark_theme_background)
#         self.right_frame.pack(side="right", fill="both", expand=True)

#         # Scrollable list view
#         self.canvas = tk.Canvas(self.right_frame, bg=dark_theme_background, borderwidth=0, highlightthickness=0)
#         self.procedure_list = tk.Frame(self.canvas, bg=dark_theme_background, borderwidth=0, highlightthickness=0)
#         self.scrollbar = ScrollBar(self.procedure_list, orient="vertical", command=self.canvas.yview)
#         self.procedure_list.bind(
#             "<Configure>",
#             lambda e: self.canvas.configure(
#                 scrollregion=self.canvas.bbox("all")
#             )
#         )
#         self.canvas.configure(yscrollcommand=self.scrollbar.set)
#         self.canvas.pack(side="top", fill="both", expand=True, pady=(40, 0))
#         self.scrollbar.pack(side="left", fill="y", expand=False)
#         self.procedure_list.pack(side="right", fill="both", expand=True)
#         self.canvas.create_window((0, 0), window=self.procedure_list, anchor="nw", tags="canvas_frame")

#         procedure = self.get_procedure()
#         # initialize lists in GUI
#         for step in procedure:
#             if step.status == IN_PROGRESS: current_step = step.index
#             step.build(self.procedure_list)

#         # Override button
#         self.override = tk.Label(self.right_frame, fg='white', bg=override_button_color, text="Override - mark done",
#                                  borderwidth=5)
#         self.override.pack(fill="x", expand=False, padx=(10, 25), pady=(20, 10))
#         self.override.bind("<ButtonRelease-1>", self.override_mark_done)

#         # Decision logic ===================================
#         # Separate thread for communicating with other functions to get decision result
#         logic_thread = threading.Thread(target=decision_logic)
#         logic_thread.daemon = True
#         logic_thread.start()

#     def get_procedure(self):
#         """
#         Gets steps in a procedure.
#         Step is a defined class. More stuff can be added in there to help with validation.
#         TODO: actual steps lol
#         """
#         global procedure, current_step

#         # clear out and initialize procedure + step count
#         procedure = []
#         current_step = 0

#         # dummy steps
#         # TODO: define steps & their individual criteria
#         for i in range(10):
#             if i < 2:
#                 t = i
#             elif i == 2:
#                 t = 2
#             else:
#                 t = 3
#             s = Step(index=i + 1, title=f"Step  {i + 1}", description="[wrench type], [other relevant info]",
#                      status=list(colors.keys())[t])
#             procedure.append(s)
#         return procedure

#     def mark_step_done(self, done_type):
#         """
#         Mark step as done and go to next step (if exists)
#         :param done_type: DONE_OV (overriden), DONE (regular auto-approved)
#         """
#         global current_step, procedure

#         isLastStep = current_step == len(procedure)
#         procedure[current_step - 1].update_status(done_type, isFocus=isLastStep)

#         self.canvas.yview_moveto(1.0)
#         if isLastStep: return

#         current_step += 1
#         procedure[current_step - 1].update_status(IN_PROGRESS)

#     def override_mark_done(self, e):
#         """
#         Overrides logic decision (mark as complete - OV)
#         """
#         self.mark_step_done(DONE_OV)

#     def _update_frame(self):
#         """
#         Helper function to load a vid stream from camera
#         """
#         # cap = cv2.VideoCapture(live_stream_url)
#         cap = cv2.VideoCapture(0)  # camera
#         if cap.isOpened():
#             while True:
#                 ret, frame = cap.read()
#                 if ret:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     frame = cv2.resize(frame, (750, 450))
#                     photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
#                     self.livestream.config(image=photo)
#                     self.livestream.image = photo
#         cap.release()

#     def _update_runtime(self):
#         """
#         Helper function to update runtime clock
#         """
#         start_time = time.time()
#         while True:
#             current_time = time.time()
#             elapsed_time = current_time - start_time

#             # Format elapsed time into hh:mm:ss
#             hours, remainder = divmod(int(elapsed_time), 3600)
#             minutes, seconds = divmod(remainder, 60)

#             runtime_str = f"Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}"
#             self.runtime_label.config(text=runtime_str)

#             time.sleep(1)  # Update the label every 1 second

#**************************************
#
# GUI -> Decision Logic
#
#**************************************

def decision_logic():
    global procedure, current_step, gui
    while True:     # prevent calling before initialization
        if gui is not None: break

    while current_step < len(procedure):
        """
        Decision making frame goes here: we're simply calling validate() on the current step. Each step has its 
        own `validate()` method that is defined at initialization of the procedure. The crux of decision logic is 
        stored in each step, since each step has different criteria (unless there's some other way to implement it)
    
        (?) Methods that should be called here are 1) CV detect and 2) sensor detect. 
    
        :param data: a dict of CV data (bounding boxes) and sensor data. Or, evoke methods in main to get these data.
        """

        data = "ALPACAS"
        # data['CV'] = blah blah
        # data['sensor'] = blob blob
        # data['lala'] = wawa

        # TESTING ONLY: always validate to true after 10 seconds (lol)
        # if procedure[current_step].validate(data):
        #     gui.mark_step_done(DONE)
        time.sleep(10)

#**************************************
#
# Helper Functions for Decision Logic
#
#**************************************

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

#**************************************
#                                     
# Main
#
#**************************************

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


    #**************************************
    #
    # GUI Execution Code
    #
    #**************************************
    root = tk.Tk()
    gui = DisplayGUI(root)

    root.mainloop()


    #**************************************
    #
    # Original Execution Code for Detect
    #
    #**************************************

    # with torch.no_grad():
    #     if opt.update:  # update all models (to fix SourceChangeWarning)
    #         for opt.weights in ['yolov7.pt']:
    #             detect()
    #             strip_optimizer(opt.weights)
    #     else:
    #         detect()