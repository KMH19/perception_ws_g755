import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
import time

class RTdetrWrapper:
    def __init__(self, model_path: str, conf_thresh: float = 0.7, imgsz: int = 1280):
        """
        initialising function for the YOLOv8 PyTorch model with confidence threshold
        """
        torch.cuda.set_device(0)
        self.model = YOLO(model_path, task="detect")
        # if model_path.endswith('.engine'):
        #    self.model.info(verbose=True, detailed=True)
           
        #self.model.conf = conf_thresh
        self.secondary_conf = conf_thresh
        self.imgsz = imgsz

    def inference(self, colour_frame: np.ndarray, verbose: bool = False):
        """
        function for running inference on a single frame
        """
        #frame_result: Results = self.model(colour_frame, verbose=verbose, imgsz=self.imgsz, device=0)[0]
        #Speed: 3.7ms preprocess, 50.6ms inference, 26.7ms postprocess per image at shape (1, 3, 640, 640)
        frame_result: Results = self.model(colour_frame)[0]
        detection_boxes = []
        if frame_result.boxes.xyxy.shape[0] == 0:
            return []
        
        # Move all tensors to CPU and convert to the appropriate types outside the loop.
        classes = frame_result.boxes.cls.to(device='cpu', dtype=torch.int32).tolist()
        boxes = frame_result.boxes.xyxy.to(device='cpu', dtype=torch.int32).tolist()
        confs = frame_result.boxes.conf.to(device='cpu').tolist()

        for i in range(len(boxes)):
            class_id = classes[i]
            box = boxes[i]
            conf = confs[i]
            
            if conf > self.secondary_conf:
                detection_boxes.append([class_id] + box)

        return detection_boxes
