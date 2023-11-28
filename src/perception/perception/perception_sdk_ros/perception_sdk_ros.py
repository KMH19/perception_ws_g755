#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cone_msgs.msg import Cone
from cone_msgs.msg import ConeStamped
from cone_msgs.msg import ConeMap
from cone_msgs.msg import ConeMapStamped

from geometry_msgs.msg import PoseArray, Pose, PoseStamped


from ultralytics import RTDETR


import numpy as np
from OpenGL.GLUT import *

import cv2
import pyzed.sl as sl


# sys.path.insert(0, './yolov7')
# from models.experimental import attempt_load
# from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
# from utils.torch_utils import select_device
# from utils.datasets import letterbox

from threading import Lock, Thread
from time import sleep

CUDA_LAUNCH_BLOCKING=1

#Basic arguments of the scripts
weights = "/home/kmhytting/Git/autonomous-cone-detect-map-test/src/moa/cone-detection/cone_detection/rtdetr_detector.pt"
img_size = 640
conf_thres = 0.7

lock = Lock()

#? ideas from: https://github.com/stereolabs/zed-yolo/blob/master/pytorch_yolov8/detector.py

class detection(Node):
    def __init__(self):
        self.run_signal = False
        self.exit_signal = False
        super().__init__('detector')

        # Initialize ZED camera and YOLOv7
        capture_thread = Thread(target=self.torch_thread,kwargs={'weights': weights, 'img_size': img_size, "conf_thres": conf_thres})
        capture_thread.start()
        
        print("Initializing Camera...")
        
        self.zed = sl.Camera()
        
        input_type = sl.InputType()
        
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # QUALITY
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 20
        
        self.runtime_params = sl.RuntimeParameters()
        status = self.zed.open(init_params)
        
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        
        self.image_left_tmp = sl.Mat()
        
        print("Initialized Camera") #? ok to here.

        
        #* Positional Tracking
        py_transform = sl.Transform() #? ..
        positional_tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
        # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
        # positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(positional_tracking_parameters)
        #*
        
        #* Object Detection
        self.obj_param = sl.ObjectDetectionParameters()
        self.obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        self.obj_param.enable_tracking = True
        self.zed.enable_object_detection(self.obj_param)
        
        self.objects_ = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        #*
        
        #! Utilities for tracks view (not used in ROS)
        # camera_config = camera_infos.camera_configuration
        # tracks_resolution = sl.Resolution(400, display_resolution.height)
        # track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
        # track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        # image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)

        #* Camera position
        self.zed_pose = sl.Pose()
        zed_sensors = sl.SensorsData()
        zed_info = self.zed.get_camera_information()
        self.py_translation = sl.Translation()
        #*

        #* Publiher init
        self.publisher = self.create_publisher(ConeMap, '/cone_detection', 10)
        self.cone_pub = self.create_publisher(PoseArray, '/perception_node/blue_cones_position', 10)

        self.timer = self.create_timer(0.5, self.run_detection)
        #* 
        
    # def img_preprocess(self, img, device, half, net_size):
    #     net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
    #     net_image = net_image.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
    #     net_image = np.ascontiguousarray(net_image)

    #     img = torch.from_numpy(net_image).to(device)
    #     img = img.half() if half else img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0

    #     if img.ndimension() == 3:
    #             img = img.unsqueeze(0)
    #     return img, ratio, pad

    #? OK
    def xywh2abcd(self, xywh, im_shape):
        output = np.zeros((4, 2))

        # Center / Width / Height -> BBox corners coordinates
        x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
        x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
        y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
        y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

        # A ------ B
        # | Object |
        # D ------ C

        output[0][0] = x_min
        output[0][1] = y_min

        output[1][0] = x_max
        output[1][1] = y_min

        output[2][0] = x_min
        output[2][1] = y_max

        output[3][0] = x_max
        output[3][1] = y_max
        return output
    
    def detections_to_custom_box(self, detections, im0):
        output = []
        
        for i, det in enumerate(detections):
            xywh = det.xywh[0]

            # Creating ingestable objects for the ZED SDK
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = self.xywh2abcd(xywh, im0.shape)
            obj.label = det.cls
            obj.probability = det.conf
            obj.is_grounded = False
            output.append(obj)
            
        return output

    def torch_thread(self, weights, img_size, conf_thres=0.7, iou_thres=0.45):

        print("Intializing Network...")

        model = RTDETR(weights)

        while not self.exit_signal:
            if self.run_signal:
                lock.acquire()

                img = cv2.cvtColor(self.image_net, cv2.COLOR_BGRA2RGB)
                
                resized_img = self.scale_image(img, 0.5)
                
                
                # https://docs.ultralytics.com/modes/predict/#video-suffixes
                det = model.predict(img, verbose=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

                # ZED CustomBox format (with inverse letterboxing tf applied)
                self.detections = self.detections_to_custom_box(det, self.image_net)
                lock.release()
                self.run_signal = False
            sleep(0.01)
            
    def scale_image(self, img, scale_factor):
        if img is not None:
            # Get the current dimensions of the image
            height, width = img.shape[:2]

            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize the image
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            return resized_img
        else:
            print("Image not loaded properly")
            return None

    def run_detection(self): #? Previosly while viewer.is_available() and not exit_signal()
        self.zed.grab(self.runtime_params)
        # -- Get the image
        lock.acquire()
        self.zed.retrieve_image(self.image_left_tmp, sl.VIEW.LEFT)
        self.image_net = self.image_left_tmp.get_data()
        lock.release()
        self.run_signal = True

        # -- Detection running on the other thread
        while self.run_signal:
            sleep(0.001)

        # Wait for detections
        lock.acquire()
        # -- Ingest detections
        self.zed.ingest_custom_box_objects(self.detections)
        lock.release()
        
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            print("sss")
            self.zed.retrieve_objects(self.objects_, self.obj_runtime_param)

        #Ingest camera orientation info
        self.zed.get_position(self.zed_pose, sl.REFERENCE_FRAME.WORLD)
        rotation = self.zed_pose.get_rotation_vector()
        translation = self.zed_pose.get_translation(self.py_translation)

        all_cones = ConeMap()
        single_cone = Cone()

        #message for camera localizationd
        single_cone.pose.pose.orientation.w = rotation[0]
        single_cone.pose.pose.position.x = translation.get()[0]
        single_cone.pose.pose.position.y = translation.get()[1]
        single_cone.pose.pose.position.z = translation.get()[2]
        all_cones.cones.append(single_cone)
        
        if self.objects_.is_new:
            obj_array = self.objects_.object_list
            print(str(len(obj_array))+" Object(s) detected\n")
            if len(obj_array) > 0 :
                first_object = obj_array[0]
                print("First object attributes:")
                print(" Label '"+repr(first_object.label)+"' (conf. "+str(int(first_object.confidence))+"/100)")
                if self.obj_param.enable_tracking :
                    print(" Tracking ID: "+str(int(first_object.id))+" tracking state: "+repr(first_object.tracking_state)+" / "+repr(first_object.action_state))
                position = first_object.position
                velocity = first_object.velocity
                dimensions = first_object.dimensions
                print(" 3D position: [{0},{1},{2}]\n Velocity: [{3},{4},{5}]\n 3D dimentions: [{6},{7},{8}]".format(position[0],position[1],position[2],velocity[0],velocity[1],velocity[2],dimensions[0],dimensions[1],dimensions[2]))
                if first_object.mask.is_init():
                    print(" 2D mask available")

                print(" Bounding Box 2D ")
                bounding_box_2d = first_object.bounding_box_2d
                for it in bounding_box_2d :
                    print("    "+str(it),end='')
                print("\n Bounding Box 3D ")
                bounding_box = first_object.bounding_box
                for it in bounding_box :
                    print("    "+str(it),end='')

        #Create messages and send
        for object in self.objects_.object_list:
            print("aa")
            single_cone.id = object.id
            single_cone.confidence = object.confidence
            #single_cone.colour = int(object.label[0])
            #single_cone.pose.covariance = object.position_covariance
            single_cone.pose.pose.position.x = object.position[0]
            single_cone.pose.pose.position.y = object.position[1]
            single_cone.pose.pose.position.z = object.position[2]
            single_cone.radius = object.dimensions[0]/2
            single_cone.height = object.dimensions[1]
            all_cones.cones.append(single_cone)

        self.publisher.publish(all_cones)
        string_output = ""
        for cone_item in all_cones.cones:
            string_output += "s"
        print(string_output)

def main(args=None):
    rclpy.init(args=args)
    cone_detection = detection()
    rclpy.spin(cone_detection)
    cone_detection.exit_signal = True
    cone_detection.zed.close()
    cone_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()