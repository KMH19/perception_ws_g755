import rclpy
import math
from cv_bridge import CvBridge
import torch
import time

import cv2 as cv
import pandas as pd
from ultralytics import RTDETR, YOLO
from bytetrack_realtime.byte_tracker import ByteTracker

from tf_transformations import quaternion_multiply, quaternion_matrix


# Synchronization of topics
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from vision_msgs.msg import Detection2D, Detection2DArray

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PoseStamped, do_transform_pose
from geometry_msgs.msg import Pose

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from builtin_interfaces.msg import Duration


from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node

# Local module (relative import .)
from .static_definitions import RTDETR_MAP_FS
from .sort import *


#TODO : fix markers (or scrap them)
#TODO : Consider mapping + pcl


# Display the processed image (you may want to publish it or save it)
import numpy as np

#? print gpu speed
#import GPUtil

class Perception(Node):
    def __init__(self, model_path: str) -> None:
        super().__init__('perception_module')
        self.node = rclpy.create_node('image_subscriber_node')
        self.image_subscription = Subscriber(self, Image, '/zed/zed_node/rgb_raw/image_raw_color')
        self.depth_subscription = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')#, 
        
        self.camera_pose_subscription = Subscriber(self, PoseStamped, '/zed/zed_node/pose')
        self.blue_cones_pos_publisher = self.create_publisher(PoseArray, '/perception_node/blue_cones_position', 10)
        self.yellow_cones_pos_publisher = self.create_publisher(PoseArray, '/perception_node/yellow_cones_position', 10)
        self.inferenced_image_publisher = self.create_publisher(Image, '/perception_node/inferenced_image', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/perception_node/visualization_marker_array', 10)
        
        self.marker_id = 0  # Initialize marker ID
        self.yellow_markers = {}  # Dictionary to store existing yellow markers
        self.blue_markers = {}    # Dictionary to store existing blue markers
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.fixed_frame = 'map' # zed_left_camera_frame

        self.color_map = {'yellow_cone': (0, 255, 255), 'blue_cone': (255, 0, 0), 'unknown_cone': (255, 0, 255)}

        self.queue_size = 5

        self.cv_bridge = CvBridge()

        self.ts = ApproximateTimeSynchronizer(
            [self.image_subscription, self.depth_subscription, self.camera_pose_subscription],
            self.queue_size,
            0.001,  # defines the delay (in seconds) with which messages can be synchronized
        )
        self.ts.registerCallback(self.image_callback)

        ## Load pretrained model
        # self.model = RTDETR(model_path)
        self.model = RTDETR(model_path)
        
        # ByteTracker
        self.tracker = ByteTracker(track_thresh=0.7, track_buffer=20, match_thresh=0.7)
        self.bytetracking = False

        self.mot_tracker = Sort() 
        self.sorttracking = False
        
        self.publish_markers = True
        
        self.frame_count = 0
        self.last_time = 0
        
        self.debugging_tracker = False
        
        print("perception node: Initiated")
        pass
    
    # def transform_pose_to_map(self, pose, camera_pose):
    #     try:
    #         # Use a slightly earlier timestamp
    #         earlier_stamp = camera_pose.header.stamp
    #         earlier_stamp.sec -= 2  # Adjust this based on your system's latency

    #         # Check if transform is available
    #         if not self.tf_buffer.can_transform('map', camera_pose.header.frame_id, earlier_stamp):
    #             self.get_logger().error('Transform not available for earlier timestamp')
    #             return None

    #         pose_stamped = PoseStamped()
    #         pose_stamped.pose = pose
    #         pose_stamped.header.frame_id = camera_pose.header.frame_id
    #         pose_stamped.header.stamp = earlier_stamp

    #         # Transform to the map frame
    #         pose_transformed = self.tf_buffer.transform(pose_stamped, 'map')
    #         return pose_transformed.pose
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
    #         self.get_logger().error(f"Failed to transform pose: {e}")
    #         return None

        
    def distance_between_points(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2) ** 0.5

    def is_far_from_existing_markers(self, cone_type, new_marker_pose):
        """Check if the new marker is at least 1m away from markers of the same type."""
        existing_markers = self.yellow_markers if cone_type == 'yellow_cone' else self.blue_markers

        for _, pose in existing_markers.items():
            distance = self.distance_between_points(new_marker_pose.position, pose.position)
            if distance < 0.5:
                return False
        return True


    def image_callback(self, rgb_msg: Image, depth_msg: Image, cam_pose_msg: PoseStamped) -> None:
        assert rgb_msg.header.stamp == depth_msg.header.stamp

        #image = self._bridge.imgmsg_to_cv2(rgb_msg)#, desired_encoding='passthrough')
        #depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='64FC1')
        try:
            # Convert ROS Image message to OpenCV image
            frame_rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            frame_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")

            #print(frame_depth.dtype.itemsize * 8)
            # 32 bits


            # Process the image
            processed_frame, yellow_cones, blue_cones = self.process_generic(frame_rgb, frame_depth, cam_pose_msg)

            depth_array = np.array(frame_depth, dtype=np.float32)


            depth_array = depth_array #/10 <- for visualization this range is better.

            # Get the center coordinates of the image
            height, width, _ = frame_rgb.shape
            center_x, center_y = width // 2, height // 2

            # Ensure processed_frame is a valid image
            #cv.circle(processed_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            #depth_at_center = frame_depth[center_y, center_x]
            #print("Depth at center:", depth_at_center, " m")
            debug = False
            if debug:
                cv.imshow("ZED2 RD-DETR RGB STREAM", processed_frame)
                cv.imshow("ZED2 RD-DETR DEPTH STREAM", depth_array)
                cv.waitKey(1)

            self.publish_cones_pose_arrays(yellow_cones, blue_cones)


            # Convert processed_frame back to ROS Image message
            inferenced_image_msg = self.cv_bridge.cv2_to_imgmsg(processed_frame, "bgr8")

            # Set the header of the inferenced image message
            inferenced_image_msg.header.stamp = self.get_clock().now().to_msg()
            inferenced_image_msg.header.frame_id = "camera_frame"  # Set the appropriate frame_id

            # Publish the processed image
            self.inferenced_image_publisher.publish(inferenced_image_msg)

        except Exception as e:
            print(e)
            #self.node.get_logger().error("Error processing image: %s", str(e))


    def draw_bbox(self, frame, df_prediction):
        """Draw bounding boxes on a frame based on the predictions."""
        for _, row in df_prediction.iterrows():
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            confidence = row['confidence']
            classification = row['name'] if 'name' in df_prediction.columns else 'Unknown'

            bbox_color = self.color_map.get(classification, (0, 255, 0))  # Default to green if not found


            label = f"{classification} {confidence:.2f}"
            cv.putText(frame, label, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # blue text
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), bbox_color, 2)  # green rectangle
        return frame
    
    def convert_ltrb_to_ltwh(self, ltrb):
        left, top, right, bottom = ltrb
        width = right - left
        height = bottom - top
        return [left, top, width, height]

    def process_generic(self, frame_rgb, frame_depth, cam_pose_msg, conf_thresh=0.65):
        """Process a single frame and return the processed frame."""

        try:
            ultralytics_results = self.model.predict(frame_rgb, verbose=False)[0]
            # rest of your code...
        except Exception as e:
            print("An error occurred:", e)
            
        #print(ultralytics_results)
        

        #data = []

        # Lists to store poses of different cone types
        yellow_cones = []
        blue_cones = []
        #! orange_cones = []
                
        if self.bytetracking:
            
            # Update frame count
            self.frame_count += 1

            # Calculate time difference
            current_time = time.time()
            time_diff = current_time - self.last_time

            # Avoid division by zero and update last_time
            if time_diff > 0:
                fps = self.frame_count / time_diff
                self.last_time = current_time
                self.frame_count = 0  # Reset frame count after calculating FPS

            # Display FPS on the frame
            cv.putText(frame_rgb, f"FPS: {fps:.2f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            detections_cpu = [(box.cpu().numpy(), int(cls.cpu().numpy()), conf.cpu().numpy()) for box, cls, conf in zip(ultralytics_results.boxes.xyxy, ultralytics_results.boxes.cls, ultralytics_results.boxes.conf)]
            
            tracks = self.tracker.update(detections=detections_cpu)
            
            cv.putText(frame_rgb, "cones tracked: "+str(len(tracks)), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1) 

            for track in tracks:
                track_id = track.track_id
                ltrb = track.ltrb # bbox
                score = track.score
                det_class = track.det_class
                
                if self.debugging_tracker: 
                    print("track id:", track_id)
                    print("bbox:",ltrb)
                    print("score:",score)
                    print("class:",det_class)
                
                ltwh = self.convert_ltrb_to_ltwh(ltrb) #? Tested conversion is OK, our native format is ltwh.

                xmin, ymin, xmax, ymax = map(int, ltwh)

                classification = RTDETR_MAP_FS[int(det_class)]

                # Determine color based on classification
                if classification == 'yellow_cone':
                    color = (0, 255, 255)
                    yellow_cones.append([classification, self.estimate_cones_poses(ltwh, frame_depth)])
                elif classification == 'blue_cone':
                    color = (255, 0, 0)
                    blue_cones.append([classification, self.estimate_cones_poses(ltwh, frame_depth)])
                else:
                    color = (255, 0, 255)
                    
                # Draw bounding box, label, and track ID on frame_rgb
                cv.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), color, 2)
                label_text = f"{classification}"
                cv.putText(frame_rgb, label_text, (xmin, ymin - 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 
                label_text = f"#{track_id}: {score:.2f}"
                cv.putText(frame_rgb, label_text, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 
                
        elif self.sorttracking:            

            # Assuming ultralytics_results is your result object from Ultralytics detection
            boxes = ultralytics_results.boxes.xyxy  # Get the detection boxes tensor
            scores = ultralytics_results.boxes.conf  # Get the scores tensor
            labels = ultralytics_results.boxes.cls
            
            #labels = RTDETR_MAP_FS[int(ultralytics_results.boxes.cls)]

            # Convert from GPU to CPU and then to numpy (if they are on GPU)
            boxes = boxes.cpu().numpy() if boxes.is_cuda else boxes.numpy()
            scores = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
            labels = labels.cpu().numpy() if labels.is_cuda else labels.numpy()

            # Combine boxes and scores to match the expected format of the tracker
            dets = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis]))

            # Now dets is in the format [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score], ...]
            #print(dets)

            track_bbs_ids = self.mot_tracker.update(dets)
            
            #print(track_bbs_ids)
            
            for track in track_bbs_ids:
                #print("!")

                xmin, ymin, xmax, ymax, score, track_id, class_id = track[:7]
                
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                #print(xmin, ymin, xmax, ymax, score, class_id, track_id)


                classification = RTDETR_MAP_FS[int(class_id)]

                # Determine color based on classification
                if classification == 'yellow_cone':
                    color = (0, 255, 255)
                    yellow_cones.append([classification, self.estimate_cones_poses(track[:4], frame_depth)])
                elif classification == 'blue_cone':
                    color = (255, 0, 0)
                    blue_cones.append([classification, self.estimate_cones_poses(track[:4], frame_depth)])
                else:
                    color = (255, 0, 255)
                    
                # Draw bounding box, label, and track ID on frame_rgb
                cv.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), color, 2)
                label_text = f"{classification}"
                cv.putText(frame_rgb, label_text, (xmin, ymin - 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1) 
                label_text = f"#{track_id}: {score:.2f}"
                cv.putText(frame_rgb, label_text, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)   

        elif self.publish_markers:
                                    # Update frame count
            
            distance_tolerance = 0.05  # meters
                        
            
            self.frame_count += 1

            # Calculate time difference
            current_time = time.time()
            time_diff = current_time - self.last_time

            # Avoid division by zero and update last_time
            if time_diff > 0:
                fps = self.frame_count / time_diff
                self.last_time = current_time
                self.frame_count = 0  # Reset frame count after calculating FPS

            # Display FPS on the frame
            cv.putText(frame_rgb, f"FPS: {fps:.2f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            cv.putText(frame_rgb, "cones located: "+str(len(ultralytics_results)), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1) 

            
            for box, label, conf in zip(ultralytics_results.boxes.xyxy, ultralytics_results.boxes.cls, ultralytics_results.boxes.conf):
                if conf < conf_thresh:
                    continue  # Skip detections below confidence threshold

                # Convert the bounding box tensor to CPU and then to a list
                box = box.cpu().numpy().tolist()
                xmin, ymin, xmax, ymax = map(int, box)

                classification = RTDETR_MAP_FS[int(label)]

                # Determine color based on classification
                if classification == 'yellow_cone':
                    color = (0, 255, 255)
                    cone_pose = self.estimate_cones_poses(box, frame_depth, cam_pose_msg)

                    # Transform the pose to the map frame
                    #trans_pose = self.transform_pose(cone_pose, 'zed_left_camera_frame', 'map')
                    trans_pose_map = self.transform_pose(cone_pose, 'zed_left_camera_frame', 'odom')
                
                    yellow_distance_to_cone = cone_pose.position.x
                    print(yellow_distance_to_cone)
                    #print("yel dist: ",yellow_distance_to_cone)
                    yellow_cones.append([classification, cone_pose])
                    # Check if closest cones are within thresholds and publish markers
                    if abs(yellow_distance_to_cone - 1.0) <= distance_tolerance:
                        # Publish marker for closest yellow cone
                        self.publish_marker('yellow_cone', trans_pose_map)
                        
                elif classification == 'blue_cone':
                    color = (255, 0, 0)
                    cone_pose = self.estimate_cones_poses(box, frame_depth, cam_pose_msg)
                    
                                        # Transform the pose to the map frame
                    #trans_pose = self.transform_pose(cone_pose, 'zed_left_camera_frame', 'map')
                    trans_pose_map = self.transform_pose(cone_pose, 'zed_left_camera_frame', 'odom')

                    #pose_in_map_frame = self.tf_pose(pose, "map", "zed_left_camera_frame")
                    #pose = pose_in_map_frame
                    blue_distance_to_cone = cone_pose.position.x
                    #print("blu dist: ",blue_distance_to_cone)

                    blue_cones.append([classification, cone_pose])
                    if abs(blue_distance_to_cone - 1.0) <= distance_tolerance:
                        # Publish marker for closest blue cone
                        self.publish_marker('blue_cone', trans_pose_map)
                else:
                    color = (255, 0, 255)
                    
                    
                # Draw bounding box and label on frame_rgb
                cv.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), color, 2)
                label_text = f"{classification} {conf:.2f}"
                cv.putText(frame_rgb, label_text, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                
        else: #? No tracking
            
                        # Update frame count
            self.frame_count += 1

            # Calculate time difference
            current_time = time.time()
            time_diff = current_time - self.last_time

            # Avoid division by zero and update last_time
            if time_diff > 0:
                fps = self.frame_count / time_diff
                self.last_time = current_time
                self.frame_count = 0  # Reset frame count after calculating FPS

            # Display FPS on the frame
            cv.putText(frame_rgb, f"FPS: {fps:.2f}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            cv.putText(frame_rgb, "cones located: "+str(len(ultralytics_results)), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1) 

            
            for box, label, conf in zip(ultralytics_results.boxes.xyxy, ultralytics_results.boxes.cls, ultralytics_results.boxes.conf):
                if conf < conf_thresh:
                    continue  # Skip detections below confidence threshold

                # Convert the bounding box tensor to CPU and then to a list
                box = box.cpu().numpy().tolist()
                xmin, ymin, xmax, ymax = map(int, box)

                classification = RTDETR_MAP_FS[int(label)]

                # Determine color based on classification
                if classification == 'yellow_cone':
                    color = (0, 255, 255)
                    yellow_cones.append([classification, self.estimate_cones_poses(box, frame_depth)])
                elif classification == 'blue_cone':
                    color = (255, 0, 0)
                    blue_cones.append([classification, self.estimate_cones_poses(box, frame_depth)])
                else:
                    color = (255, 0, 255)

                # Draw bounding box and label on frame_rgb
                cv.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), color, 2)
                label_text = f"{classification} {conf:.2f}"
                cv.putText(frame_rgb, label_text, (xmin, ymin - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame_rgb, yellow_cones, blue_cones
    
    
    def transform_pose(self, input_pose, from_frame, to_frame):
        # Wait for the transform to be available
        while not self.tf_buffer.can_transform(to_frame, from_frame, rclpy.time.Time()):
            rclpy.spin_once(self.node)

        # Transform the pose
        try:
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rclpy.time.Time())
            #print("transform: ", transform)
            #print("input_pose: ", input_pose)
            transformed_pose = do_transform_pose(input_pose, transform)
            return transformed_pose
        
        except Exception as e:
            print(f"Failed to transform pose: {e}")
            return None
    
    def create_marker(self, cone_type, pose):
        if not self.is_far_from_existing_markers(cone_type, pose):
            return None

        # Create a Marker object
        marker = Marker()
        marker.header.frame_id = "map"  # Set to the appropriate frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = cone_type  # Namespace based on cone type
        marker.id = self.marker_id
        marker.type = Marker.SPHERE  # You can change the shape
        marker.action = Marker.ADD

        # Set the pose of the marker
        marker.pose = pose

        # Set the scale of the marker
        marker.scale.x = 0.3
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Set the color of the marker based on cone type
        if cone_type == 'yellow_cone':
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif cone_type == 'blue_cone':
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        marker.color.a = 1.0  # Alpha value

        # Set marker lifetime to zero for an infinite lifetime
        marker.lifetime = Duration(sec=0, nanosec=0)

        # Increment ID for the next marker
        self.marker_id += 1

        # Store the new marker's position based on its type
        if cone_type == 'yellow_cone':
            self.yellow_markers[marker.id] = pose
        elif cone_type == 'blue_cone':
            self.blue_markers[marker.id] = pose

        return marker

    def publish_marker(self, cone_type, pose):
        marker = self.create_marker(cone_type, pose)
        if marker:
            # Create a MarkerArray and add the marker to it
            marker_array = MarkerArray()
            marker_array.markers.append(marker)

            # Publish the MarkerArray
            self.marker_pub.publish(marker_array)

    def interpolate_point(self, data, x, y, max_kernel_size=30):
        for kernel_size in range(3, max_kernel_size, 2):  # Increase kernel size dynamically
            half_kernel = kernel_size // 2
            start_x = int(max(x - half_kernel, 0))
            end_x = int(min(x + half_kernel + 1, data.shape[1]))
            start_y = int(max(y - half_kernel, 0))
            end_y = int(min(y + half_kernel + 1, data.shape[0]))

            neighborhood = data[start_y:end_y, start_x:end_x]
            valid_values = neighborhood[~np.isnan(neighborhood)]

            if valid_values.size > 0:
                return float(np.median(valid_values))  # Using median here

        return None  # No valid data found within the maximum kernel size
 
    #! Not tested...
    def estimate_cones_poses(self, box, depth, cam_pose_msg, cam_width=640, cam_height=360, h_fov_deg=110, v_fov_deg=70):
        
        #print("cam_pose_msg: ", cam_pose_msg)
        
        try:
        
            # Convert FOV from degrees to radians
            h_fov = math.radians(h_fov_deg)
            v_fov = math.radians(v_fov_deg)
        
            pose = Pose()
            xmin, ymin, xmax, ymax = map(int, box)

            center_x = (xmin + xmax) / 2
            height = ymax - ymin
            one_third_height = height / 3
            center_y = ymax - one_third_height / 2

            d = float(depth[int(center_y), int(center_x)])

            if np.isnan(d):
                d = self.interpolate_point(depth, center_x, center_y)

            # Calculate angular displacement per pixel
            angle_per_pixel_x = h_fov / cam_width
            angle_per_pixel_y = v_fov / cam_height

            # Calculate angles
            angle_x = (center_x - cam_width / 2) * angle_per_pixel_x
            angle_y = (center_y - cam_height / 2) * angle_per_pixel_y

            # Modify position calculations using angles
            pose.position.x = d * math.cos(angle_x)
            pose.position.y = d * math.sin(angle_x)
            pose.position.z = 0.1 #d * math.sin(angle_y)
            
            # # Extract camera position and orientation from cam_pose_msg
            # cam_position = [cam_pose_msg.pose.position.x, cam_pose_msg.pose.position.y, cam_pose_msg.pose.position.z]
            # cam_orientation = [cam_pose_msg.pose.orientation.x, cam_pose_msg.pose.orientation.y, cam_pose_msg.pose.orientation.z, cam_pose_msg.pose.orientation.w]

            # # Convert orientation to rotation matrix and create a translation matrix
            # rotation_matrix = quaternion_matrix(cam_orientation)
            # translation_matrix = np.identity(4)
            # translation_matrix[:3, 3] = cam_position

            # # Combine rotation and translation into a single transformation matrix
            # transformation_matrix = np.dot(translation_matrix, rotation_matrix)

            # # Transform cone position to world frame
            # cone_position_camera_frame = [pose.position.x, pose.position.y, pose.position.z, 1]  # Homogeneous coordinates
            # cone_position_world_frame = np.dot(transformation_matrix, cone_position_camera_frame)

            # # Update pose with world frame position
            # pose.position.x, pose.position.y, pose.position.z = cone_position_world_frame[:3]

            #print(pose.position.x,pose.position.y,pose.position.z)

            # Adjust orientation based on the angular position
            pose.orientation.x = 0.707
            pose.orientation.y = 0.0
            pose.orientation.z = 0.707
            pose.orientation.w = 0.0
            
            #print(pose)

            return pose
    
        except Exception as e:
            print("An error occurred:", e)
            return None

    def publish_cones_pose_arrays(self, yellow_cones, blue_cones):
        """Generates and publishes posearrays for blue and yellow cones."""

        yellow_pose_array = PoseArray()
        blue_pose_array = PoseArray()

        timestamp = self.get_clock().now()

        # Assuming yellow_cones and blue_cones are lists of [classification, Pose]
        yellow_poses = [cone_pose for _, cone_pose in yellow_cones]
        blue_poses = [cone_pose for _, cone_pose in blue_cones]

        yellow_pose_array.poses = yellow_poses
        blue_pose_array.poses = blue_poses

        yellow_pose_array.header.stamp = timestamp.to_msg()
        yellow_pose_array.header.frame_id = self.fixed_frame#'zed_left_camera_frame' #'/zed_camera_center'

        blue_pose_array.header.stamp = timestamp.to_msg()
        blue_pose_array.header.frame_id = self.fixed_frame#'zed_left_camera_frame' #'/zed_camera_center'

        self.yellow_cones_pos_publisher.publish(yellow_pose_array)
        self.blue_cones_pos_publisher.publish(blue_pose_array)

    def getBbox_aslist(self, df_prediction):
        """Extract bounding boxes from the DataFrame"""
        return df_prediction[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

    def acquire_image(self):

        return

    def object_detection(self):
        return


def main(args=None):
    print('Hi from perception.')
    rclpy.init(args=args)


    try :
        perception_node = Perception(model_path="src/perception/models/rtdetr_detector.pt")
        rclpy.spin(perception_node)
    except KeyboardInterrupt :
        perception_node.get_logger().info('Stopped by Keyboard')
    finally :
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
