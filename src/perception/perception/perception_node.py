
import math
from math import cos, isinf, isnan, radians, sin, sqrt

import time
import cv2
import numpy as np

from cv_bridge import CvBridge
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher

from custom_sys_msgs.msg import Cone, ConeDetectionStamped
from obr_msgs.msg import Cone2, ConeArray, Label
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, PointStamped

from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
import tf_transformations, tf2_geometry_msgs

#from rclpy.impl.logging_severity import LoggingSeverity


from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from .rect import Rect, draw_box

from typing import Callable, List, Tuple
from .rtdetr_wrapper import RTdetrWrapper


# translate ROS image messages to OpenCV
cv_bridge = CvBridge()

MAX_RANGE = 16  # m
MIN_RANGE = 0.4  # m

Colour = Tuple[int, int, int]

# # Define display colors
# YELLOW_DISP_COLOUR: Colour = (0, 255, 255)  # bgr - yellow
# BLUE_DISP_COLOUR: Colour = (255, 0, 0)      # bgr - blue
# ORANGE_DISP_COLOUR: Colour = (0, 80, 255)   # bgr - orange
# UNKNOWN_DISP_COLOUR: Colour = (0, 0, 0)     # bgr - black

# Cone display parameters
CONE_DISPLAY_PARAMETERS: List[Colour] = [
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (0, 80, 255),     # orange
    (0, 80, 255),     # orange
    (0, 0, 0)         # black
]

ConeMsgColour = int  # define arbitrary variable type


def cone_distance_old(
    colour_frame_cone_bounding_box: Rect,
    depth_frame: np.ndarray,
) -> Tuple[float, Rect]:
    """
    Calculate the distance to the cone using a region of interest in the depth image.
    """
    scale: int = 2

    # resize depth frame
    depth_frame = cv2.resize(depth_frame, (0, 0), fx=scale, fy=scale)
    # resize bounding box
    colour_frame_cone_bounding_box = colour_frame_cone_bounding_box.scale(scale)

    # get center as roi
    y_height = int(colour_frame_cone_bounding_box.height / 5)
    depth_rect = Rect(
        x=colour_frame_cone_bounding_box.center.x - 3,
        y=colour_frame_cone_bounding_box.center.y + y_height,
        width=6,
        height=6,
    )
    depth_roi: np.ndarray = depth_rect.as_roi(depth_frame)

    # filter out nans
    depth_roi = depth_roi[~np.isnan(depth_roi) & ~np.isinf(depth_roi)]
    return np.mean(depth_roi), depth_rect

def cone_distance_and_position_test(bounding_box: Rect, depth_frame: np.ndarray, depth_camera_info: CameraInfo) -> Tuple[float, Tuple[float, float, float]]:
    # Use depth camera's intrinsic parameters
    fx = depth_camera_info.k[0]  # Focal length in x
    fy = depth_camera_info.k[4]  # Focal length in y
    cx = depth_camera_info.k[2]  # Optical center x
    cy = depth_camera_info.k[5]  # Optical center y

    # Define and extract the ROI
    roi_x = bounding_box.center.x - 3
    roi_y = bounding_box.center.y + int(bounding_box.height / 5)
    roi_width = 6
    roi_height = 6
    depth_roi = depth_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Filter and compute average depth
    valid_depths = depth_roi[~np.isnan(depth_roi) & ~np.isinf(depth_roi)]
    if valid_depths.size == 0:
        return float('nan'), (float('nan'), float('nan'), float('nan'))

    average_depth = np.mean(valid_depths)

    # Calculate the 3D position of the cone
    X = (roi_x - cx) * average_depth / fx
    Y = (roi_y - cy) * average_depth / fy
    Z = average_depth

    return average_depth, (X, Y, Z)

def interpolate_point(data, x, y, max_kernel_size=30):
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

def cone_distance_and_position(bounding_box: Rect, depth_frame: np.ndarray, depth_camera_info: CameraInfo, depth_confidence_frame: np.ndarray, confidence_threshold: float = 60.0) -> Tuple[float, Tuple[float, float, float]]:
    #print(depth_camera_info)
    # Use depth camera's intrinsic parameters
    fx = depth_camera_info.k[0]  # Focal length in x
    fy = depth_camera_info.k[4]  # Focal length in y
    cx = depth_camera_info.k[2]  # Optical center x
    cy = depth_camera_info.k[5]  # Optical center y
    
    #fx = 638.01
    #fy= 348.69

    # Define and extract the ROI
    roi_x = bounding_box.center.x - 3
    roi_y = bounding_box.center.y + int(bounding_box.height / 5)
    roi_width = 6
    roi_height = 6
    depth_roi = depth_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    
    #* Confidence map gives every pixel (X, Y) in the image a value in the range [1,100], pixels having a value close to 100 are not to be trusted.
    #* https://www.stereolabs.com/docs/depth-sensing/confidence-filtering/
    confidence_roi = depth_confidence_frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    
    # Create a mask for thresholding confident depths
    confidence_mask = confidence_roi < confidence_threshold


    if np.any(np.isnan(depth_roi[confidence_mask])) or np.any(np.isinf(depth_roi[confidence_mask])):
        # Interpolate NaN or infinite values where confidence is high
        for y in range(depth_roi.shape[0]):
            for x in range(depth_roi.shape[1]):
                if confidence_mask[y, x] and (np.isnan(depth_roi[y, x]) or np.isinf(depth_roi[y, x])):
                    interpolated_value = interpolate_point(depth_roi, x, y)  # Ensure this function handles 2D arrays
                    if interpolated_value is not None:
                        depth_roi[y, x] = interpolated_value

    # Extract confident depths using the mask
    confident_depths = depth_roi[confidence_mask]

    # Check if there are valid depths after filtering
    if confident_depths.size == 0:
        return float('nan'), (float('nan'), float('nan'), float('nan'))
    
    if np.any(np.isnan(confident_depths)) or np.any(np.isinf(confident_depths)):
        print("Error: NaN value occurred after interpolation")
        return float('nan'), (float('nan'), float('nan'), float('nan'))


    average_depth = np.mean(confident_depths[~np.isnan(confident_depths) & ~np.isinf(confident_depths)])

    # Calculate the 3D position of the cone
    # X-axis (Forward): The depth (average_depth) from the camera to the object. 
    # Y-axis (Lateral): The horizontal offset from the center of the camera's view. 
    # Z-axis (Vertical): The vertical offset from the center of the camera's view.

    x_pos = float(average_depth)  # Depth: Forward direction from the camera to the object
    y_pos = -float((roi_x - cx) * average_depth / fx)  # Negate for left-right direction if necessary
    z_pos = float((roi_y - cy) * average_depth / fy)  # Vertical offset: up-down direction


    return average_depth, (x_pos, y_pos, z_pos)


def cone_bearing(bounding_box: Rect, rgb_camera_info: CameraInfo) -> float:
    cx = rgb_camera_info.k[2]  # Optical center x from RGB CameraInfo
    fx = rgb_camera_info.k[0]  # Focal length in x from RGB CameraInfo

    # Calculate the x-coordinate of the cone's center in the RGB image
    cone_center_x = bounding_box.center.x

    # Bearing calculation using the intrinsic parameters of the RGB camera
    bearing = math.atan((cone_center_x - cx) / fx)
    
    return math.degrees(bearing)  # Convert radians to degrees if needed

def cone_msg_3d(
    position: Tuple[float, float, float],
    colour: int,
) -> Cone:
    #print("cone_msg_3d: ",position)
    location = Point(
        x=position[0],
        y=position[1],
        z=position[2]
    )
    #if colour == Cone.ORANGE_SMALL:
    #    colour = Cone.ORANGE_BIG

    return Cone(
        location=location,
        color=colour,
    )
    
    
def cone2_msg(position: Tuple[float, float, float], label: int, confidence: float) -> Cone2:
    """
    Create a Cone message.

    :param position: A tuple representing the (x, y, z) coordinates of the cone.
    :param label: An integer representing the label of the cone.
    :param confidence: A float representing the confidence of the detection.

    :return: A Cone message.
    """
    cone_msg = Cone2()
    cone_msg.position = Point(x=position[0], y=position[1], z=position[2])
    cone_msg.label = label
    cone_msg.confidence = confidence

    return cone_msg
    
class VisionProcessor(Node):
    end: float = 0.0

    distortion_coefficients = None
    matrix_coefficients = None
    
    # fps = FPSHandler()

    def __init__(
        self,
        get_bounding_boxes_callable: Callable[[np.ndarray], List[Tuple[Rect, ConeMsgColour, Colour]]],
        enable_cv_filters: bool = False,
    ):
        super().__init__("vision_processor_node")

        # declare ros param for debug images
        self.declare_parameter("debug_bbox", True)
        self.declare_parameter("debug_depth", False)
        self.debug_bbox: bool = self.get_parameter("debug_bbox").value
        self.debug_depth: bool = self.get_parameter("debug_depth").value
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        #LOG_LEVEL: str = self.declare_parameter("log_level", "DEBUG").value
        # self.
        #log_level: int = getattr(rclpy.impl.logging_severity.LoggingSeverity, LOG_LEVEL.upper())
        # self.get_logger().set_level()


        # subscribers
        colour_sub = message_filters.Subscriber(self, Image, "/zed/zed_node/rgb_raw/image_raw_color")
        colour_camera_info_sub = message_filters.Subscriber(self, CameraInfo, "/zed/zed_node/rgb_raw/camera_info")
        depth_camera_info_sub = message_filters.Subscriber(self, CameraInfo, "/zed/zed_node/depth/camera_info")
        depth_sub = message_filters.Subscriber(self, Image, "/zed/zed_node/depth/depth_registered")
        depth_confidence_sub = message_filters.Subscriber(self, Image, "/zed/zed_node/confidence/confidence_map")

        synchronizer = message_filters.ApproximateTimeSynchronizer(
            fs=[colour_sub, colour_camera_info_sub, depth_sub, depth_camera_info_sub, depth_confidence_sub],
            queue_size=10,
            slop=0.1,
        )
        synchronizer.registerCallback(self.callback)

        # publishers
        self.detection_publisher: Publisher = self.create_publisher(ConeDetectionStamped, "/vision/cone_detection", 1)
        self.cone_array_publisher: Publisher = self.create_publisher(ConeArray, "/cones/positions", 1)

        if self.debug_bbox:
            self.debug_img_publisher: Publisher = self.create_publisher(Image, "/debug_imgs/vision_bbs_img", 1)
        if self.debug_depth:
            self.depth_debug_img_publisher: Publisher = self.create_publisher(Image, "/debug_imgs/vision_depth_img", 1)
        #if self.debug_rt_detect:
        #    self.depth_debug_img_publisher: Publisher = self.create_publisher(PoseArray, "/debug_imgs/cone_detection", 1)

        # set which cone detection this will be using
        self.enable_cv_filters = enable_cv_filters
        self.get_bounding_boxes_callable = get_bounding_boxes_callable
                                        
        self.end = time.perf_counter()
        self.get_logger().info("                                _   _                                _      ")
        self.get_logger().info("                               | | (_)                              | |     ")
        self.get_logger().info("  _ __   ___ _ __ ___ ___ _ __ | |_ _  ___  _ __     _ __   ___   __| | ___ ")
        self.get_logger().info(" | '_ \ / _ \ '__/ __/ _ \ '_ \| __| |/ _ \| '_ \   | '_ \ / _ \ / _` |/ _ \\")
        self.get_logger().info(" | |_) |  __/ | | (_|  __/ |_) | |_| | (_) | | | |  | | | | (_) | (_| |  __/")
        self.get_logger().info(" | .__/ \___|_|  \___\___| .__/ \__|_|\___/|_| |_|  |_| |_|\___/ \__,_|\___|")
        self.get_logger().info(" | |                     | |                                                ")
        self.get_logger().info(" |_|                     |_|                                                ")
        self.get_logger().info("")
        self.get_logger().info("PARAMS: debug_bbox: " + str(self.debug_bbox) + "\t debug_depth: " + str(self.debug_depth))
        
    def is_nan_point(self, point):
        return math.isnan(point.x) or math.isnan(point.y) or math.isnan(point.z)

    def transform_cone(self, cone_msg, timestamp):
        from_frame_rel = 'zed_left_camera_frame'
        to_frame_rel = 'track'
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return
        
        #print(t.transform)
        
        # Apply the transformation to the Cone2 message
        transformed_cone2_msg = Cone2(
            position=Point(
                x=cone_msg.position.x + t.transform.translation.x,
                y=cone_msg.position.y + t.transform.translation.y,
                z=cone_msg.position.z + t.transform.translation.z
            ),
            label=cone_msg.label,
            confidence=cone_msg.confidence
        )
        
        return transformed_cone2_msg

        # msg = Twist()
        # scale_rotation_rate = 1.0
        # msg.angular.z = scale_rotation_rate * math.atan2(
        #     t.transform.translation.y,
        #     t.transform.translation.x)

        # scale_forward_speed = 0.5
        # msg.linear.x = scale_forward_speed * math.sqrt(
        #     t.transform.translation.x ** 2 +
        #     t.transform.translation.y ** 2)
        # try:
        #     # Create a PointStamped object
        #     point_stamped = PointStamped()
        #     point_stamped.point = cone_msg.position
        #     point_stamped.header.frame_id = 'zed_left_camera_frame'  # Use the appropriate frame_id
        #     point_stamped.header.stamp =             trans_stamped = self.tf_buffer.lookup_transform('track', 'zed_left_camera_frame', timestamp)


        #     # Look up the transformation
        #     trans_stamped = self.tf_buffer.lookup_transform('track', 'zed_left_camera_frame', rclpy.time.Time())

        #     # Transform the PointStamped object
        #     transformed_cone = tf2_geometry_msgs.do_transform_point(point_stamped, trans_stamped)

        #     # Update the cone message with the new position
        #     cone_msg.position = transformed_cone.point

        #     return cone_msg

        # except (LookupException, ConnectivityException, ExtrapolationException) as e:
        #     # Handle exceptions, e.g., log an error
        #     self.get_logger().error(f"Error transforming cone: {e}")
        #     return None

        
    def callback(self, colour_msg: Image, colour_camera_info_msg: CameraInfo, depth_msg: Image, depth_camera_info_msg: CameraInfo, depth_confidence_msg: Image):
        
        if self.distortion_coefficients is None or self.matrix_coefficients is None:
            # extract distortion coefficients and calibration matrix from camera info
            self.distortion_coefficients = np.array(colour_camera_info_msg.d)
            self.matrix_coefficients = np.array(colour_camera_info_msg.k).reshape(3, 3)
            self.get_logger().info("Distortion coefficients and calibration matrix read and set from camera.")
            self.get_logger().info("Recalibrate, if factory calibration..")
            self.get_logger().debug("Distortion Coefficients: " + str(self.distortion_coefficients))
            self.get_logger().debug("Calibration Matrix: " + str(self.matrix_coefficients))

        colour_frame: np.ndarray = cv_bridge.imgmsg_to_cv2(colour_msg, desired_encoding="bgr8")
        depth_confidence_frame = cv_bridge.imgmsg_to_cv2(depth_confidence_msg, desired_encoding="32FC1")
        depth_frame: np.ndarray = cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        detected_cones: List[Cone] = []
        
        #detected_cones2: List[Cone2] = []
        cone_array_msg = ConeArray()  # Create a ConeArray message
        cone_array_msg.header = Header(frame_id="zed_left_camera_optical_frame", stamp=self.get_clock().now().to_msg())

        for bounding_box, cone_colour, display_colour in self.get_bounding_boxes_callable(colour_frame):
            # # filter by height
            # if bounding_box.tl.y < colour_camera_info_msg.height / 2.5:
            #     print(bounding_box.tl.y, "<", colour_camera_info_msg.height / 4 )    
            #     print("sorted box 1")
            #     continue
            # # filter on area
            # if bounding_box.area < 10:
            #     print("sorted box 2")

            #     continue
            # # filter by aspect ratio
            # if bounding_box.aspect_ratio < 0.4 or bounding_box.aspect_ratio > 1.5:
            #     print("sorted box 3")

            #     continue

            result = cone_distance_and_position(bounding_box, depth_frame, depth_camera_info_msg, depth_confidence_frame)
            
            if result is None:
                # Handle the error case, for example:
                continue  # Skip this iteration
            else:
                distance, position = result

            bearing = cone_bearing(bounding_box, colour_camera_info_msg)
            
            #! Check if the cone is within the front-facing angle threshold, issues occur at edges of FOV
            if abs(bearing) > 30: #deg
                continue  
            
            # filter on distance
            if distance > MAX_RANGE or distance < MIN_RANGE:
                continue

            detected_cones.append(cone_msg_3d(position, cone_colour))
            
            
            if self.debug_bbox:
                draw_box(colour_frame, box=bounding_box, colour=display_colour, distance=distance)

            self.get_logger().debug("Range: " + str(round(distance, 2)) + "\t Bearing: " + str(round(bearing, 2)))

                # Create a Label message
            label_msg = Label()
            # Set properties of label_msg as needed, for example:
            label_msg.label = cone_colour  # if 'type' is a field in the Label message
            
            # Create a Cone2 message
            cone_msg = Cone2(
                position=Point(x=position[0], y=position[1], z=position[2]),
                label=label_msg,  # Replace with appropriate label
                confidence=1.0  # Replace with appropriate confidence value
            )
            #print(cone_msg)
            #print(cone_array_msg)
            
            cone_array_msg.cones.append(cone_msg)
            
            # transformed_cone_msg = self.transform_cone(cone_msg, cone_array_msg.header.stamp)

            # if transformed_cone_msg is not None:
            #     cone_array_msg.cones.append(transformed_cone_msg)


        
        # Create a ConeDetectionStamped message
        detection_msg = ConeDetectionStamped(
            header=Header(
                frame_id="zed_left_camera_frame",
                stamp=self.get_clock().now().to_msg()  # Assign a new timestamp
            ),
            cones=detected_cones,
        )
        self.detection_publisher.publish(detection_msg)


        # Publish the ConeArray message
        self.cone_array_publisher.publish(cone_array_msg)

        if self.debug_bbox:
            debug_msg = cv_bridge.cv2_to_imgmsg(colour_frame, encoding="bgr8")
            debug_msg.header = Header(frame_id="zed2", stamp=colour_msg.header.stamp)
            self.debug_img_publisher.publish(debug_msg)

def main(args=None):    
    MODEL_PATH = "src/perception/models/yolov8n_cones.pt"
    
    CONFIDENCE = 0.70  # higher = tighter filter
    IMGSZ = 640 # Can be read in CameraInfo or set in common.yaml -> zed2
    wrapper = RTdetrWrapper(MODEL_PATH, CONFIDENCE, IMGSZ)

    def get_torch_bounding_boxes(
        colour_frame: np.ndarray,
    ) -> List[Tuple[Rect, ConeMsgColour, Colour]]:  # bbox, msg colour, display colour
        bounding_boxes: List[Tuple[Rect, ConeMsgColour, Colour]] = []
        data = wrapper.inference(colour_frame)

        for detection in data:
            cone_colour = int(detection[0])
            bounding_box = Rect(
                int(detection[1]),
                int(detection[2]),
                int(detection[3] - detection[1]),
                int(detection[4] - detection[2]),
            )
            bounding_boxes.append((bounding_box, cone_colour, CONE_DISPLAY_PARAMETERS[cone_colour]))
        return bounding_boxes

    rclpy.init(args=args)
    node = VisionProcessor(get_torch_bounding_boxes)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()