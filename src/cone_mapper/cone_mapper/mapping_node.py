import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from custom_sys_msgs.msg import Cone, ConeDetectionStamped, ConeWithCovariance#, State
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf_transformations
import math
import time
import sys

from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np


class ConeEKF(ExtendedKalmanFilter):
    def __init__(self, initial_state, dt):
        super().__init__(dim_x=4, dim_z=2)  # Assuming 4 state variables and 2 measurement variables
        
        self.x = initial_state
        
        # State transition matrix (assuming constant velocity model)
        self.F = np.array([[1, 0, dt, 0],  # State transition matrix must be 4x4
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],  # Measurement function must be 2x4
                           [0, 1, 0, 0]])
        
        # Initial covariance: Initial uncertainty about the state (position and velocity) of the cone.
        self.P = np.eye(4) * 10.
        
        # Measurement noise: Indicates a certain level of noise in the measurements of the cone's position.
        self.R = np.eye(2) * 0.1
        
        #Process noise: 'var' parameter should represent the expected variability in the cone's motion.
        self.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.05)  # Process noise must be 4x4
        
        # Set the Jacobian of the measurement function and the measurement function
        self.HJacobian = self.HJacobian_at
        self.Hx = self.Hx_function

    def HJacobian_at(self, x):
        """ Compute the Jacobian of the measurement function at state x """
        # If your measurement function is linear, this is just your measurement matrix
        return self.H

    def Hx_function(self, x):
        """ Compute the measurement prediction from state x """
        # Apply the measurement function to the state vector
        return np.dot(self.H, x)


    def predict(self):
        # Call the parent class predict function
        super().predict()

    def update(self, measurement):
        # Call the parent class update function
        super().update(measurement, self.HJacobian_at, self.Hx_function)


class TrackedCone:
    def __init__(self, cone, frame_id, pose, dt=0.01):

        #* Physical offset
        # self.local_x = cone.location.x - 0.2
        # self.sensor = "camera"
        
        self.local_x = cone.location.x 
        self.local_y = cone.location.y
        self.colour = cone.color
        self.frame_count = 1

        # transform detection to map
        rotation_mat = [[math.cos(pose[2]), -math.sin(pose[2])], [math.sin(pose[2]), math.cos(pose[2])]]
        map_coords = [rotation_mat[0][0] * self.local_x + rotation_mat[0][1] * self.local_y + pose[0],
                      rotation_mat[1][0] * self.local_x + rotation_mat[1][1] * self.local_y + pose[1]]
        self.map_x = map_coords[0]
        self.map_y = map_coords[1]
        #self.cov = [[1.0 / self.frame_count, 0.0], [0.0, 1.0/ self.frame_count]]
        self.yellow_count = 0
        self.blue_count = 0
        self.orange_count = 0
        
        # Initialize the EKF with the initial position and zero velocity
        initial_state = np.array([cone.location.x, cone.location.y, 0, 0])
        self.ekf = ConeEKF(initial_state, dt)
                
        

    def distance(self, other):
        # Euclidean distance from this to another TrackedCone
        diff_x = self.map_x - other.map_x
        diff_y = self.map_y - other.map_y
        return math.sqrt(diff_x * diff_x + diff_y * diff_y)

    def update(self, other, pose):
        
        # Predict the state using the EKF
        self.ekf.predict()

        # Update with new measurement using the EKF
        measurement = np.array([other.map_x, other.map_y])
        self.ekf.update(measurement)

        # Extract the state after update
        self.map_x, self.map_y, _, _ = self.ekf.x

        # The EKF's estimate is now the most up-to-date estimate of position.
        # There is no need to re-average the positions, as the EKF already provides the optimal estimate.
        # Similarly, the covariance is most accurately given by the EKF's calculation.
        self.cov = self.ekf.P[0:2, 0:2]
        
        # Update colour with camera
        if other.colour == Cone.YELLOW:
            self.yellow_count += 1
        elif other.colour == Cone.BLUE:
            self.blue_count += 1
        elif other.colour == Cone.ORANGE_BIG:
            self.orange_count += 1

        if self.yellow_count > self.blue_count and self.yellow_count > self.orange_count:
            self.colour = Cone.YELLOW
        elif self.blue_count > self.yellow_count and self.blue_count > self.orange_count:
            self.colour = Cone.BLUE
        elif self.orange_count > 10 or self.orange_count > self.yellow_count or self.orange_count > self.blue_count:
            self.colour = Cone.ORANGE_BIG
            
        # Transform map to car
        rotation_mat = [[math.cos(-pose[2]), -math.sin(-pose[2])], [math.sin(-pose[2]), math.cos(-pose[2])]]
        local_coords = [rotation_mat[0][0] * self.map_x + rotation_mat[0][1] * self.map_y - pose[0],
                        rotation_mat[1][0] * self.map_x + rotation_mat[1][1] * self.map_y - pose[1]]
        self.local_x = local_coords[0]
        self.local_y = local_coords[1]
        
        self.frame_count += 1


    def cov_as_msg(self):
        cov_flat = [self.cov[0][0], self.cov[0][1], self.cov[1][0], self.cov[1][1]]
        return cov_flat
    
    def cone_as_msg(self):
        cone_msg = Cone()
        cone_msg.location.x = self.map_x
        cone_msg.location.y = self.map_y
        cone_msg.location.z = 0.0
        cone_msg.color = self.colour
        return cone_msg

    def local_cone_as_msg(self):
        cone_msg = Cone()
        cone_msg.location.x = self.local_x
        cone_msg.location.y = self.local_y
        cone_msg.location.z = 0.0
        cone_msg.color = self.colour
        return cone_msg

class ConeAssociation(Node):
    def __init__(self):
        super().__init__('cone_placement_node')

        self.debug = False

        self.perception_subscriber = self.create_subscription(
            ConeDetectionStamped, '/vision/cone_detection', self.callback, QoSProfile(depth=1))
        
        self.pose_with_cov_subscriber = self.create_subscription(PoseWithCovarianceStamped, 'zed/zed_node/pose_with_covariance', self.pose_callback, QoSProfile(depth=10))
        #self.odometry_subscriber = self.create_subscription(Odometry, 'zed/zed_node/odom', self.odom_callback, QoSProfile(depth=10))
        
        self.global_publisher = self.create_publisher(ConeDetectionStamped, '/slam/global_map', QoSProfile(depth=1))
        self.local_publisher = self.create_publisher(ConeDetectionStamped, '/slam/local_map', QoSProfile(depth=1))


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        

        self.declare_parameter('view_x', 20.0)
        self.declare_parameter('view_y', 10.0)
        self.declare_parameter('radius', 0.8)
        self.declare_parameter('min_detections', 20)

        self.view_x = self.get_parameter('view_x').value
        self.view_y = self.get_parameter('view_y').value
        self.radius = self.get_parameter('radius').value
        self.min_detections = self.get_parameter('min_detections').value

        self.get_logger().info("                              _                                _      ")
        self.get_logger().info("                             (_)                              | |     ")
        self.get_logger().info("  _ __ ___   __ _ _ __  _ __  _ _ __   __ _    _ __   ___   __| | ___ ")
        self.get_logger().info(" | '_ ` _ \\ / _` | '_ \\| '_ \\| | '_ \\ / _` |  | '_ \\ / _ \\ / _` |/ _ \\")
        self.get_logger().info(" | | | | | | (_| | |_) | |_) | | | | | (_| |  | | | | (_) | (_| |  __/")
        self.get_logger().info(" |_| |_| |_|\\__,_| .__/| .__/|_|_| |_|\\__, |  |_| |_|\\___/ \\__,_|\\___|")
        self.get_logger().info("                 | |   | |             __/ |                          ")
        self.get_logger().info("                 |_|   |_|            |___/                           ")
        self.get_logger().info("")
        self.get_logger().info('PARAMETERS: view_x: %f, view_y: %f, radius: %f, min_detections: %d' % 
                              (self.view_x, self.view_y, self.radius, self.min_detections))

        self.mapping = False
        self.track = []
        self.pose = [0.0, 0.0, 0.0]
        
        time.sleep(5)
        
        self.mapping = True

    # # def state_callback(self, msg):
    # #     # We haven't started driving yet
    # #     if msg.state == State.DRIVING and msg.lap_count == 0:
    # #         self.mapping = True

    # #     # We have finished mapping
    # #     if msg.lap_count > 0 and self.mapping:
    # #         self.get_logger().info_once('Lap completed, mapping completed')
    # #         # self.mapping = False

    def pose_callback(self, msg):
        self.pose = msg
        #print(msg)
        
    def odom_callback(self, msg):
        self.odom = msg
        #print(msg)

    def callback(self, msg):
        if not self.mapping:
            return
        try:
            map_to_base = self.tf_buffer.lookup_transform('track', 'zed_left_camera_frame', rclpy.time.Time())
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().debug('Transform exception: %s' % str(e))
            return

        # Get the transform values
        q = [map_to_base.transform.rotation.x, map_to_base.transform.rotation.y,
             map_to_base.transform.rotation.z, map_to_base.transform.rotation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(q)

        # Update the last pose
        self.pose = [map_to_base.transform.translation.x, map_to_base.transform.translation.y, yaw]
        
        # Process detected cones
        for cone in msg.cones:
            self.find_closest_cone(cone, msg.header.frame_id)

        # If no cones were detected, return
        if not self.track:
            return
        
        if self.debug:
            print(self.track)
        
        self.create_cone_detections(msg)

    def find_closest_cone(self, cone, frame_id):
        # find the closest cone on the map to the detected cone
        # if it is within a radius, update the location of the cone
        # otherwise, add it to the list

        # create a tracked cone
        current_cone = TrackedCone(cone, frame_id, self.pose)
        closest_dist = float('inf')
        closest_index = -1

        for i, tracked_cone in enumerate(self.track):
            distance = math.sqrt((current_cone.map_x - tracked_cone.map_x) ** 2 +
                                (current_cone.map_y - tracked_cone.map_y) ** 2)
            if distance < closest_dist and distance < self.radius:
                closest_dist = distance
                closest_index = i

        # check if we found a cone within the radius
        if closest_index != -1:
            # update the location of the cone using the average of the previous location and the new location
            tracked_cone = self.track[closest_index]
            tracked_cone.update(current_cone, self.pose)
        else:
            # add the cone to the list
            self.track.append(current_cone)
        
        #print("track:", self.track)

    def create_cone_detections(self, msg):
        global_msg = ConeDetectionStamped()
        global_msg.header.stamp = msg.header.stamp
        global_msg.header.frame_id = "zed_left_camera_frame"

        local_msg = ConeDetectionStamped()
        local_msg.header.stamp = msg.header.stamp
        local_msg.header.frame_id = "zed_left_camera_frame"
        
        for cone in self.track:
            if cone.frame_count < self.min_detections:
                continue
    

            cone_msg = cone.cone_as_msg()
            global_msg.cones.append(cone_msg)

            cone_with_cov = ConeWithCovariance()
            cone_with_cov.cone = cone_msg
            cone_with_cov.covariance = cone.cov_as_msg()
            global_msg.cones_with_cov.append(cone_with_cov)

            if cone.local_x < 0 or cone.local_x > self.view_x or cone.local_y < -self.view_y or cone.local_y > self.view_y:
                continue

            local_cone_msg = cone.local_cone_as_msg()
            local_msg.cones.append(local_cone_msg)

            local_cone_with_cov = ConeWithCovariance()
            local_cone_with_cov.cone = local_cone_msg
            local_cone_with_cov.covariance = cone.cov_as_msg()
            local_msg.cones_with_cov.append(local_cone_with_cov)

        self.global_publisher.publish(global_msg)
        self.local_publisher.publish(local_msg)


def main(args=None):
    rclpy.init(args=args)


    try :
        node = ConeAssociation()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopped by Keyboard')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()