#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <queue>
#include <vector>
#include <cmath>


#include "../markers.h"
#include "ackermann_msgs/msg/ackermann_drive.hpp"
#include "builtin_interfaces/msg/time.hpp"
#include "custom_sys_msgs/msg/cone.hpp"
#include "custom_sys_msgs/msg/cone_detection_stamped.hpp"
//#include "custom_sys_msgs/msg/debug_msg.hpp"
//#include "custom_sys_msgs/msg/double_matrix.hpp"
#include "custom_sys_msgs/msg/camera_velocity.hpp"
#include "ekf_slam.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "nav_msgs/msg/odometry.hpp"


#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/header.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

// Common functionality: BEGIN
const rclcpp::QoS QOS_LATEST(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data);
const rclcpp::QoS QOS_ALL(rclcpp::KeepLast(100), rmw_qos_profile_parameters);

inline double dist(double x1, double y1, double x2, double y2) {
    double diff_x = x1 - x2;
    double diff_y = y1 - y2;
    return sqrt(diff_x * diff_x + diff_y * diff_y);
}

inline double fast_dist(double x1, double y1, double x2, double y2) {
    double diff_x = x1 - x2;
    double diff_y = y1 - y2;
    return diff_x * diff_x + diff_y * diff_y;
}

inline double angle(double x1, double y1, double x2, double y2) {
    double x_disp = x2 - x1;
    double y_disp = y2 - y1;
    return atan2(y_disp, x_disp);
}

// https://stackoverflow.com/a/29871193
inline double wrap_to_pi(double x) {
    double min = x - (-M_PI);
    double max = M_PI - (-M_PI);
    return -M_PI + fmod(max + fmod(min, max), max);
}

struct EulerAngle {
    double roll;
    double pitch;
    double yaw;
};

EulerAngle quaternionToEuler(const geometry_msgs::msg::Quaternion& q) {
    EulerAngle euler;

    // roll (x-axis rotation)
    double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
    double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    euler.roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    if (std::abs(sinp) >= 1)
        euler.pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        euler.pitch = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    euler.yaw = std::atan2(siny_cosp, cosy_cosp);

    return euler;
}
// Common functionality: END

using std::placeholders::_1;

const std::string PARAM_RANGE_VARIANCE = "range_variance";
const std::string PARAM_BEARING_VARIANCE = "bearing_variance";
const std::string PARAM_UNCERTANTY_TIME_WEIGHT = "uncertainty_time_weight";
const std::string PARAM_UNCERTANTY_ROTATION_WEIGHT = "uncertainty_rotation_weight";
const std::string PARAM_UNCERTANTY_FORWARD_WEIGHT = "uncertainty_forward_weight";
const std::string PARAM_UNCERTANTY_HEADING_TIME_WEIGHT = "uncertainty_heading_time_weight";
const std::string PARAM_ASSOCIATION_DIST_THRESHOLD = "association_dist_threshold";
const std::string PARAM_USE_TOTAL_ABS_VEL = "use_total_abs_vel";
const std::string PARAM_USE_KNOWN_ASSOCIATION = "use_known_association";
const std::string PARAM_USE_ODOM_ONLY = "use_odom_only";
const std::string PARAM_REVERSE_ROTATION = "reverse_rotation";
const std::string PARAM_POSE_SOURCE = "pose_source";

double compute_dt(rclcpp::Time start_, rclcpp::Time end_) { return (end_ - start_).nanoseconds() * 1e-9; }

class EKFSLAMNode : public rclcpp::Node {
   private:
    EKFslam ekf_slam;

    double forward_vel;
    double rotational_vel;
    std::optional<rclcpp::Time> last_update;

    double range_variance;
    double bearing_variance;

    double uncertainty_time_weight;
    double uncertainty_rotation_weight;
    double uncertainty_forward_weight;
    double uncertainty_heading_time_weight;

    double association_dist_threshold;
    bool use_total_abs_vel;
    bool use_known_association;
    bool use_odom_only;
    bool reverse_rotation;

    std::string pose_source_;

    std::queue<geometry_msgs::msg::TwistStamped::SharedPtr> twist_queue;

    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_sub;
    rclcpp::Subscription<custom_sys_msgs::msg::ConeDetectionStamped>::SharedPtr detection_sub;

    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;

    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub;
    rclcpp::Publisher<custom_sys_msgs::msg::ConeDetectionStamped>::SharedPtr track_pub;

   public:
    EKFSLAMNode() : Node("ekf_node") {
        twist_sub = create_subscription<geometry_msgs::msg::TwistStamped>(
            "/camera_velocity", QOS_LATEST, std::bind(&EKFSLAMNode::twist_callback, this, _1));

        detection_sub = create_subscription<custom_sys_msgs::msg::ConeDetectionStamped>(
            "/vision/cone_detection", QOS_ALL, std::bind(&EKFSLAMNode::cone_detection_callback, this, _1));

        pose_pub = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("slam/pose", 10);
        track_pub = create_publisher<custom_sys_msgs::msg::ConeDetectionStamped>("slam/track", 10);

        pose_sub = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/zed/zed_node/pose", QOS_ALL, std::bind(&EKFSLAMNode::pose_callback, this, _1));

        if (pose_source_ == "pose") {
            std::cout << "ok" << std::endl;
        } else if (pose_source_ == "odom") {
            odom_sub = create_subscription<nav_msgs::msg::Odometry>(
                "/zed/zed_node/odom", QOS_ALL, std::bind(&EKFSLAMNode::odom_callback, this, _1));
        } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid pose source selected");
        }

        declare_parameter<double>(PARAM_RANGE_VARIANCE, 0.15);
        declare_parameter<double>(PARAM_BEARING_VARIANCE, 0.15);
        declare_parameter<double>(PARAM_UNCERTANTY_TIME_WEIGHT, 0.05);
        declare_parameter<double>(PARAM_UNCERTANTY_ROTATION_WEIGHT, 0.01);
        declare_parameter<double>(PARAM_UNCERTANTY_FORWARD_WEIGHT, 0.01);
        declare_parameter<double>(PARAM_UNCERTANTY_HEADING_TIME_WEIGHT, 0.05);
        declare_parameter<double>(PARAM_ASSOCIATION_DIST_THRESHOLD, 0.9);
        declare_parameter<bool>(PARAM_USE_TOTAL_ABS_VEL, false);
        declare_parameter<bool>(PARAM_USE_KNOWN_ASSOCIATION, false);
        declare_parameter<bool>(PARAM_USE_ODOM_ONLY, false);
        declare_parameter<bool>(PARAM_REVERSE_ROTATION, false);
        declare_parameter<std::string>(PARAM_POSE_SOURCE, "pose");
        

        range_variance = get_parameter(PARAM_RANGE_VARIANCE).as_double();
        bearing_variance = get_parameter(PARAM_BEARING_VARIANCE).as_double();
        uncertainty_time_weight = get_parameter(PARAM_UNCERTANTY_TIME_WEIGHT).as_double();
        uncertainty_rotation_weight = get_parameter(PARAM_UNCERTANTY_ROTATION_WEIGHT).as_double();
        uncertainty_forward_weight = get_parameter(PARAM_UNCERTANTY_FORWARD_WEIGHT).as_double();
        uncertainty_heading_time_weight = get_parameter(PARAM_UNCERTANTY_HEADING_TIME_WEIGHT).as_double();
        association_dist_threshold = get_parameter(PARAM_ASSOCIATION_DIST_THRESHOLD).as_double();
        use_known_association = get_parameter(PARAM_USE_KNOWN_ASSOCIATION).as_bool();
        use_odom_only = get_parameter(PARAM_USE_ODOM_ONLY).as_bool();
        reverse_rotation = get_parameter(PARAM_REVERSE_ROTATION).as_bool();
        pose_source_ = get_parameter(PARAM_POSE_SOURCE).as_string();

        std::cout << "Node started" << std::endl;
    }

    geometry_msgs::msg::PoseWithCovarianceStamped waitForPoseFromZED() {
        auto promise = std::make_shared<std::promise<geometry_msgs::msg::PoseWithCovarianceStamped>>();
        auto future = promise->get_future();

        auto subscription = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/zed/zed_node/pose", QOS_LATEST,
            [this, promise](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                promise->set_value(*msg);
                // Unsubscribe after receiving the first message
                this->pose_sub = nullptr;
            });

        // Save temporary subscription in the class member to prevent it from going out of scope
        pose_sub = subscription;

        // Wait for the future to be set by the callback, which indicates the message has been received
        future.wait();
        // Return the received pose
        return future.get();
    }

    void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        // Extract pose data and update EKF state
        auto& pose = msg->pose.pose;
        EulerAngle euler = quaternionToEuler(pose.orientation);
        ekf_slam.update_pose(pose.position.x, pose.position.y, euler.yaw);
        // Additional logic...
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Extract pose and velocity data, and update EKF state
        auto& pose = msg->pose.pose;
        EulerAngle euler = quaternionToEuler(pose.orientation);
        ekf_slam.update_pose_and_velocity(pose.position.x, pose.position.y, euler.yaw, msg->twist.twist.linear, msg->twist.twist.angular);
        // Additional logic...
    }

    void twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
        std::cout << "Received twist message" << std::endl;

        if (twist_queue.empty()) {
            twist_queue.push(msg);
            std::cout << "Added twist message to queue" << std::endl;
        } else {
            auto& last_msg = twist_queue.back();
            if (msg->header.stamp.sec > last_msg->header.stamp.sec ||
                (msg->header.stamp.sec == last_msg->header.stamp.sec && msg->header.stamp.nanosec > last_msg->header.stamp.nanosec)) {
                twist_queue.push(msg);
                std::cout << "Added twist message to queue" << std::endl;
            } else {
                std::cout << "Discarded twist message with duplicate or older timestamp" << std::endl;
            }
        }
    }

    void predict(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
        rclcpp::Time stamp = msg->header.stamp;
        //return;

        if (use_total_abs_vel) {
            forward_vel = abs(msg->twist.linear.x) + abs(msg->twist.linear.y) + abs(msg->twist.linear.z);
        } else {
            forward_vel = msg->twist.linear.x;
        }

        if (reverse_rotation) {
            rotational_vel = -msg->twist.angular.z;
        } else {
            rotational_vel = msg->twist.angular.z;
        }

        if (!last_update.has_value()) {
            last_update = stamp;
            return;
        }

        double dt = compute_dt(last_update.value(), stamp);

        std::cout << "Predict:" << std::endl;
        std::cout << "Last update time: " << last_update.value().seconds() << "." << last_update.value().nanoseconds() << std::endl;
        std::cout << "Message stamp time: " << stamp.seconds() << "." << stamp.nanoseconds() << std::endl;
        std::cout << "Time difference (dt): " << dt << " seconds" << std::endl;

        if (!(dt > 0)) {
            return;
        }

        ekf_slam.predict(forward_vel, rotational_vel, dt, uncertainty_time_weight, uncertainty_rotation_weight,
                         uncertainty_forward_weight, uncertainty_heading_time_weight);

        last_update = stamp;
    }

    void cone_detection_callback(const custom_sys_msgs::msg::ConeDetectionStamped::SharedPtr detection_msg) {
        rclcpp::Time stamp = detection_msg->header.stamp;

        std::cout << "Entered cone_detection_callback" << std::endl;

        if (!last_update.has_value()) {
            last_update = stamp;
            std::cout << "if (!last_update.has_value()) {" << std::endl;

            return;
        }

        while (!twist_queue.empty() && compute_dt(twist_queue.front()->header.stamp, stamp) >= 0) {
            predict(twist_queue.front());
            twist_queue.pop();
        }

        double dt = compute_dt(last_update.value(), stamp);

        std::cout << "Cone det callback:" << std::endl;
        std::cout << "Last update time: " << last_update.value().seconds() << "." << last_update.value().nanoseconds() << std::endl;
        std::cout << "Message stamp time: " << stamp.seconds() << "." << stamp.nanoseconds() << std::endl;
        std::cout << "Time difference (dt): " << dt << " seconds" << std::endl;


        if (!(dt > 0)) {
            return;
        }

        std::transform(detection_msg->cones_with_cov.cbegin(), detection_msg->cones_with_cov.cend(),
                       std::back_inserter(detection_msg->cones),
                       [](const custom_sys_msgs::msg::ConeWithCovariance& c) { return c.cone; });

        ekf_slam.predict(forward_vel, rotational_vel, dt, uncertainty_time_weight, uncertainty_rotation_weight,
                         uncertainty_forward_weight, uncertainty_heading_time_weight);

        if (!use_odom_only) {
            ekf_slam.update(detection_msg->cones, range_variance, bearing_variance, association_dist_threshold,
                            use_known_association, this->get_logger());
        }

        last_update = stamp;
        publish_state(detection_msg->header.stamp);
        std::cout << "publish state" << std::endl;
    }

    void publish_state(builtin_interfaces::msg::Time stamp) {
        double x, y, theta;
        ekf_slam.get_state(x, y, theta);
        geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
        pose_msg.header.frame_id = "track";
        pose_msg.header.stamp = stamp;
        pose_msg.pose.pose.position.x = x;
        pose_msg.pose.pose.position.y = y;
        pose_msg.pose.pose.position.z = 0;
        Eigen::Quaterniond q(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));
        auto coeffs = q.coeffs();
        pose_msg.pose.pose.orientation.x = coeffs(0);
        pose_msg.pose.pose.orientation.y = coeffs(1);
        pose_msg.pose.pose.orientation.z = coeffs(2);
        pose_msg.pose.pose.orientation.w = coeffs(3);
        pose_msg.pose.covariance[0 + 0 * 6] = ekf_slam.get_cov()(0, 0);
        pose_msg.pose.covariance[1 + 1 * 6] = ekf_slam.get_cov()(1, 1);
        pose_msg.pose.covariance[5 + 5 * 6] = ekf_slam.get_cov()(2, 2);
        pose_pub->publish(pose_msg);

        custom_sys_msgs::msg::ConeDetectionStamped cones_msg;
        cones_msg.header.frame_id = "track";
        cones_msg.header.stamp = stamp;
        cones_msg.cones_with_cov = ekf_slam.get_cones();
        track_pub->publish(cones_msg);
        std::cout << "publish" << std::endl;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    std::cout << "1" << std::endl;
    auto ekf_node = std::make_shared<EKFSLAMNode>();
    std::cout << "2" << std::endl;
    rclcpp::spin(ekf_node);
    rclcpp::shutdown();

    return 0;
}
