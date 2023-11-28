#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"  // Replace with your actual package name
#include <cmath>

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

class VelocityPublisher : public rclcpp::Node
{
public:
    VelocityPublisher() : Node("velocity_estimator_node")
    {
        twist_publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("camera_velocity", 10);
        pose_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/zed/zed_node/pose_with_covariance", 10, 
            std::bind(&VelocityPublisher::pose_callback, this, std::placeholders::_1));
    }

private:
void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    static constexpr double alpha = 0.8; // Filter constant
    // ...

    if (last_pose_) {
        double dt = (msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9) -
                    (last_pose_->header.stamp.sec + last_pose_->header.stamp.nanosec * 1e-9);

        if (dt > 0) {
            // Apply low-pass filter to smooth the pose data
            geometry_msgs::msg::Pose filtered_pose;
            filtered_pose.position.x = alpha * msg->pose.pose.position.x + (1 - alpha) * last_pose_->pose.pose.position.x;
            filtered_pose.position.y = alpha * msg->pose.pose.position.y + (1 - alpha) * last_pose_->pose.pose.position.y;
            filtered_pose.position.z = alpha * msg->pose.pose.position.z + (1 - alpha) * last_pose_->pose.pose.position.z;

            // Compute linear velocity using filtered pose
            double dx = filtered_pose.position.x - last_pose_->pose.pose.position.x;
            double dy = filtered_pose.position.y - last_pose_->pose.pose.position.y;
            double dz = filtered_pose.position.z - last_pose_->pose.pose.position.z;

            geometry_msgs::msg::Vector3 linear_velocity;
            linear_velocity.x = dx / dt;
            linear_velocity.y = dy / dt;
            linear_velocity.z = dz / dt;

            // Compute angular velocity (example with Euler angles)
            auto last_euler = quaternionToEuler(last_pose_->pose.pose.orientation);
            auto current_euler = quaternionToEuler(msg->pose.pose.orientation);

            geometry_msgs::msg::Vector3 angular_velocity;
            angular_velocity.x = (current_euler.roll - last_euler.roll) / dt;
            angular_velocity.y = (current_euler.pitch - last_euler.pitch) / dt;
            angular_velocity.z = (current_euler.yaw - last_euler.yaw) / dt;

            // Create and publish TwistStamped message
            geometry_msgs::msg::TwistStamped twist_msg;
            twist_msg.header.stamp = msg->header.stamp;
            twist_msg.twist.linear = linear_velocity;
            twist_msg.twist.angular = angular_velocity;
            twist_publisher_->publish(twist_msg);
            std::cout << "linear_velocity::" << std::endl;
            std::cout << "x: " << linear_velocity.x <<  std::endl;
            std::cout << "y: " << linear_velocity.y <<  std::endl;
            std::cout << "z: " << linear_velocity.z << std::endl;
            std::cout << "angular_velocity:" << std::endl;
            std::cout << "x: " << angular_velocity.x << std::endl;
            std::cout << "y: " << angular_velocity.y << std::endl;
            std::cout << "z: " << angular_velocity.z << std::endl;
        }
    }

    // Update last pose for the next iteration
    last_pose_ = msg;
}

    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_publisher_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_subscriber_;
    geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr last_pose_;
};



int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VelocityPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

//! Old publisher below (estimating from IMU, atm not good -> drift)

// #include "rclcpp/rclcpp.hpp"
// #include "geometry_msgs/msg/twist_stamped.hpp"
// #include "sensor_msgs/msg/imu.hpp"
// #include <vector>
// #include <numeric>

// class VelocityPublisher : public rclcpp::Node
// {
// public:
//     VelocityPublisher() 
//         : Node("velocity_estimator_node"), gravity_(0.0), is_initialized_(false), last_time_(this->get_clock()->now())
//     {
//         twist_publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("imu_velocity", 10);
//         imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
//             "/zed/zed_node/imu/data", 10, 
//             std::bind(&VelocityPublisher::imu_callback, this, std::placeholders::_1));
//     }

// private:
//     double gravity_;
//     bool is_initialized_;
//     rclcpp::Time last_time_;
//     geometry_msgs::msg::Vector3 linear_velocity_;
//     geometry_msgs::msg::Vector3 linear_acceleration_filtered_;
//     std::vector<double> gravity_readings_;
//     rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_publisher_;
//     rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
//     std::vector<double> gravity_initial_readings_;
//     sensor_msgs::msg::Imu::SharedPtr last_imu_msg_;


//     void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
//     {
//         if (!is_initialized_) {
//             initialize_gravity(msg);
//             last_time_ = msg->header.stamp;
//             return;
//         }

//         rclcpp::Time current_time = msg->header.stamp;
//         double dt = (current_time - last_time_).seconds();
//         last_time_ = current_time;

//         // Subtract gravity component (assuming it's along Z-axis in Earth frame)
//         auto corrected_acc_x = msg->linear_acceleration.x;
//         auto corrected_acc_y = msg->linear_acceleration.y;
//         auto corrected_acc_z = msg->linear_acceleration.z - gravity_;

//         // Apply a simple low-pass filter for noise reduction
//         double alpha = 0.99; // Filter constant, tune this value (0 < alpha < 1)
//         linear_acceleration_filtered_.x = alpha * corrected_acc_x + (1 - alpha) * linear_acceleration_filtered_.x;
//         linear_acceleration_filtered_.y = alpha * corrected_acc_y + (1 - alpha) * linear_acceleration_filtered_.y;
//         linear_acceleration_filtered_.z = alpha * corrected_acc_z + (1 - alpha) * linear_acceleration_filtered_.z;

//         // Integrate acceleration to get velocity
//         linear_velocity_.x += linear_acceleration_filtered_.x * dt;
//         linear_velocity_.y += linear_acceleration_filtered_.y * dt;
//         linear_velocity_.z += linear_acceleration_filtered_.z * dt;

//         // Publish velocity
//         geometry_msgs::msg::TwistStamped twist_msg;
//         twist_msg.header.stamp = current_time;
//         twist_msg.twist.linear = linear_velocity_;
//         twist_msg.twist.angular = msg->angular_velocity; // Directly use IMU's gyroscope data
//         twist_publisher_->publish(twist_msg);


//         RCLCPP_INFO(this->get_logger(), "Linear velocity X: %f", linear_velocity_.x);
//         RCLCPP_INFO(this->get_logger(), "Linear velocity Y: %f", linear_velocity_.y);
//         RCLCPP_INFO(this->get_logger(), "Linear velocity Z: %f", linear_velocity_.z);
//         RCLCPP_INFO(this->get_logger(), "Angular acceleration X: %f", msg->angular_velocity.x);
//         RCLCPP_INFO(this->get_logger(), "Angular acceleration Y: %f", msg->angular_velocity.y);
//         RCLCPP_INFO(this->get_logger(), "Angular acceleration Z: %f", msg->angular_velocity.z);

//     }


//     void initialize_gravity(const sensor_msgs::msg::Imu::SharedPtr& msg) {
//     gravity_readings_.push_back(msg->linear_acceleration.z);

//     if (gravity_readings_.size() >= 100) {
//         gravity_ = std::accumulate(gravity_readings_.begin(), gravity_readings_.end(), 0.0) / gravity_readings_.size();
//         is_initialized_ = true;
//         gravity_readings_.clear(); // Clear the vector as it's no longer needed
//         RCLCPP_INFO(this->get_logger(), "Gravity initialized: %f", gravity_);
//     }
// }



// };

// int main(int argc, char **argv)
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<VelocityPublisher>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }




