#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

class VelocityExtractor : public rclcpp::Node
{
public:
    VelocityExtractor() : Node("velocity_extractor_node")
    {
        // Subscriber to the odometry topic
        odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/zed/zed_node/odom", 10,
            std::bind(&VelocityExtractor::odomCallback, this, std::placeholders::_1));

        // Publisher for the Twist message
        twist_publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("extracted_velocity", 10);
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // Extract the Twist message
        auto twist_msg = msg->twist.twist;

        // Log the linear and angular velocity
        RCLCPP_INFO(this->get_logger(), "Linear Velocity: [%f, %f, %f]", 
                    twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z);
        RCLCPP_INFO(this->get_logger(), "Angular Velocity: [%f, %f, %f]", 
                    twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z);

        // Publish the Twist message
        geometry_msgs::msg::TwistStamped twist_stamped;
        twist_stamped.header.stamp = this->get_clock()->now();
        twist_stamped.twist = twist_msg;
        twist_publisher_->publish(twist_stamped);
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_publisher_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VelocityExtractor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
