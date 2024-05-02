// Copyright 2021, Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"

#include <chrono>
#include <functional>
#include <memory>
using namespace std::chrono_literals;
image_transport::Publisher realsense_depth_pub;
image_transport::Publisher realsense_color_pub;

image_transport::Publisher left_zed_depth_pub;
image_transport::Publisher left_zed_left_color_pub;
image_transport::Publisher left_zed_right_color_pub;

image_transport::Publisher right_zed_depth_pub;
image_transport::Publisher right_zed_left_color_pub;
image_transport::Publisher right_zed_right_color_pub;

double frequency_diff_ = 1;
double desired_fps_ = 5;
std::chrono::duration<double> delay_ = std::chrono::duration<double>(1.0 / (desired_fps_ + frequency_diff_));
auto realsense_depth_start_time_ = std::chrono::steady_clock::now();
auto realsense_color_start_time_ = std::chrono::steady_clock::now();
auto left_zed_left_start_time_ = std::chrono::steady_clock::now();
auto left_zed_right_start_time_ = std::chrono::steady_clock::now();
auto left_zed_depth_start_time_ = std::chrono::steady_clock::now();
auto right_zed_left_start_time_ = std::chrono::steady_clock::now();
auto right_zed_right_start_time_ = std::chrono::steady_clock::now();
auto right_zed_depth_start_time_ = std::chrono::steady_clock::now();

void realsenseColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto method_start_time = std::chrono::steady_clock::now();
  auto color_current_time = std::chrono::steady_clock::now();
  if(color_current_time - realsense_color_start_time_ >= std::chrono::duration<double>(delay_)) {
    realsense_color_pub.publish(*msg);
    realsense_color_start_time_ = std::chrono::steady_clock::now();
  }
}

void realsenseDepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto method_start_time = std::chrono::steady_clock::now();
  auto depth_current_time = std::chrono::steady_clock::now();
  if(depth_current_time - realsense_depth_start_time_ >= std::chrono::duration<double>(delay_)) {
    realsense_depth_pub.publish(*msg);
    realsense_depth_start_time_ = std::chrono::steady_clock::now();
  }
}

void leftZedDepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto method_start_time = std::chrono::steady_clock::now();
  auto left_zed_depth_current_time = std::chrono::steady_clock::now();
  if(left_zed_depth_current_time - left_zed_depth_start_time_ >= std::chrono::duration<double>(delay_)) {
    left_zed_depth_pub.publish(*msg);
    left_zed_depth_start_time_ = std::chrono::steady_clock::now();
  }
}

void leftZedLeftColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto method_start_time = std::chrono::steady_clock::now();
  auto left_zed_left_current_time = std::chrono::steady_clock::now();
  if(left_zed_left_current_time - left_zed_left_start_time_ >= std::chrono::duration<double>(delay_)) {
    left_zed_left_color_pub.publish(*msg);
    left_zed_left_start_time_ = std::chrono::steady_clock::now();
  }
}

void leftZedRightColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto method_start_time = std::chrono::steady_clock::now();
  auto left_zed_right_current_time = std::chrono::steady_clock::now();
  if(left_zed_right_current_time - left_zed_right_start_time_ >= std::chrono::duration<double>(delay_)) {
    left_zed_right_color_pub.publish(*msg);
    left_zed_right_start_time_ = std::chrono::steady_clock::now();
  }
}

void rightZedDepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto method_start_time = std::chrono::steady_clock::now();
  auto right_zed_depth_current_time = std::chrono::steady_clock::now();
  if(right_zed_depth_current_time - right_zed_depth_start_time_ >= std::chrono::duration<double>(delay_)) {
    right_zed_depth_pub.publish(*msg);
    right_zed_depth_start_time_ = std::chrono::steady_clock::now();
  }
}

void rightZedLeftColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto method_start_time = std::chrono::steady_clock::now();
  auto right_zed_left_current_time = std::chrono::steady_clock::now();
  if(right_zed_left_current_time - right_zed_left_start_time_ >= std::chrono::duration<double>(delay_)) {
    right_zed_left_color_pub.publish(*msg);
    right_zed_left_start_time_ = std::chrono::steady_clock::now();
  }
}

void rightZedRightColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  auto method_start_time = std::chrono::steady_clock::now();
  auto right_zed_right_current_time = std::chrono::steady_clock::now();
  if(right_zed_right_current_time - right_zed_right_start_time_ >= std::chrono::duration<double>(delay_)) {
    right_zed_right_color_pub.publish(*msg);
    right_zed_right_start_time_ = std::chrono::steady_clock::now();
  }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("image_listener", options);
    image_transport::ImageTransport it(node);
    realsense_depth_pub = it.advertise("/repub_realsense_depth", 10);
    realsense_color_pub = it.advertise("/repub_realsense_color", 10);

    left_zed_depth_pub = it.advertise("/repub_left_zed_depth", 10);
    left_zed_left_color_pub = it.advertise("/repub_left_zed_left_color", 10);
    left_zed_right_color_pub = it.advertise("/repub_left_zed_right_color", 10);

    right_zed_depth_pub = it.advertise("/repub_right_zed_depth", 10);
    right_zed_left_color_pub = it.advertise("/repub_right_zed_left_color", 10);
    right_zed_right_color_pub = it.advertise("/repub_right_zed_right_color", 10);
    image_transport::Subscriber realsense_color_sub = it.subscribe("/camera/color/image_raw", 10, realsenseColorCallback);
    image_transport::Subscriber realsense_depth_sub = it.subscribe("/camera/depth/image_rect_raw", 10, realsenseDepthCallback);

    image_transport::Subscriber left_zed_depth_sub = it.subscribe("/tri_left_zed_depth", 10, leftZedDepthCallback);
    image_transport::Subscriber left_zed_left_color_sub = it.subscribe("/tri_left_zed_cropped", 10, leftZedLeftColorCallback);
    
    image_transport::Subscriber right_zed_depth_sub = it.subscribe("/tri_right_zed_depth", 10, rightZedDepthCallback);
    image_transport::Subscriber right_zed_left_color_sub = it.subscribe("/tri_right_zed_cropped", 10, rightZedLeftColorCallback);
    rclcpp::spin(node);
}

/*

    image_transport::Subscriber depth_sub = it.subscribe("/ros2_camera/depth/image_rect_raw", 10, realsenseDepthCallback);
    image_transport::Subscriber color_sub = it.subscribe("/ros2_camera/color/image_raw", 10, realsenseColorCallback);
    image_transport::Subscriber left_sub = it.subscribe("/camLeft/image_raw", 10, leftCallback);
    image_transport::Subscriber right_sub = it.subscribe("/camRight/image_raw", 10, rightCallback);
    rclcpp::spin(node);
}
*/