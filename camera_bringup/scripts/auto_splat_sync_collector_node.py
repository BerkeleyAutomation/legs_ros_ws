#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
import os
import cv2
from cv_bridge import CvBridge
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.output_folder = 'sync_images'  # Main folder to save images

        # Create the main folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Create subscribers for each topic
        self.camera_color_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.camera_depth_sub = Subscriber(self, Image, '/camera/depth/image_rect_raw')
        self.left_zed_depth_sub = Subscriber(self, Image, '/tri_left_zed_depth')
        self.left_zed_left_sub = Subscriber(self, Image, '/tri_left_zed_cropped')
        self.right_zed_depth_sub = Subscriber(self, Image, '/tri_right_zed_depth')
        self.right_zed_left_sub = Subscriber(self, Image, '/tri_right_zed_cropped')

        # Create the ApproximateTimeSynchronizer
        self.approx_sync = ApproximateTimeSynchronizer(
            [self.camera_color_sub, self.camera_depth_sub, self.left_zed_depth_sub,
             self.left_zed_left_sub, self.right_zed_depth_sub,
             self.right_zed_left_sub],
            queue_size=10,
            slop=0.125)  # Slop is the maximum time difference allowed between messages

        # Register the callback function
        self.approx_sync.registerCallback(self.callback)
        self.counter_ = 0
        self.bridge = CvBridge()

    def callback(self, camera_color_msg, camera_depth_msg, left_zed_depth_msg,
                 left_zed_left_msg, right_zed_depth_msg,
                 right_zed_left_msg):
        # Create folders for each type of image
        camera_color_folder = os.path.join(self.output_folder, 'camera_color')
        camera_depth_folder = os.path.join(self.output_folder, 'camera_depth')
        left_zed_depth_folder = os.path.join(self.output_folder, 'left_tri_depth')
        left_zed_left_folder = os.path.join(self.output_folder, 'left_tri_rgb')
        right_zed_depth_folder = os.path.join(self.output_folder, 'right_tri_depth')
        right_zed_left_folder = os.path.join(self.output_folder, 'right_tri_rgb')

        for folder in [camera_color_folder, camera_depth_folder, left_zed_depth_folder,
                       left_zed_left_folder, right_zed_depth_folder,
                       right_zed_left_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Save images
        self.save_image(camera_color_msg, camera_color_folder, 'camera_color')
        self.save_image(camera_depth_msg, camera_depth_folder, 'camera_depth',is_depth=True)
        self.save_image(left_zed_depth_msg, left_zed_depth_folder, 'left_tri_depth',is_depth=True)
        self.save_image(left_zed_left_msg, left_zed_left_folder, 'left_tri_rgb')
        self.save_image(right_zed_depth_msg, right_zed_depth_folder, 'right_tri_depth',is_depth=True)
        self.save_image(right_zed_left_msg, right_zed_left_folder, 'right_tri_rgb')
        print(self.counter_)
        self.counter_ += 1

    def save_image(self, msg, folder_path, image_name,is_depth=False):
        # Convert ROS image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # Save image with zerofilled counter
        filename = f'{image_name}_{str(msg.header.stamp.sec).zfill(10)}_{str(msg.header.stamp.nanosec).zfill(9)}.png'
        if is_depth:
            np.save(os.path.join(folder_path,filename),cv_image)
        else:
            cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(folder_path, filename), cv_image)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
