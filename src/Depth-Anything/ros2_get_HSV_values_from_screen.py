#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Subscribe to the 'video_frames' topic
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize variables for mouse callback
        self.drawing = False  # True if the mouse is pressed
        self.ix, self.iy = -1, -1  # Initial mouse coordinates
        self.fx, self.fy = -1, -1  # Final mouse coordinates
        self.selected = False  # True if a region has been selected
        self.hsv_lower = None
        self.hsv_upper = None
        self.frame_original = None

        # Create a named window and set the mouse callback
        cv2.namedWindow('Camera Feed')
        cv2.setMouseCallback('Camera Feed', self.mouse_callback)

        self.get_logger().info('Image Subscriber Node has been started.')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_original = frame.copy()

            if self.drawing:
                # Draw the current rectangle
                cv2.rectangle(frame, (self.ix, self.iy), (self.fx, self.fy), (0, 255, 0), 2)

            cv2.imshow('Camera Feed', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.get_logger().info('Quitting the node.')
                rclpy.shutdown()
            elif key == ord('c'):
                self.selected = False
                self.hsv_lower = None
                self.hsv_upper = None
                self.get_logger().info('Selection cleared.')

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.fx, self.fy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.fx, self.fy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.fx, self.fy = x, y
            self.selected = True

            if self.frame_original is None:
                self.get_logger().warning('No frame available for selection.')
                self.selected = False
                return

            # Ensure coordinates are within the frame
            x0, y0 = min(self.ix, self.fx), min(self.iy, self.fy)
            x1, y1 = max(self.ix, self.fx), max(self.iy, self.fy)

            # Extract the selected region
            selected_region = self.frame_original[y0:y1, x0:x1]

            if selected_region.size == 0:
                self.get_logger().warning("Selected region is empty. Please select a valid area.")
                self.selected = False
                return

            # Convert to HSV
            hsv_selected = cv2.cvtColor(selected_region, cv2.COLOR_BGR2HSV)

            # Compute the lower and upper bounds using percentiles to avoid outliers
            h_lower, s_lower, v_lower = np.percentile(hsv_selected, 5, axis=(0,1))
            h_upper, s_upper, v_upper = np.percentile(hsv_selected, 95, axis=(0,1))

            # Define tolerance (adjust as needed)
            h_tol = 10
            s_tol = 50
            v_tol = 50

            # Compute lower and upper bounds with clipping
            lower = np.array([
                max(h_lower - h_tol, 0),
                max(s_lower - s_tol, 0),
                max(v_lower - v_tol, 0)
            ], dtype=np.uint8)

            upper = np.array([
                min(h_upper + h_tol, 179),
                min(s_upper + s_tol, 255),
                min(v_upper + v_tol, 255)
            ], dtype=np.uint8)

            self.hsv_lower = lower
            self.hsv_upper = upper

            self.get_logger().info("Selected Region HSV Bounds:")
            self.get_logger().info(f"Lower HSV: {self.hsv_lower}")
            self.get_logger().info(f"Upper HSV: {self.hsv_upper}")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()
        self.get_logger().info('Image Subscriber Node has been shut down.')

def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        image_subscriber.get_logger().info('Image Subscriber Node interrupted by user.')
    finally:
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
