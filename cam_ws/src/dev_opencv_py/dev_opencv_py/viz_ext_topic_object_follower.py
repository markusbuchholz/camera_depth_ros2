# Markus Buchholz
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ColorDetectorNode(Node):
    def __init__(self):
        super().__init__('color_detector_node')

        # Publisher for target position
        self.publisher = self.create_publisher(Float64MultiArray, '/target_position', 10)

        # Subscriber for image data
        self.subscription = self.create_subscription(
            Image,
            '/video_frames',  # Replace with your image topic name
            self.image_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

        self.br = CvBridge()

        # Color ranges in HSV
        self.color_ranges = {
            'blue': {
                'low': np.array([110, 50, 50]),
                'high': np.array([130, 255, 255])
            },
            'yellow': {
                'low': np.array([20, 100, 100]),
                'high': np.array([30, 255, 255])
            },
            'red1': {
                'low': np.array([170, 50, 50]),
                'high': np.array([180, 255, 255])
            },
            'red2': {
                'low': np.array([0, 50, 50]),
                'high': np.array([10, 255, 255])
            },
        }

        # Select the color you want to detect
        self.selected_color = 'red2'  # change as needed

        self.image_width = 600  
        self.image_height = 480

        # Define the square corner pixel coordinates and their corresponding XYZ coordinates
        # Format: (x, y) : (X, Y, Z)
        self.corner_points = {
            'top_left':     {'pixel': (410, 150), 'xyz': (0.2, -0.15, 0.2)},
            'top_right':    {'pixel': (115, 150), 'xyz': (0.2,  0.15, 0.2)},
            'bottom_left':  {'pixel': (410, 375), 'xyz': (0.2, -0.15, -0.2)},
            'bottom_right': {'pixel': (115, 375), 'xyz': (0.2, 0.15, -0.2)}
        }

        # Extract corners for easier reference
        self.x1, self.y1 = self.corner_points['top_right']['pixel']   # (115, 150)
        self.x2, self.y2 = self.corner_points['bottom_left']['pixel'] # (410, 375)

        self.X1, self.Y1, self.Z1 = self.corner_points['top_right']['xyz']    # (0.2,  0.15, 0.2)
        self.X2, self.Y2, self.Z2 = self.corner_points['top_left']['xyz']     # (0.2, -0.15, 0.2)
        self.X3, self.Y3, self.Z3 = self.corner_points['bottom_right']['xyz'] # (0.13, 0.15, -0.16)
        self.X4, self.Y4, self.Z4 = self.corner_points['bottom_left']['xyz']  # (0.2, -0.15, -0.05)

        # Minimum radius to be considered a valid detection
        self.MIN_RADIUS = 2

        # Create a named OpenCV window for debugging/visualization
        cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Image", 800, 600)

        self.get_logger().info("Color Detector Node has been started and is subscribed to image topic.")

    def image_callback(self, data):
        """
        Callback function to handle incoming Image messages.
        """
        try:
            current_frame = self.br.imgmsg_to_cv2(data, desired_encoding="bgr8")
            self.get_logger().debug('Received a new image frame.')
            self.process_and_publish(current_frame)
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def process_and_publish(self, img):
        """
        Processes the image to detect the selected color and publishes the target position.
        Also handles visualization.
        """
        # Resize for consistent processing
        img_resized = cv2.resize(img, (self.image_width, self.image_height))

        # Optional smoothing
        img_filter = cv2.GaussianBlur(img_resized.copy(), (3, 3), 0)
        img_hsv = cv2.cvtColor(img_filter, cv2.COLOR_BGR2HSV)

        # Get the correct color thresholds
        if self.selected_color in self.color_ranges:
            THRESHOLD_LOW = self.color_ranges[self.selected_color]['low']
            THRESHOLD_HIGH = self.color_ranges[self.selected_color]['high']
        else:
            self.get_logger().warning(f"Selected color '{self.selected_color}' not defined.")
            # Always show the frame, even if no valid color is selected
            cv2.imshow("Processed Image", img_resized)
            cv2.waitKey(1)
            return

        # Handle the special case for red range
        if self.selected_color.startswith('red'):
            mask1 = cv2.inRange(img_hsv, self.color_ranges['red1']['low'], self.color_ranges['red1']['high'])
            mask2 = cv2.inRange(img_hsv, self.color_ranges['red2']['low'], self.color_ranges['red2']['high'])
            img_binary = cv2.bitwise_or(mask1, mask2)
        else:
            img_binary = cv2.inRange(img_hsv, THRESHOLD_LOW, THRESHOLD_HIGH)

        # Dilate to make blobs larger
        img_binary = cv2.dilate(img_binary, None, iterations=1)

        # Find contours on the binary image
        contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If there are no contours, show the video, log, and return
        if not contours:
            self.get_logger().info("No object detected.")
            cv2.imshow("Processed Image", img_resized)
            cv2.waitKey(1)
            return

        # Grab the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Threshold checks to avoid noise
        AREA_THRESHOLD = 500.0  # Tweak this value as needed
        if contour_area < AREA_THRESHOLD or radius < self.MIN_RADIUS:
            self.get_logger().info("No object detected (contour too small).")
            cv2.imshow("Processed Image", img_resized)
            cv2.waitKey(1)
            return

        # If we reach here, consider it a valid detection
        M = cv2.moments(largest_contour)
        mapped_xyz = [0.0, 0.0, 0.0]  # default
        center = None

        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Draw the detection on the image
            cv2.circle(img_resized, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(img_resized, center, 5, (0, 0, 255), -1)
            cv2.putText(img_resized, f"Center: {center}", (center[0] + 10, center[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Map center pixel coordinates to 3D coordinates
            mapped_xyz = self.map_to_square(center)

        # Publish the coordinates
        msg = Float64MultiArray()
        msg.data = mapped_xyz
        self.publisher.publish(msg)

        # Log the published data
        self.get_logger().info(
            f"Published to /target_position: [X: {mapped_xyz[0]:.5f}, "
            f"Y: {mapped_xyz[1]:.5f}, Z: {mapped_xyz[2]:.5f}]"
        )

        # Show the processed image every time
        cv2.imshow("Processed Image", img_resized)
        cv2.waitKey(1)

    def map_to_square(self, center):
        """
        Maps the pixel coordinates to the predefined square's XYZ coordinates using bilinear interpolation.
        """
        x_pixel, y_pixel = center

        # Extract corner pixel coordinates
        x1, y1 = self.x1, self.y1  # top right
        x2, y2 = self.x2, self.y2  # bottom left

        # Calculate normalized coordinates [0,1]
        t = (x_pixel - x1) / (x2 - x1) if (x2 - x1) != 0 else 0
        u = (y_pixel - y1) / (y2 - y1) if (y2 - y1) != 0 else 0

        # Clamp t and u to [0,1] to handle points outside the defined rectangle
        t = max(0.0, min(t, 1.0))
        u = max(0.0, min(u, 1.0))

        # Bilinear interpolation for X
        X = (
            (1 - t) * (1 - u) * self.X1
            + t       * (1 - u) * self.X2
            + (1 - t) *       u * self.X3
            + t       *       u * self.X4
        )

        # For Y, a simple linear interpolation example:
        # Y decreases from 0.15 to -0.15 as x goes from 115 to 410
        Y = 0.15 - 0.3 * t

        # Bilinear interpolation for Z
        Z = (
            (1 - t) * (1 - u) * self.Z1
            + t       * (1 - u) * self.Z2
            + (1 - t) *       u * self.Z3
            + t       *       u * self.Z4
        )

        self.get_logger().debug(
            f"Pixel ({x_pixel}, {y_pixel}) -> XYZ ({X:.5f}, {Y:.5f}, {Z:.5f})"
        )

        return [X, Y, Z]

    def destroy_node(self):
        # Close OpenCV windows if any
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
