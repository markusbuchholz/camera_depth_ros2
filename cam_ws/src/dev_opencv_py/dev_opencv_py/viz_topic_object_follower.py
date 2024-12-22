#Markus Buchholz
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

        self.selected_color = 'red2'  # change as needed

        self.image_width = 600  
        self.image_height = 480

        # define the square corner pixel coordinates and their corresponding XYZ coordinates
        # Format: (x, y) : (X, Y, Z)
        self.corner_points = {
            'top_left':     {'pixel': (410, 150), 'xyz': (0.2, -0.15, 0.2)},
            'top_right':    {'pixel': (115, 150), 'xyz': (0.2, 0.15, 0.2)},
            'bottom_left':  {'pixel': (410, 375), 'xyz': (0.2, -0.15, -0.05)},
            'bottom_right': {'pixel': (115, 375), 'xyz': (0.13, 0.15, -0.16)}
        }

        self.x1, self.y1 = self.corner_points['top_right']['pixel']
        self.x2, self.y2 = self.corner_points['bottom_left']['pixel']  # Assuming rectangle, x2=410, y2=375

        self.X1, self.Y1, self.Z1 = self.corner_points['top_right']['xyz']
        self.X2, self.Y2, self.Z2 = self.corner_points['top_left']['xyz']
        self.X3, self.Y3, self.Z3 = self.corner_points['bottom_right']['xyz']
        self.X4, self.Y4, self.Z4 = self.corner_points['bottom_left']['xyz']

        self.MIN_RADIUS = 2

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
        img_resized = cv2.resize(img, (self.image_width, self.image_height))

        img_filter = cv2.GaussianBlur(img_resized.copy(), (3, 3), 0)

        img_hsv = cv2.cvtColor(img_filter, cv2.COLOR_BGR2HSV)

        if self.selected_color in self.color_ranges:
            THRESHOLD_LOW = self.color_ranges[self.selected_color]['low']
            THRESHOLD_HIGH = self.color_ranges[self.selected_color]['high']
        else:
            self.get_logger().warning(f"Selected color '{self.selected_color}' not defined.")
            return

        if self.selected_color.startswith('red'):
            mask1 = cv2.inRange(img_hsv, self.color_ranges['red1']['low'], self.color_ranges['red1']['high'])
            mask2 = cv2.inRange(img_hsv, self.color_ranges['red2']['low'], self.color_ranges['red2']['high'])
            img_binary = cv2.bitwise_or(mask1, mask2)
        else:
            # Binary image
            img_binary = cv2.inRange(img_hsv, THRESHOLD_LOW, THRESHOLD_HIGH)

        # Dilate to make blobs larger
        img_binary = cv2.dilate(img_binary, None, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None
        mapped_xyz = [0.0, 0.0, 0.0]  # Default values

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            M = cv2.moments(largest_contour)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius >= self.MIN_RADIUS:
                    # Draw circle around the detected object
                    cv2.circle(img_resized, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    # Draw center point
                    cv2.circle(img_resized, center, 5, (0, 0, 255), -1)
                    # Annotate center coordinates
                    cv2.putText(img_resized, f"Center: {center}", (center[0] + 10, center[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Map the center to XYZ coordinates
                    mapped_xyz = self.map_to_square(center)

        # Publish the coordinates
        msg = Float64MultiArray()
        msg.data = mapped_xyz
        self.publisher.publish(msg)

        # Log the published data
        self.get_logger().info(
            f"Published to /target_position: [X: {mapped_xyz[0]:.5f}, Y: {mapped_xyz[1]:.5f}, Z: {mapped_xyz[2]:.5f}]"
        )

        # Visualization: Display the processed image
        cv2.imshow("Processed Image", img_resized)
        cv2.waitKey(1)  # Necessary for OpenCV to update the window

    def map_to_square(self, center):
        """
        Maps the pixel coordinates to the predefined square's XYZ coordinates using bilinear interpolation.
        """
        x_pixel, y_pixel = center

        # Extract corner pixel coordinates
        x1, y1 = self.x1, self.y1  # Top Right
        x2, y2 = self.x2, self.y2  # Bottom Left

        # Calculate normalized coordinates [0,1]
        t = (x_pixel - x1) / (x2 - x1) if (x2 - x1) != 0 else 0
        u = (y_pixel - y1) / (y2 - y1) if (y2 - y1) != 0 else 0

        # Clamp t and u to [0,1] to handle points outside the defined rectangle
        t = max(0.0, min(t, 1.0))
        u = max(0.0, min(u, 1.0))

        # Bilinear Interpolation for X
        # X = (1-t)(1-u)X1 + t(1-u)X2 + (1-t)uX3 + t*uX4
        X = (1 - t) * (1 - u) * self.X1 + t * (1 - u) * self.X2 + (1 - t) * u * self.X3 + t * u * self.X4

        # Y Mapping based on x only (as per user specification)
        # Y decreases from 0.15 to -0.15 as x increases from 115 to 410
        Y = 0.15 - 0.3 * t

        # Bilinear Interpolation for Z
        # Z = (1-t)(1-u)Z1 + t(1-u)Z2 + (1-t)uZ3 + t*uZ4
        Z = (1 - t) * (1 - u) * self.Z1 + t * (1 - u) * self.Z2 + (1 - t) * u * self.Z3 + t * u * self.Z4

        self.get_logger().debug(
            f"Pixel Coordinates: ({x_pixel}, {y_pixel}) -> Mapped XYZ: ({X:.5f}, {Y:.5f}, {Z:.5f})"
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