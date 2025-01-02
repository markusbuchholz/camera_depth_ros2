#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

import torch
import torch.nn.functional as F

import cv2
import numpy as np

from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from std_srvs.srv import Trigger  # Correct import for Trigger service
import threading  # For thread safety


class DepthRedGrid6Node(Node):
    def __init__(self):
        super().__init__('depth_red_grid_6_node')

        # -------------------------------------------------------------
        # 1. Load/initialize DepthAnything model
        # -------------------------------------------------------------
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_type = 'vits'  # 'vits', 'vitb', or 'vitl'

        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]}
        }
        self.depth_anything = DepthAnything(model_configs[self.encoder_type])

        # Path to your checkpoint
        state_dict_path = f'/home/devuser/src/Depth-Anything/checkpoints/depth_anything_{self.encoder_type}14.pth'
        self.depth_anything.load_state_dict(torch.load(state_dict_path, map_location=self.DEVICE))
        self.depth_anything.to(self.DEVICE)
        self.depth_anything.eval()

        total_params = sum(param.numel() for param in self.depth_anything.parameters())
        self.get_logger().info(f'Total parameters: {total_params / 1e6:.2f}M')

        # -------------------------------------------------------------
        # 2. Transforms (for DepthAnything)
        # -------------------------------------------------------------
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # -------------------------------------------------------------
        # 3. Red color detection in HSV
        # -------------------------------------------------------------
        self.red_lower1 = np.array([0, 70, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 70, 50])
        self.red_upper2 = np.array([180, 255, 255])

        self.min_contour_area_red = 2000

        # -------------------------------------------------------------
        # 4. ROS Publishers
        # -------------------------------------------------------------
        self.red_publisher = self.create_publisher(Float64MultiArray, '/object_localizer', 10)

        # Subscriber for camera frames
        self.subscription = self.create_subscription(
            Image,
            '/video_frames',  # your camera topic
            self.image_callback,
            10
        )
        self.br = CvBridge()

        # -------------------------------------------------------------
        # 5. OpenCV stuff
        # -------------------------------------------------------------
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        # Initialize depth_resized to None
        self.depth_resized = None  # To store the original depth map

        # Define fixed_max_depth as an instance variable
        self.fixed_max_depth = 3.0  # meters

        # Define depth thresholds and labels (optional for debugging)
        self.depth_thresholds = [
            (0.2, "Very Far"),
            (0.5, "Far"),
            (1.0, "Medium"),
            (2.0, "Close"),
            (3.0, "Very Close")
        ]

        # -------------------------------------------------------------
        # 6. Goal Achievement Service
        # -------------------------------------------------------------
        self.goal_achieved = False  # Initialize the goal status
        self.lock = threading.Lock()  # To ensure thread safety
        self.srv = self.create_service(Trigger, '/goal_achived', self.handle_goal_achived)
        self.get_logger().info('Service /goal_achived created and ready.')

        self.get_logger().info("DepthRedGrid6Node started. Subscribed to /video_frames.")

    def handle_goal_achived(self, request, response):
        """Service callback to report whether the goal has been achieved."""
        with self.lock:
            response.success = self.goal_achieved
            response.message = f'Goal achieved: {self.goal_achieved}'
        return response

    def image_callback(self, msg):
        """ Main callback. Depth inference + Red detection. """
        try:
            raw_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Resize for consistent usage
        raw_image = cv2.resize(raw_image, (640, 480))
        h, w = raw_image.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

        # Depth inference (optional for color map display)
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image_transformed = self.transform({'image': image_rgb})['image']
        image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            depth_pred = self.depth_anything(image_tensor)

        depth_resized = F.interpolate(depth_pred[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        # Linear scaling to map depth_resized to [0.2, 3.0] with inversion
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        desired_min = 0.2
        desired_max = 3.0
        depth_scaled = desired_min + (depth_max - depth_resized) * (desired_max - desired_min) / (depth_max - depth_min + 1e-8)
        depth_scaled = torch.clamp(depth_scaled, desired_min, desired_max)  # Ensure values are within [0.2, 3.0]

        # For color map display: scale to [0,255]
        depth_uint8 = (depth_scaled / desired_max * 255.0).cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        # Store the original scaled depth map for later use
        self.depth_resized = depth_scaled.cpu().numpy()  # Convert to NumPy array for easier indexing

        # Side-by-side display
        margin_width = 50
        split_region = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([raw_image, split_region, depth_color])

        caption_height = 60
        caption_space = np.ones((caption_height, combined.shape[1], 3), dtype=np.uint8) * 255
        captions = ['Raw Image', 'Depth Map']
        segment_width = w + margin_width
        for i, caption in enumerate(captions):
            text_size = cv2.getTextSize(caption, self.font, self.font_scale, self.font_thickness)[0]
            text_x = int((segment_width * i) + (w - text_size[0]) / 2)
            cv2.putText(caption_space, caption, (text_x, 40),
                        self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        final_result = cv2.vconcat([caption_space, combined])

        # Detect the largest red object
        self.detect_red(hsv, final_result, w, h, caption_height)

        cv2.imshow("Depth Red 6-Grid Mapping", final_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def detect_red(self, hsv, final_result, width, height, caption_height):
        """Detect red object and map center to real-world coordinates."""
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().info("No red object detected.")
            return

        # Largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_contour_area_red:
            self.get_logger().info("No red object detected (too small).")
            return

        ((x_c, y_c), radius) = cv2.minEnclosingCircle(largest)
        M = cv2.moments(largest)
        if M["m00"] <= 0:
            self.get_logger().info("No red object detected (invalid moments).")
            return

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cx = np.clip(cx, 0, width - 1)
        cy = np.clip(cy, 0, height - 1)

        # Draw the circle on the final_result image
        cv2.circle(final_result, (cx, cy + caption_height), int(radius), (0, 255, 0), 2)
        cv2.circle(final_result, (cx, cy + caption_height), 5, (0, 0, 255), -1)

        # Dynamic Mapping: Map (cx, cy) to (Y, Z)
        Y, Z = self.map_center_to_coordinates(cx, cy, width, height)

        # Retrieve depth information at the center point (Correctly scaled and inverted)
        if self.depth_resized is not None:
            depth_z = self.depth_resized[cy, cx]  # Access depth at (cx, cy)
            depth_label = self.map_depth_to_label(depth_z)  # Map depth to label
            depth_text = f"Depth Z: {depth_z:.2f}m ({depth_label})"
            # Print depth on the screen near the center point
            cv2.putText(final_result, depth_text, (cx + 10, cy + 30 + caption_height),
                        self.font, 0.5, (255, 0, 0), 1)
        else:
            depth_text = "Depth Z: N/A"
            cv2.putText(final_result, depth_text, (cx + 10, cy + 30 + caption_height),
                        self.font, 0.5, (255, 0, 0), 1)
            depth_z = 0.0  # Default value if depth is not available

        X = depth_z - 1.2 
        
        # Clamp Y and Z as per requirements
        Y = max(min(Y, 0.15), -0.15)
        Z = max(min(Z, 0.15), 0.0)

        # Overlay position information on the image
        position_text = f"RED -> (X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f})"
        cv2.putText(final_result, position_text, (cx + 10, cy + 10 + caption_height),
                    self.font, self.font_scale, (0, 255, 0), 1)

        # Check if X is below 0.5 meters
        with self.lock:
            previous_goal = self.goal_achieved  # Store previous state for logging
            if X < 0.5:
                self.goal_achieved = True
            else:
                self.goal_achieved = False

        # Log only if the goal status has changed
        if self.goal_achieved and not previous_goal:
            self.get_logger().info(f"Goal achieved: X ({X:.2f}m) is below 0.5 meters.")
        elif not self.goal_achieved and previous_goal:
            self.get_logger().info(f"Goal not achieved: X ({X:.2f}m) is above or equal to 0.5 meters.")

        # Log the detected position and depth value
        self.get_logger().info(f"Detected Red => center=({cx},{cy}), Depth: {depth_z:.2f}m, Position=(X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f})")

        # Prepare and publish the message
        msg = Float64MultiArray()
        msg.data = [float(X), float(Y), float(Z)]  # Ensure all elements are Python floats
        self.red_publisher.publish(msg)

    def map_center_to_coordinates(self, cx, cy, width, height):
        """
        Maps image coordinates (cx, cy) to real-world coordinates (Y, Z).

        Y is mapped based on horizontal position:
            - cx = 0      => Y = +0.15
            - cx = width  => Y = -0.15

        Z is mapped based on vertical position:
            - cy = 0      => Z = +0.15
            - cy = height => Z = -0.15

        Parameters:
            cx (int): X-coordinate in the image.
            cy (int): Y-coordinate in the image.
            width (int): Width of the image.
            height (int): Height of the image.

        Returns:
            tuple: (Y, Z) in meters.
        """
        # Map cx from [0, width] to [0.15, -0.15]
        Y = ((width / 2 - cx) / (width / 2)) * 0.15

        # Map cy from [0, height] to [0.15, -0.15]
        Z = ((height / 2 - cy) / (height / 2)) * 0.15

        return Y, Z

    def map_depth_to_label(self, depth_z):
        """
        Maps the raw depth value to a descriptive label based on predefined thresholds.
        Example thresholds:
            - Z <= 0.2: "Very Far"
            - 0.2 < Z <= 0.5: "Far"
            - 0.5 < Z <= 1.0: "Medium"
            - 1.0 < Z <= 2.0: "Close"
            - 2.0 < Z <= 3.0: "Very Close"

        You can adjust these thresholds and labels as per your requirements.
        """
        for threshold, label in self.depth_thresholds:
            if depth_z <= threshold:
                return label
        return "Unknown"

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthRedGrid6Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("DepthRedGrid6Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
