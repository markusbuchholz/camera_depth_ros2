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


class DepthRedGrid6Node(Node):
    def __init__(self):
        super().__init__('depth_red_grid_6_node')

        # -------------------------------------------------------------
        # 1. Load/initialize DepthAnything model (used for color map, if desired)
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
        self.red_lower1 = np.array([0,   70,  50])
        self.red_upper1 = np.array([10,  255, 255])
        self.red_lower2 = np.array([170, 70,  50])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.min_contour_area_red = 2000
        
        # We won't worry about white detection, etc. Focus on red object only
        self.white_lower = np.array([0, 0, 0])
        self.white_upper = np.array([0, 0, 0])
        self.min_contour_area_white = 0

        # -------------------------------------------------------------
        # 4. Depth scale (Removed depth_scale_factor as it is not needed)
        # -------------------------------------------------------------
        # self.depth_scale_factor = 2.0  # Removed

        # -------------------------------------------------------------
        # 5. Hard-coded 6 discrete positions
        #    We do a 2 (top vs. bottom) × 3 (left, center, right)
        #    So row ∈ {0=top, 1=bottom}, col ∈ {0=left, 1=center, 2=right}
        #    Each yields a unique (X, Y, Z)
        # -------------------------------------------------------------
        self.grid_6 = {
            'top_left':      (0.25,  0.15, 0.15),
            'top_center':    (0.25,  0.00, 0.15),
            'top_right':     (0.25, -0.15, 0.15),
            'bottom_left':   (0.25,  0.15, 0.15),
            'bottom_center': (0.25,  0.00, 0.00),
            'bottom_right':  (0.25, -0.15, 0.00)
        }

        # -------------------------------------------------------------
        # 6. ROS Publishers
        # -------------------------------------------------------------
        #self.red_publisher = self.create_publisher(Float64MultiArray, '/target_position', 10)
        
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
        # 7. OpenCV stuff
        # -------------------------------------------------------------
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        # Initialize depth_resized to None (Added)
        self.depth_resized = None  # To store the original depth map

        # Define depth thresholds and labels (Added)
        self.depth_thresholds = [
            (3.0, "Very Close"),
            (2.0, "Close"),
            (1.0, "Medium"),
            (0.5, "Far"),
            (0.2, "Very Far")
        ]

        self.get_logger().info("DepthRedGrid6Node started. Subscribed to /video_frames.")

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

        # Store the original scaled depth map for later use (Added)
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
        """Detect red object and map center to one of 6 discrete positions."""
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morph ops
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

        # Draw the circle
        cv2.circle(final_result, (cx, cy + caption_height), int(radius), (0, 255, 0), 2)
        cv2.circle(final_result, (cx, cy + caption_height), 5, (0, 0, 255), -1)

        # Map center to one of 6 discrete positions
        chosen_xyz = self.pick_6_grid_position(cx, cy, width, height)

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

        # Print & publish
        text = f"RED -> {chosen_xyz}"
        cv2.putText(final_result, text, (cx + 10, cy + 10 + caption_height),
                    self.font, 0.5, (0, 255, 0), 1)
        self.get_logger().info(f"Detected Red => center=({cx},{cy}), picked {chosen_xyz}, {depth_text}")

        msg = Float64MultiArray()
        msg.data = list(chosen_xyz)
        self.red_publisher.publish(msg)

    def pick_6_grid_position(self, cx, cy, width, height):
        """
        Divides the image (640x480) into 2 rows × 3 columns:
        - row = 0 => top,    row=1 => bottom
        - col = 0 => left,   col=1 => center, col=2 => right

        Returns one of 6 (X, Y, Z) from self.grid_6 dict.
        """
        # Each row is 240px high => row=0 if cy < 240 else row=1
        # Each col is approx 213px wide => col=0 if cx<213, col=1 if 213<=cx<426, col=2 if >=426
        row = 0 if cy < height / 2 else 1
        col_boundary_1 = width / 3   # ~213
        col_boundary_2 = 2 * width / 3 # ~426

        if cx < col_boundary_1:
            col = 0
        elif cx < col_boundary_2:
            col = 1
        else:
            col = 2

        if row == 0 and col == 0:
            return self.grid_6['top_left']
        if row == 0 and col == 1:
            return self.grid_6['top_center']
        if row == 0 and col == 2:
            return self.grid_6['top_right']

        if row == 1 and col == 0:
            return self.grid_6['bottom_left']
        if row == 1 and col == 1:
            return self.grid_6['bottom_center']
        if row == 1 and col == 2:
            return self.grid_6['bottom_right']

        # default fallback (should never happen)
        return (0.0, 0.0, 0.0)

    def map_depth_to_label(self, depth_z):
        """
        Maps the raw depth value to a descriptive label based on predefined thresholds.
        Example thresholds:
            - Z <= 0.2: "Very Far"
            - 0.2 < Z <= 1.0: "Far"
            - 1.0 < Z <= 2.0: "Medium"
            - 2.0 < Z <= 3.0: "Close"
            - Z > 3.0: "Very Close"

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
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
