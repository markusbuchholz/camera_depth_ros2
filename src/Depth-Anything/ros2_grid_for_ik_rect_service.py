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


class DepthRedWhiteMappingNode(Node):
    def __init__(self):
        super().__init__('depth_red_white_mapping_node')

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
        
        # Path to your checkpoint (adjust if needed)
        state_dict_path = f'/home/devuser/src/Depth-Anything/checkpoints/depth_anything_{self.encoder_type}14.pth'
        self.depth_anything.load_state_dict(torch.load(state_dict_path, map_location=self.DEVICE))
        self.depth_anything.to(self.DEVICE)
        self.depth_anything.eval()

        total_params = sum(param.numel() for param in self.depth_anything.parameters())
        self.get_logger().info(f'Total parameters: {total_params / 1e6:.2f}M')

        # -------------------------------------------------------------
        # 2. Prepare transforms
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

        # Red color range in HSV ORGINAL
        #self.red_lower1 = np.array([0,   70,  50])
        #self.red_upper1 = np.array([10,  255, 255])
        #self.red_lower2 = np.array([170, 70,  50])
        #self.red_upper2 = np.array([180, 255, 255])



        # -------------------------------------------------------------
        # 3. Color detection settings
        # -------------------------------------------------------------
        # Example: Overwriting ranges for "blue plate" or other color
        
        
        
        self.red_lower1 = np.array([80, 30, 80])    # Lower bound
        self.red_upper1 = np.array([130, 255, 255]) # Upper bound
        self.red_lower2 = np.array([80, 30, 80])
        self.red_upper2 = np.array([130, 255, 255])
        self.min_contour_area_red = 1000  



        # Red color range in HSV ORGINAL
        #self.red_lower1 = np.array([0,   70,  50])
        #self.red_upper1 = np.array([10,  255, 255])
        #self.red_lower2 = np.array([170, 70,  50])
        #self.red_upper2 = np.array([180, 255, 255])


        # White detection disabled here
        self.white_lower = np.array([0, 0, 0])
        self.white_upper = np.array([0, 0, 0])
        self.min_contour_area_white = 0

        # -------------------------------------------------------------
        # 4. Depth scale factor (for color-map visualization)
        # -------------------------------------------------------------
        self.depth_scale_factor = 2.0

        # -------------------------------------------------------------
        # 5. Corners for Red & White (for mapping)
        # -------------------------------------------------------------
        # We'll keep your original corner definitions for red
        self.corner_points_red = {
            'top_left':     {'pixel': (410, 150), 'xyz': (0.2, -0.15)},
            'top_right':    {'pixel': (115, 150), 'xyz': (0.2,  0.15)},
            'bottom_left':  {'pixel': (410, 375), 'xyz': (0.2, -0.15)},
            'bottom_right': {'pixel': (115, 375), 'xyz': (0.2,  0.15)}
        }
        self.x1r, self.y1r = self.corner_points_red['top_right']['pixel']
        self.x2r, self.y2r = self.corner_points_red['bottom_left']['pixel']

        self.XR1, self.YR1 = self.corner_points_red['top_right']['xyz']
        self.XR2, self.YR2 = self.corner_points_red['top_left']['xyz']
        self.XR3, self.YR3 = self.corner_points_red['bottom_right']['xyz']
        self.XR4, self.YR4 = self.corner_points_red['bottom_left']['xyz']

        # For white (unused in this snippet, but we keep for reference)
        self.corner_points_white = {
            'top_left':     {'pixel': (100, 100), 'xyz': (0.0,  1.25)},
            'top_right':    {'pixel': (540, 100), 'xyz': (3.0,  1.25)},
            'bottom_left':  {'pixel': (100, 380), 'xyz': (0.0, -1.25)},
            'bottom_right': {'pixel': (540, 380), 'xyz': (3.0, -1.25)},
        }
        self.x1w, self.y1w = self.corner_points_white['top_right']['pixel']
        self.x2w, self.y2w = self.corner_points_white['bottom_left']['pixel']

        self.XW1, self.YW1 = self.corner_points_white['top_right']['xyz']
        self.XW2, self.YW2 = self.corner_points_white['top_left']['xyz']
        self.XW3, self.YW3 = self.corner_points_white['bottom_right']['xyz']
        self.XW4, self.YW4 = self.corner_points_white['bottom_left']['xyz']

        # -------------------------------------------------------------
        # 6. ROS Publishers/Subscribers
        # -------------------------------------------------------------
        self.red_publisher = self.create_publisher(Float64MultiArray, '/object_localizer', 10)
        self.white_publisher = self.create_publisher(Float64MultiArray, '/obstacle_position', 10)

        self.subscription = self.create_subscription(
            Image,
            '/video_frames',   # your camera topic
            self.image_callback,
            10
        )
        self.br = CvBridge()

        # -------------------------------------------------------------
        # 7. OpenCV display
        # -------------------------------------------------------------
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        self.get_logger().info("DepthRedWhiteMappingNode started. Subscribed to /video_frames.")

    def image_callback(self, msg):
        """ Main callback: Depth inference + red detection (+ optional white). """
        try:
            raw_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Resize
        raw_image = cv2.resize(raw_image, (640, 480))
        h, w = raw_image.shape[:2]
        hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

        # Depth inference
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image_transformed = self.transform({'image': image_rgb})['image']
        image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            depth_pred = self.depth_anything(image_tensor)

        depth_resized = F.interpolate(
            depth_pred[None], (h, w),
            mode='bilinear', align_corners=False
        )[0, 0]

        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        depth_norm_01 = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)

        # For color visualization
        depth_scaled = depth_norm_01 * self.depth_scale_factor
        depth_scaled_clipped = torch.clamp(depth_scaled, 0.0, 1.0)
        depth_uint8 = (depth_scaled_clipped * 255.0).cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        # Combine frames side by side
        margin_width = 50
        split_region = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([raw_image, split_region, depth_color])

        # Add captions on top
        caption_height = 60
        caption_space = np.ones((caption_height, combined.shape[1], 3), dtype=np.uint8) * 255
        captions = ['Raw Image', 'Depth Map']
        segment_width = w + margin_width
        for i, ctext in enumerate(captions):
            text_size = cv2.getTextSize(ctext, self.font, self.font_scale, self.font_thickness)[0]
            text_x = int((segment_width * i) + (w - text_size[0]) / 2)
            cv2.putText(caption_space, ctext, (text_x, 40),
                        self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        final_result = cv2.vconcat([caption_space, combined])

        # Detect red (with bounding rectangle)
        self.detect_red(hsv, final_result, w, h, caption_height, depth_norm_01)

        # If needed, detect white
        # self.detect_white(hsv, final_result, w, h, caption_height, depth_norm_01)

        cv2.imshow("Depth Red + White Mapping", final_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def detect_red(self, hsv, final_result, width, height, caption_height, depth_norm_01):
        """
        Finds the largest red contour and draws a bounding rectangle
        (like the white object approach).
        """
        # Combine two red masks
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

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_contour_area_red:
            self.get_logger().info("No red object detected (contour too small).")
            return

        # Get bounding rectangle of largest contour
        x, y, w_box, h_box = cv2.boundingRect(largest)
        M = cv2.moments(largest)
        if M["m00"] <= 0:
            self.get_logger().info("No red object detected (invalid moments).")
            return

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cx = np.clip(cx, 0, width - 1)
        cy = np.clip(cy, 0, height - 1)

        # Draw bounding rectangle for red object
        cv2.rectangle(final_result,
                      (x, y + caption_height),
                      (x + w_box, y + h_box + caption_height),
                      (0, 255, 0), 2)
        # Mark center with a small circle or bigger rectangle as you wish
        cv2.circle(final_result, (cx, cy + caption_height),
                   5, (0, 0, 255), -1)

        # Map to X, Y, Z
        mapped_xyz = self.map_to_square_red(cx, cy, depth_norm_01)
        text = f"R: X={mapped_xyz[0]:.3f},Y={mapped_xyz[1]:.3f},Z={mapped_xyz[2]:.3f}"
        cv2.putText(final_result, text,
                    (cx + 10, cy + 10 + caption_height),
                    self.font, 0.5, (0, 255, 0), 1)

        self.get_logger().info(
            f"Detected Red Object -> boundingRect=({x},{y},{w_box},{h_box}), center=({cx},{cy}), XYZ={mapped_xyz}"
        )

        # Publish
        msg = Float64MultiArray()
        msg.data = mapped_xyz
        self.red_publisher.publish(msg)

    def map_to_square_red(self, px, py, depth_norm_01):
        """
        For Red: we do the same bilinear approach from your corners,
        plus a scaled depth from 'depth_norm_01'.
        """
        x1, y1 = self.x1r, self.y1r   # top_right
        x2, y2 = self.x2r, self.y2r   # bottom_left

        t = (px - x1) / (x2 - x1) if (x2 - x1) else 0
        u = (py - y1) / (y2 - y1) if (y2 - y1) else 0
        t = max(0.0, min(t, 1.0))
        u = max(0.0, min(u, 1.0))

        X = ((1 - t) * (1 - u) * self.XR1
             + t * (1 - u) * self.XR2
             + (1 - t) * u * self.XR3
             + t * u * self.XR4)

        Y = ((1 - t) * (1 - u) * self.YR1
             + t * (1 - u) * self.YR2
             + (1 - t) * u * self.YR3
             + t * u * self.YR4)

        z_val_01 = depth_norm_01[py, px].item()
        Z = (1.0 - z_val_01) * 0.2   # or any range scaling

        return [X, Y, Z]

    # (Optional) White detection if needed
    # def detect_white(...)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthRedWhiteMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
