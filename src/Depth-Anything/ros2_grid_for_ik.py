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


class DepthRedOneRangeNode(Node):
    def __init__(self):
        super().__init__('depth_red_one_range_node')

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
        # 2. Prepare transforms (DepthAnything)
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
        # 3. Define a single red color range in HSV
        # -------------------------------------------------------------
        # Example: detect red hue in [0, 10] range.
        # Adjust these if your red lies in a different hue band.
      
        #BLUE
        self.color_lower = np.array([80, 30, 80])
        self.color_upper = np.array([130, 255, 255])
        self.min_contour_area_red = 2000  # min area threshold for red object
        
        
        #RED
        #Lower HSV: [  0 124  30]
        #Upper HSV: [179 255 255]

        #self.color_lower = np.array([0, 124, 30])
        #self.color_upper = np.array([179, 255, 255])
        #self.min_contour_area_red = 2000  # min area threshold for red object

        # -------------------------------------------------------------
        # 4. Depth scale factor (for color-map visualization)
        # -------------------------------------------------------------
        self.depth_scale_factor = 2.0

        # -------------------------------------------------------------
        # 5. ROS Publishers
        # -------------------------------------------------------------
        self.red_publisher = self.create_publisher(Float64MultiArray, '/target_position', 10)
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
        # 6. OpenCV display stuff
        # -------------------------------------------------------------
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        self.get_logger().info("DepthRedOneRangeNode started. Subscribed to /video_frames.")

    def image_callback(self, msg):
        """ Main callback: Depth inference + single-range Red detection. """
        try:
            raw_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Resize for consistent usage
        raw_image = cv2.resize(raw_image, (640, 480))
        h, w = raw_image.shape[:2]

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

        # Depth inference (if you want to visualize or use it)
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image_transformed = self.transform({'image': image_rgb})['image']
        image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            depth_pred = self.depth_anything(image_tensor)

        # Resize depth to (640, 480)
        depth_resized = F.interpolate(depth_pred[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        depth_norm_01 = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)

        # Color map for visualization
        depth_scaled = depth_norm_01 * self.depth_scale_factor
        depth_scaled_clipped = torch.clamp(depth_scaled, 0.0, 1.0)
        depth_uint8 = (depth_scaled_clipped * 255.0).cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

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

        # Detect single-range red
        self.detect_red(hsv, final_result, w, h)

        # Show in a window
        cv2.imshow("Single-Range Red Detection", final_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def detect_red(self, hsv, final_result, width, height):
        """
        Detects red using a single HSV range.
        Publishes the center as [cx, cy] for demonstration.
        """
        # Single mask
        red_mask = cv2.inRange(hsv, self.color_lower, self.color_upper)

        # Morphological ops
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().info("No red object detected in single-range.")
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

        # Draw circle
        caption_height = 60  # match the offset from the earlier code
        cv2.circle(final_result, (cx, cy + caption_height), int(radius), (0, 255, 0), 2)
        cv2.circle(final_result, (cx, cy + caption_height), 5, (0, 0, 255), -1)

        text = f"Red Center: ({cx}, {cy})"
        cv2.putText(final_result, text, (cx + 10, cy + 10 + caption_height),
                    self.font, 0.5, (0, 255, 0), 1)

        self.get_logger().info(f"Detected single-range red => center=({cx},{cy})")

        # Example: Publish [cx, cy] (or any data you want)
        msg = Float64MultiArray()
        msg.data = [float(cx), float(cy)]
        self.red_publisher.publish(msg)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthRedOneRangeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
