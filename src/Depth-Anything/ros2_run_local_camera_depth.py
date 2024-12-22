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


class DepthRedMappingNode(Node):
    def __init__(self):
        super().__init__('depth_red_mapping_node')

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

        # Path to your checkpoint (adjust to your environment)
        state_dict_path = f'/home/devuser/src/Depth-Anything/checkpoints/depth_anything_{self.encoder_type}14.pth'
        self.depth_anything.load_state_dict(torch.load(state_dict_path, map_location=self.DEVICE))
        self.depth_anything.to(self.DEVICE)
        self.depth_anything.eval()

        total_params = sum(param.numel() for param in self.depth_anything.parameters())
        self.get_logger().info(f'Total parameters: {total_params / 1e6:.2f}M')

        # -------------------------------------------------------------
        # 2. Prepare transforms (same as original script)
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
        # 3. Define color detection settings
        # -------------------------------------------------------------
        # Red color range in HSV
        self.red_lower1 = np.array([0,   70, 50])
        self.red_upper1 = np.array([10,  255, 255])
        self.red_lower2 = np.array([170, 70, 50])
        self.red_upper2 = np.array([180, 255, 255])

        # Minimum area threshold for red object
        self.min_contour_area = 1000

        # -------------------------------------------------------------
        # 4. Depth scale factor
        # -------------------------------------------------------------
        # Used primarily to scale the color-map visualization or for your custom logic
        self.depth_scale_factor = 2.0

        # -------------------------------------------------------------
        # 5. Set up for 2D->3D plane mapping
        # -------------------------------------------------------------
        self.corner_points = {
            'top_left':     {'pixel': (410, 150), 'xyz': (0.2,  -0.15,  0.2)},
            'top_right':    {'pixel': (115, 150), 'xyz': (0.2,   0.15,  0.2)},
            'bottom_left':  {'pixel': (410, 375), 'xyz': (0.2,  -0.15, -0.05)},
            'bottom_right': {'pixel': (115, 375), 'xyz': (0.13,  0.15, -0.16)}
        }

        # Extract the pixel corners
        self.x1, self.y1 = self.corner_points['top_right']['pixel']     # (115, 150)
        self.x2, self.y2 = self.corner_points['bottom_left']['pixel']   # (410, 375)

        # Extract the corresponding xyz corners
        self.X1, self.Y1, self.Z1 = self.corner_points['top_right']['xyz']     # (0.2,  0.15,  0.2)
        self.X2, self.Y2, self.Z2 = self.corner_points['top_left']['xyz']      # (0.2, -0.15,  0.2)
        self.X3, self.Y3, self.Z3 = self.corner_points['bottom_right']['xyz']  # (0.13, 0.15, -0.16)
        self.X4, self.Y4, self.Z4 = self.corner_points['bottom_left']['xyz']   # (0.2, -0.15, -0.05)

        # -------------------------------------------------------------
        # 6. ROS Publishers/Subscribers
        # -------------------------------------------------------------
        # Publisher for the mapped 3D position
        self.publisher = self.create_publisher(Float64MultiArray, '/target_position', 10)

        # Subscriber for the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/video_frames',   # <-- Replace with your actual image topic name
            self.image_callback,
            10
        )
        self.br = CvBridge()

        # -------------------------------------------------------------
        # 7. Misc. OpenCV stuff for display
        # -------------------------------------------------------------
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2

        self.get_logger().info("DepthRedMappingNode started. Subscribed to /video_frames.")

    def image_callback(self, msg):
        """Callback to receive images from ROS 2 and run Depth + Red Detection + 2D->3D Mapping."""
        try:
            # Convert ROS Image to OpenCV
            raw_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Keep image at 640x480 (or adapt to your preference)
        raw_image = cv2.resize(raw_image, (640, 480))

        # -------------------------------------------------------------
        # Depth inference
        # -------------------------------------------------------------
        image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image_rgb.shape[:2]

        # Apply transforms
        image_transformed = self.transform({'image': image_rgb})['image']
        image_tensor = torch.from_numpy(image_transformed).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            depth = self.depth_anything(image_tensor)

        # Resize depth map to the original (h,w)
        depth_resized = F.interpolate(
            depth[None], (h, w),
            mode='bilinear', align_corners=False
        )[0, 0]

        # Normalize to [0,1]
        depth_min, depth_max = depth_resized.min(), depth_resized.max()
        depth_norm_01 = (depth_resized - depth_min) / (depth_max - depth_min + 1e-8)

        # Apply depth scale factor (mainly for color map visualization)
        depth_scaled = depth_norm_01 * self.depth_scale_factor
        depth_scaled_clipped = torch.clamp(depth_scaled, 0.0, 1.0)

        # Convert to 8-bit
        depth_uint8 = (depth_scaled_clipped * 255.0).cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        # -------------------------------------------------------------
        # Red object detection
        # -------------------------------------------------------------
        hsv = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations (open, dilate) to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # We'll create a side-by-side result image
        margin_width = 50
        split_region = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
        combined = cv2.hconcat([raw_image, split_region, depth_color])

        # Add captions on a separate band above
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

        # -------------------------------------------------------------
        # Determine if we have a valid red object
        # -------------------------------------------------------------
        if not contours:
            self.get_logger().info("No object detected.")
            # Show the result anyway
            cv2.imshow("Depth Red Mapping", final_result)
            cv2.waitKey(1)
            return

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < self.min_contour_area:
            self.get_logger().info("No object detected (contour too small).")
            cv2.imshow("Depth Red Mapping", final_result)
            cv2.waitKey(1)
            return

        # If we reach here, we have a valid detection
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

            # Clamp to avoid out-of-range indexing
            center_x = np.clip(center_x, 0, w - 1)
            center_y = np.clip(center_y, 0, h - 1)

            # -- Draw on final_result (offset for caption)
            cv2.circle(final_result, (center_x, center_y + caption_height),
                       int(radius), (0, 255, 0), 2)
            cv2.circle(final_result, (center_x, center_y + caption_height),
                       5, (0, 0, 255), -1)

            # 2D->3D Mapping (like your original function)
            mapped_xyz = self.map_to_square((center_x, center_y))

            # Display the mapped XYZ on the image
            text = f"X: {mapped_xyz[0]:.3f}, Y: {mapped_xyz[1]:.3f}, Z: {mapped_xyz[2]:.3f}"
            cv2.putText(final_result, text,
                        (center_x + 10, center_y + 10 + caption_height),
                        self.font, 0.5, (0, 255, 0), 1)

            self.get_logger().info(
                f"Detected Red Object - Center ({center_x}, {center_y}) => Mapped XYZ: {mapped_xyz}"
            )

            # ---------------------------------------------------------
            # Publish the [X, Y, Z] to /target_position
            # ---------------------------------------------------------
            msg = Float64MultiArray()
            msg.data = mapped_xyz
            self.publisher.publish(msg)
        else:
            self.get_logger().info("No object detected (invalid moments).")

        # Show the annotated result
        cv2.imshow("Depth Red Mapping", final_result)
        cv2.waitKey(1)

    def map_to_square(self, center):
        """
        Maps the 2D pixel coordinates (center_x, center_y)
        to 3D (X,Y,Z) using bilinear interpolation among
        the four known corner points.
        """
        x_pixel, y_pixel = center

        # Pixel corners
        x1, y1 = self.x1, self.y1  # top_right
        x2, y2 = self.x2, self.y2  # bottom_left

        # Normalize [0,1]
        # Watch out for x2-x1 or y2-y1 = 0
        t = (x_pixel - x1) / (x2 - x1) if (x2 - x1) != 0 else 0
        u = (y_pixel - y1) / (y2 - y1) if (y2 - y1) != 0 else 0

        # Clamp [0,1]
        t = max(0.0, min(t, 1.0))
        u = max(0.0, min(u, 1.0))

        # Corner XYZ
        # self.X1, self.Y1, self.Z1 = top_right
        # self.X2, self.Y2, self.Z2 = top_left
        # self.X3, self.Y3, self.Z3 = bottom_right
        # self.X4, self.Y4, self.Z4 = bottom_left

        # Bilinear Interpolation for X
        X = (1 - t) * (1 - u) * self.X1 \
            + t        * (1 - u) * self.X2 \
            + (1 - t)  * u       * self.X3 \
            + t        * u       * self.X4

        # (Example) Y linearly interpolated based on x only
        # Original formula from your code:
        # Y = 0.15 - 0.3 * t
        # or do a fully bilinear approach for Y as well if needed:
        # Y = (1 - t)(1 - u)Y1 + ...
        # But let's keep your simplified approach:
        Y = 0.15 - 0.3 * t

        # Bilinear Interpolation for Z
        Z = (1 - t) * (1 - u) * self.Z1 \
            + t        * (1 - u) * self.Z2 \
            + (1 - t)  * u       * self.Z3 \
            + t        * u       * self.Z4

        return [X, Y, Z]

    def destroy_node(self):
        """Destroy the node and close OpenCV windows when shutting down."""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthRedMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
