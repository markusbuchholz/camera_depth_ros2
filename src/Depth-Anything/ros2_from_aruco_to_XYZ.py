#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError

import torch
import torch.nn.functional as F

import cv2
import cv2.aruco as aruco
import numpy as np

from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from std_srvs.srv import Trigger  # Correct import for Trigger service
import threading  # For thread safety


class DepthArucoGrid6Node(Node):
    def __init__(self):
        super().__init__('depth_aruco_grid_6_node')

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
        # 3. ArUco Marker Detection Initialization
        # -------------------------------------------------------------
        # Create a dictionary and detector parameters for ArUco
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

        # Determine OpenCV version to choose the correct API
        cv_version = cv2.__version__
        major_version = int(cv_version.split('.')[0])
        minor_version = int(cv_version.split('.')[1])

        if major_version >= 4 and minor_version >= 7:
            # OpenCV 4.7.0 and above
            self.aruco_params = aruco.DetectorParameters()
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.get_logger().info("Using ArUcoDetector (OpenCV 4.7.0+)")
        elif major_version >= 4:
            # OpenCV 4.x (below 4.7.0)
            if hasattr(aruco, 'DetectorParameters_create'):
                self.aruco_params = aruco.DetectorParameters_create()
                self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
                self.get_logger().info("Using ArUcoDetector with DetectorParameters_create (OpenCV 4.x)")
            else:
                # Fallback to detectMarkers function
                self.aruco_params = aruco.DetectorParameters()
                self.detector = None
                self.get_logger().info("Using detectMarkers function (OpenCV 4.x without DetectorParameters_create)")
        else:
            # OpenCV 3.x and below
            self.aruco_params = aruco.DetectorParameters_create()
            self.detector = None
            self.get_logger().info("Using detectMarkers function (OpenCV 3.x)")

        # -------------------------------------------------------------
        # 4. ROS Publishers
        # -------------------------------------------------------------
        self.marker_publisher = self.create_publisher(Float64MultiArray, '/marker_localizer', 10)

        # Subscriber for camera frames
        self.subscription = self.create_subscription(
            Image,
            '/video_frames',  # your camera topic
            self.image_callback,
            10
        )
        self.br = CvBridge()

        # -------------------------------------------------------------
        # 5. OpenCV Display Settings
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

        self.get_logger().info("DepthArucoGrid6Node started. Subscribed to /video_frames.")

    def handle_goal_achived(self, request, response):
        """Service callback to report whether the goal has been achieved."""
        with self.lock:
            response.success = self.goal_achieved
            response.message = f'Goal achieved: {self.goal_achieved}'
        return response

    def image_callback(self, msg):
        """ Main callback. Depth inference + ArUco detection. """
        try:
            raw_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Resize for consistent usage
        raw_image = cv2.resize(raw_image, (640, 480))
        h, w = raw_image.shape[:2]

        # Detect ArUco markers on the raw_image first
        self.detect_aruco(raw_image)

        # Convert to RGB for depth inference
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

        # Display the final combined image
        cv2.imshow("Depth ArUco 6-Grid Mapping", final_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def detect_aruco(self, raw_image):
        """Detect ArUco markers and map their centers to real-world coordinates."""
        try:
            if self.detector:
                # Using ArUcoDetector class (OpenCV 4.7.0+ or 4.x with DetectorParameters_create)
                corners, ids, rejected = self.detector.detectMarkers(raw_image)
            else:
                # Using detectMarkers function (Older OpenCV versions)
                corners, ids, rejected = aruco.detectMarkers(raw_image, self.aruco_dict, parameters=self.aruco_params)
        except Exception as e:
            self.get_logger().error(f'ArUco Detection Error: {e}')
            return

        if ids is not None and len(ids) > 0:
            # Draw detected markers on the raw_image
            aruco.drawDetectedMarkers(raw_image, corners, ids)
            self.get_logger().info(f'Detected ArUco markers with IDs: {ids.flatten()}')

            # Iterate through each detected marker
            for i, marker_id in enumerate(ids.flatten()):
                # Get the corners of the marker
                marker_corners = corners[i].reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = marker_corners

                # Calculate the center of the marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                # Print the location on the terminal
                self.get_logger().info(f"Marker ID {marker_id} - Center: (X: {cX}, Y: {cY})")

                # Create a Float64MultiArray message for the marker center
                marker_msg = Float64MultiArray()
                marker_msg.data = [float(cX), float(cY)]

                # Publish the marker center location
                self.marker_publisher.publish(marker_msg)

                # Proceed with depth and coordinate mapping for the marker
                self.process_marker(cX, cY, raw_image)  # Pass raw_image here

                # If you want to process only the first detected marker, uncomment the next line
                # break
        else:
            self.get_logger().info('No ArUco markers detected.')

    def process_marker(self, cx, cy, raw_image):
        """Process the detected marker's center with depth and coordinate mapping."""
        # Ensure depth_resized is available
        if self.depth_resized is not None:
            # Clamp coordinates to prevent IndexError
            cx_clamped = int(np.clip(cx, 0, self.depth_resized.shape[1] - 1))
            cy_clamped = int(np.clip(cy, 0, self.depth_resized.shape[0] - 1))

            # Retrieve depth information at the center point
            depth_z = self.depth_resized[cy_clamped, cx_clamped]  # Access depth at (cx, cy)
            depth_label = self.map_depth_to_label(depth_z)  # Map depth to label

            # Dynamic Mapping: Map (cx, cy) to (Y, Z)
            Y, Z = self.map_center_to_coordinates(cx, cy, self.depth_resized.shape[1], self.depth_resized.shape[0])

            X = depth_z - 1.2

            # Clamp Y and Z as per requirements
            Y = max(min(Y, 0.15), -0.15)
            Z = max(min(Z, 0.15), 0.0)

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
            self.get_logger().info(f"Detected Marker => center=({cx},{cy}), Depth: {depth_z:.2f}m, Position=(X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f})")

            # Prepare and publish the message
            msg = Float64MultiArray()
            msg.data = [float(X), float(Y), float(Z)]  # Ensure all elements are Python floats
            self.marker_publisher.publish(msg)

            # Draw a circle at the center point
            cv2.circle(raw_image, (cx, cy), 5, (0, 0, 255), -1)

            # Overlay the mapped coordinates near the center point
            text = f"({X:.2f}, {Y:.2f}, {Z:.2f})"
            text_position = (cx + 10, cy + 10)  # Adjust as needed to prevent overlap
            cv2.putText(raw_image, text, text_position,
                        self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            self.get_logger().warning("Depth information is not available.")

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
    node = DepthArucoGrid6Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("DepthArucoGrid6Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
