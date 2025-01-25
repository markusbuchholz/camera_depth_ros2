#!/usr/bin/env python3

# Markus Buchholz, 2025

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
# Removed trajectory_msgs import since it's no longer used

from cv_bridge import CvBridge, CvBridgeError
import cv2
import mediapipe as mp
import math
import numpy as np

class HandPoseEstimationNode(Node):
    def __init__(self):
        super().__init__('hand_pose_estimation_node')

        # Initialize MediaPipe Hands.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,       # Video stream
            max_num_hands=2,               # Maximum number of hands to detect
            min_detection_confidence=0.5,  # Minimum confidence for detection
            min_tracking_confidence=0.5    # Minimum confidence for tracking
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.bridge = CvBridge()

        self.image_subscriber = self.create_subscription(
            Image,
            'video_frames',
            self.image_callback,
            10
        )

        self.joint_states_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )

        self.forward_position_publisher = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )

        self.target_position_publisher = self.create_publisher(
            Float64MultiArray,
            '/target_position',
            10
        )

        self.previous_states = {}  # To track previous gripper states per hand
        self.latest_joint_states = JointState()  # To store the latest joint states

        self.display_scale = 2.0

        self.get_logger().info('Hand Pose Estimation Node has been started.')

    def joint_states_callback(self, msg):
        """
        Callback function for '/joint_states' topic.
        Stores the latest joint states.
        """
        self.latest_joint_states = msg

    def image_callback(self, msg):
        """
        Callback function for 'video_frames' topic.
        Processes the incoming image for hand pose estimation.
        """
        try:
            # Convert ROS Image message to OpenCV image.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        image_height, image_width, _ = cv_image.shape

        # Remove horizontal flipping to correct mirroring.
        # If you want to flip vertically or apply other transformations, modify here.
        # cv_image = cv2.flip(cv_image, 1)  # Original horizontal flip

        # Convert the BGR image to RGB.
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands.
        results = self.hands.process(rgb_image)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks on the frame.
                self.mp_draw.draw_landmarks(
                    cv_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Get hand label (Left/Right)
                hand_label = handedness.classification[0].label

                # Initialize previous state for this hand if not present
                if hand_label not in self.previous_states:
                    self.previous_states[hand_label] = None

                # Determine gripper state
                gripper_state = self.determine_gripper_state(
                    hand_landmarks, image_width, image_height, previous_state=self.previous_states[hand_label]
                )

                # Update previous state
                self.previous_states[hand_label] = gripper_state

                # Get landmark 0 coordinates (Wrist)
                wrist = hand_landmarks.landmark[0]
                wrist_x = wrist.x * image_width
                wrist_y = wrist.y * image_height

                # Calculate image center
                center_x = image_width / 2
                center_y = image_height / 2

                # Normalize coordinates to new ranges
                # X: 0.0 (bottom) to 0.5 (top)
                # Y: -0.5 (right) to 0.5 (left)
                norm_x = 0.5 - (wrist_y / image_height) * 0.5  # X-axis mapping
                norm_y = (wrist_x / image_width) - 0.5          # Y-axis mapping
                norm_y = -norm_y  # Swap Y-axis left and right

                # Prepare the normalized position data
                target_position_msg = Float64MultiArray()
                target_position_msg.data = [norm_x, norm_y, 0.0]

                # Publish the target position
                self.target_position_publisher.publish(target_position_msg)

                # Print the gripper state and normalized coordinates to the terminal with hand label
                self.get_logger().info(
                    f"{hand_label}: {gripper_state}, Wrist Position -> X: {norm_x:.3f}, Y: {norm_y:.3f}"
                )

                # Publish forward position controller commands
                self.publish_forward_position_commands(gripper_state)

                # Determine color based on gripper state
                color = (0, 255, 0) if gripper_state == "Gripper Open" else (0, 0, 255)

                # Prepare the text to display
                display_text = f"{hand_label}: {gripper_state} | X: {norm_x:.3f}, Y: {norm_y:.3f}"

                # Position the text on the frame
                # Adjust the y-coordinate based on hand label to prevent overlap
                text_y = 30 if hand_label == "Right" else 60

                cv2.putText(cv_image, display_text, (10, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        else:
            # Optional: Reset previous states when no hands are detected
            self.previous_states = {}

        # Resize the image for display
        scaled_image = cv2.resize(cv_image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_LINEAR)

        # Display the scaled frame.
        cv2.imshow('Hand Pose Estimation', scaled_image)
        cv2.waitKey(1)

    def determine_gripper_state(self, hand_landmarks, image_width, image_height, previous_state=None):
        """
        Determine whether the gripper is open or closed based on the positions
        of Thumb Tip (landmark 4) and Index Finger Tip (landmark 8).

        Parameters:
            hand_landmarks: The landmarks detected by MediaPipe for a single hand.
            image_width: Width of the image frame.
            image_height: Height of the image frame.
            previous_state: The previous state of the gripper to implement hysteresis.

        Returns:
            state (str): "Gripper Open" or "Gripper Closed"
        """
        # Extract landmark positions
        landmarks = hand_landmarks.landmark

        # Get wrist coordinates for normalization (landmark 0)
        wrist = landmarks[0]
        wrist_coords = (wrist.x * image_width, wrist.y * image_height, wrist.z)

        # Get middle finger MCP for normalization (landmark 9)
        middle_mcp = landmarks[9]
        middle_mcp_coords = (middle_mcp.x * image_width, middle_mcp.y * image_height, middle_mcp.z)

        # Calculate reference distance (wrist to middle MCP)
        reference_distance = self.calculate_distance(wrist_coords, middle_mcp_coords)
        if reference_distance == 0:
            reference_distance = 1  # Prevent division by zero

        # Get Thumb Tip (landmark 4) and Index Finger Tip (landmark 8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_tip_coords = (thumb_tip.x * image_width, thumb_tip.y * image_height, thumb_tip.z)
        index_tip_coords = (index_tip.x * image_width, index_tip.y * image_height, index_tip.z)

        # Calculate distance between Thumb Tip and Index Finger Tip
        distance = self.calculate_distance(thumb_tip_coords, index_tip_coords)

        # Normalize the distance
        normalized_distance = distance / reference_distance

        # Define threshold and margin for hysteresis
        threshold = 0.4  # Base threshold; adjust based on testing
        margin = 0.05    # Margin to prevent rapid toggling

        # Initialize state
        state = "Unknown"

        if previous_state is None:
            # Initial state determination
            if normalized_distance > threshold:
                state = "Gripper Open"
            else:
                state = "Gripper Closed"
        else:
            # Hysteresis implementation
            if previous_state == "Gripper Open":
                if normalized_distance < (threshold - margin):
                    state = "Gripper Closed"
                else:
                    state = "Gripper Open"
            elif previous_state == "Gripper Closed":
                if normalized_distance > (threshold + margin):
                    state = "Gripper Open"
                else:
                    state = "Gripper Closed"

        # Debug logs
        self.get_logger().debug(f"Normalized Distance: {normalized_distance:.3f}, Threshold: {threshold}, Margin: {margin}")
        self.get_logger().debug(f"Previous State: {previous_state}, New State: {state}")

        return state

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        Each point is a tuple of (x, y, z).
        """
        return math.sqrt((point1[0] - point2[0])**2 +
                         (point1[1] - point2[1])**2 +
                         (point1[2] - point2[2])**2)

    def publish_forward_position_commands(self, gripper_state):
        """
        Publish a Float64MultiArray message to '/forward_position_controller/commands'
        based on the gripper state and current joint positions.

        Parameters:
            gripper_state (str): Current state of the gripper ("Gripper Open" or "Gripper Closed").
        """
        # Ensure that joint names and positions are available
        if not self.latest_joint_states.name or not self.latest_joint_states.position:
            self.get_logger().warning('No joint states received yet.')
            return

        # Define the joint names we are interested in
        target_joint_names = ['axis_a', 'axis_b', 'axis_c', 'axis_d', 'axis_e']

        # Initialize the positions list
        positions = [0.0] * len(target_joint_names)

        # Populate the positions based on latest_joint_states
        for idx, joint_name in enumerate(target_joint_names):
            if joint_name in self.latest_joint_states.name:
                joint_idx = self.latest_joint_states.name.index(joint_name)
                positions[idx] = self.latest_joint_states.position[joint_idx]
            else:
                # If joint name not found, assume 0.0
                positions[idx] = 0.0
                self.get_logger().warning(f"Joint '{joint_name}' not found in /joint_states.")

        # Update axis_a based on gripper state
        if gripper_state == "Gripper Open":
            positions[0] = 0.1  # axis_a
        elif gripper_state == "Gripper Closed":
            positions[0] = 0.0  # axis_a
        else:
            self.get_logger().warning(f"Unknown gripper state: {gripper_state}")

        # Log the positions before publishing
        self.get_logger().debug(f"Command Positions: {positions}")

        # Create the Float64MultiArray message
        cmd_msg = Float64MultiArray()
        cmd_msg.data = positions

        # Publish the command
        self.forward_position_publisher.publish(cmd_msg)
        self.get_logger().info(f'Published Forward Position Commands: {positions}')

    def __del__(self):
        # Close all OpenCV windows upon node destruction
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = HandPoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
