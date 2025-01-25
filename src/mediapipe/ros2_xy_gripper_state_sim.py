#!/usr/bin/env python3

#Markus Buchholz, 2025
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

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

        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/xsubsea/joint_trajectory',
            10
        )

        self.target_position_publisher = self.create_publisher(
            Float64MultiArray,
            '/target_position',
            10
        )

        # Initialize variables.
        self.previous_states = {}  # To track previous gripper states per hand
        self.latest_joint_states = JointState()  # To store the latest joint states

        self.get_logger().info('Hand Pose Estimation Node has been started.')

    def joint_states_callback(self, msg):
        """
        Callback function for '/joint_states' topic.
        Stores the latest joint states.
        """
        self.latest_joint_states = msg

    def image_callback(self, msg):
  
        try:
            # Convert ROS Image message to OpenCV image.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        image_height, image_width, _ = cv_image.shape

        # frame = cv2.flip(frame, 1)  # Original flipping

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

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

                # Normalize coordinates to range [-0.25, 0.25]
                norm_x = -((wrist_x - center_x) / image_width) * 0.5  # Inverted X-axis
                norm_y = -((wrist_y - center_y) / image_height) * 0.5  # Inverted Y-axis

                # Prepare the normalized position data
                target_position_msg = Float64MultiArray()
                target_position_msg.data = [0.25, norm_x, norm_y]

                self.target_position_publisher.publish(target_position_msg)

                # Print the gripper state and normalized coordinates to the terminal with hand label
                self.get_logger().info(
                    f"{hand_label}: {gripper_state}, Wrist Position -> X: {norm_x:.3f}, Y: {norm_y:.3f}"
                )

                # Publish joint trajectory if gripper state has changed
                self.publish_joint_trajectory(hand_label, gripper_state)

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

        # Display the frame.
        cv2.imshow('Hand Pose Estimation', cv_image)
        cv2.waitKey(1)

    def determine_gripper_state(self, hand_landmarks, image_width, image_height, previous_state=None):

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

        return state

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        Each point is a tuple of (x, y, z).
        """
        return math.sqrt((point1[0] - point2[0])**2 +
                         (point1[1] - point2[1])**2 +
                         (point1[2] - point2[2])**2)

    def publish_joint_trajectory(self, hand_label, gripper_state):
        """
        Publish a JointTrajectory message to '/xsubsea/joint_trajectory'
        when the gripper state changes.

        Parameters:
            hand_label (str): Label of the hand ("Left" or "Right").
            gripper_state (str): Current state of the gripper ("Gripper Open" or "Gripper Closed").
        """
        # Check if latest_joint_states has the necessary information
        if not self.latest_joint_states.name:
            self.get_logger().warning('No joint states received yet.')
            return

        # Create JointTrajectory message
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ['axis_a', 'axis_b', 'axis_c', 'axis_d', 'axis_e']

        # Create JointTrajectoryPoint
        point = JointTrajectoryPoint()
        point.positions = [0.0] * len(traj_msg.joint_names)  

        for idx, joint_name in enumerate(traj_msg.joint_names):
            if joint_name in self.latest_joint_states.name:
                joint_idx = self.latest_joint_states.name.index(joint_name)
                point.positions[idx] = self.latest_joint_states.position[joint_idx]
            else:
                point.positions[idx] = 0.0

        if gripper_state == "Gripper Open":
            point.positions[0] = 0.1  # axis_a
        elif gripper_state == "Gripper Closed":
            point.positions[0] = 0.0  # axis_a

        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0

        traj_msg.points.append(point)

        self.joint_trajectory_publisher.publish(traj_msg)
        self.get_logger().info(f'Published JointTrajectory with axis_a: {point.positions[0]}')

    def __del__(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = HandPoseEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
