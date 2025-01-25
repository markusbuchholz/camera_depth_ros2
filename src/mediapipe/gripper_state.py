import cv2
import mediapipe as mp
import math

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    Each point is a tuple of (x, y, z).
    """
    return math.sqrt((point1[0] - point2[0])**2 +
                     (point1[1] - point2[1])**2 +
                     (point1[2] - point2[2])**2)

def determine_gripper_state(hand_landmarks, image_width, image_height, previous_state=None):
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
    reference_distance = calculate_distance(wrist_coords, middle_mcp_coords)
    if reference_distance == 0:
        reference_distance = 1  # Prevent division by zero

    # Get Thumb Tip (landmark 4) and Index Finger Tip (landmark 8)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    thumb_tip_coords = (thumb_tip.x * image_width, thumb_tip.y * image_height, thumb_tip.z)
    index_tip_coords = (index_tip.x * image_width, index_tip.y * image_height, index_tip.z)

    # Calculate distance between Thumb Tip and Index Finger Tip
    distance = calculate_distance(thumb_tip_coords, index_tip_coords)

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

    return state

def main():
    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,       # Video stream
        max_num_hands=2,               # Maximum number of hands to detect
        min_detection_confidence=0.5,  # Minimum confidence for detection
        min_tracking_confidence=0.5    # Minimum confidence for tracking
    )
    mp_draw = mp.solutions.drawing_utils

    # Initialize the webcam.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting hand pose estimation. Press 'q' to exit.")

    # Dictionary to store previous state for each detected hand
    previous_states = {}

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for natural (mirror) viewing.
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands.
        results = hands.process(rgb_frame)

        # Get image dimensions
        image_height, image_width, _ = frame.shape

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks on the frame.
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Get hand label (Left/Right)
                hand_label = handedness.classification[0].label

                # Initialize previous state for this hand if not present
                if hand_label not in previous_states:
                    previous_states[hand_label] = None

                # Determine gripper state
                gripper_state = determine_gripper_state(
                    hand_landmarks, image_width, image_height, previous_state=previous_states[hand_label]
                )

                # Update previous state
                previous_states[hand_label] = gripper_state

                # Print the gripper state to the terminal with hand label
                print(f"{hand_label}: {gripper_state}")

                # Display the gripper state on the video frame
                cv2.putText(frame, f"{hand_label}: {gripper_state}", (10, 30 if hand_label == "Right" else 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        else:
            # Optional: Reset previous states when no hands are detected
            previous_states = {}

        # Display the frame.
        cv2.imshow('Hand Pose Estimation', frame)

        # Exit when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
