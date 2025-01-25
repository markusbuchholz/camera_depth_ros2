import cv2
import mediapipe as mp

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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame.
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # Iterate through each landmark and print its coordinates.
                for idx, lm in enumerate(hand_landmarks.landmark):
                    # Convert normalized coordinates to pixel values.
                    h, w, c = frame.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z

                    # Print the landmark index and its coordinates.
                    print(f"Landmark {idx}: (x={cx}, y={cy}, z={cz:.4f})")

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
