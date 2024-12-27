import cv2
import numpy as np

# Initialize global variables
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial mouse coordinates
fx, fy = -1, -1  # Final mouse coordinates
selected = False  # True if a region has been selected
hsv_lower = None
hsv_upper = None
frame_original = None

def mouse_callback(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, selected, hsv_lower, hsv_upper, frame_original

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        selected = True

        # Ensure coordinates are within the frame
        x0, y0 = min(ix, fx), min(iy, fy)
        x1, y1 = max(ix, fx), max(iy, fy)

        # Extract the selected region
        selected_region = frame_original[y0:y1, x0:x1]

        if selected_region.size == 0:
            print("Selected region is empty. Please select a valid area.")
            selected = False
            return

        # Convert to HSV
        hsv_selected = cv2.cvtColor(selected_region, cv2.COLOR_BGR2HSV)

        # Compute the lower and upper bounds using percentiles to avoid outliers
        h_lower, s_lower, v_lower = np.percentile(hsv_selected, 5, axis=(0,1))
        h_upper, s_upper, v_upper = np.percentile(hsv_selected, 95, axis=(0,1))

        # Define tolerance (adjust as needed)
        h_tol = 10
        s_tol = 50
        v_tol = 50

        # Compute lower and upper bounds with clipping
        lower = np.array([
            max(h_lower - h_tol, 0),
            max(s_lower - s_tol, 0),
            max(v_lower - v_tol, 0)
        ], dtype=np.uint8)

        upper = np.array([
            min(h_upper + h_tol, 179),
            min(s_upper + s_tol, 255),
            min(v_upper + v_tol, 255)
        ], dtype=np.uint8)

        hsv_lower = lower
        hsv_upper = upper

        print(f"Selected Region HSV Bounds:")
        print(f"Lower HSV: {hsv_lower}")
        print(f"Upper HSV: {hsv_upper}")

def main():
    global selected, hsv_lower, hsv_upper, frame_original

    # Open the default camera (usually the first camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open the camera.")
        return

    cv2.namedWindow('Camera Feed')

    # Set the mouse callback function
    cv2.setMouseCallback('Camera Feed', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_original = frame.copy()

        if drawing:
            # Draw the current rectangle
            cv2.rectangle(frame, (ix, iy), (fx, fy), (0, 255, 0), 2)

        cv2.imshow('Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Press 'q' to quit
            break
        elif key == ord('c'):
            # Press 'c' to clear the selection
            selected = False
            hsv_lower = None
            hsv_upper = None
            print("Selection cleared.")

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
