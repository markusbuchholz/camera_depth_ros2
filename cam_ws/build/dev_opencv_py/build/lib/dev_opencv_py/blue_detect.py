import cv2

THRESHOLD_LOW = (110, 50, 50)
THRESHOLD_HIGH = (130, 255, 255)

# Webcam parameters (your desired resolution)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# Minimum required radius of enclosing circle of contour
MIN_RADIUS = 2

# Initialize camera and get actual resolution
cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Main loop
while True:
    # Get image from camera
    ret_val, img = cam.read()
    if not ret_val:
        print("Failed to grab frame")
        break
    img = cv2.resize(img, (600, 480))

    # Blur image to remove noise
    img_filter = cv2.GaussianBlur(img.copy(), (3, 3), 0)

    # Convert image from BGR to HSV
    img_hsv = cv2.cvtColor(img_filter, cv2.COLOR_BGR2HSV)

    # Set pixels to white if in color range, others to black (binary bitmap)
    img_binary = cv2.inRange(img_hsv, THRESHOLD_LOW, THRESHOLD_HIGH)

    # Dilate image to make white blobs larger
    img_binary = cv2.dilate(img_binary, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour and use it to compute the min enclosing circle
    center = None
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius >= MIN_RADIUS:
                # Draw a green circle around the largest enclosed contour
                cv2.circle(img, center, int(radius), (0, 255, 0), 2)
                cv2.putText(img, f"Center: {center}", (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display images
    cv2.imshow("Real-time Feed", img)
    cv2.imshow("Binary Image", img_binary)

    # Exit on ESC press
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
