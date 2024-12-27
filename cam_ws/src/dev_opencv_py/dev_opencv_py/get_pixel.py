import cv2
import numpy as np

# Mouse callback function to get pixel values
def get_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        img = param['image']
        if y < img.shape[0] and x < img.shape[1]:
            bgr_pixel = img[y, x]
            hsv_pixel = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
            print(f"Pixel at ({x}, {y}): BGR={bgr_pixel}, HSV={hsv_pixel}")
            # Store the point and its values
            param['points'].append({'x': x, 'y': y, 'bgr': bgr_pixel, 'hsv': hsv_pixel})
        else:
            print("Clicked outside the image boundaries.")

def display_values_on_image(frame, points):
    """
    Draws rectangles and displays BGR and HSV values for all clicked points.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White text
    thickness = 1
    line_type = cv2.LINE_AA

    for point in points:
        x, y = point['x'], point['y']
        bgr = point['bgr']
        hsv = point['hsv']
        
        bgr_text = f"BGR: {bgr}"
        hsv_text = f"HSV: {hsv}"
        
        text_x = x + 10 if x + 10 < frame.shape[1] - 200 else x - 200
        text_y = y - 30 if y - 30 > 30 else y + 30
        
        (bgr_text_width, bgr_text_height), _ = cv2.getTextSize(bgr_text, font, font_scale, thickness)
        (hsv_text_width, hsv_text_height), _ = cv2.getTextSize(hsv_text, font, font_scale, thickness)
        
        cv2.rectangle(frame, 
                      (text_x, text_y - bgr_text_height - 10), 
                      (text_x + max(bgr_text_width, hsv_text_width) + 10, text_y + hsv_text_height + 10), 
                      (0, 0, 0), 
                      -1)
        
        cv2.putText(frame, bgr_text, 
                    (text_x, text_y - 5), 
                    font, font_scale, font_color, thickness, line_type)
        cv2.putText(frame, hsv_text, 
                    (text_x, text_y + hsv_text_height + 5), 
                    font, font_scale, font_color, thickness, line_type)
        
        # Optionally, draw a small circle at the clicked point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open the camera.")
        return

    cv2.namedWindow('Camera Feed')

    param = {'image': None, 'frame': None, 'points': []}

    cv2.setMouseCallback('Camera Feed', get_pixel_value, param)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        param['image'] = frame.copy()
        param['frame'] = frame.copy()

        # Display all clicked points
        display_values_on_image(param['frame'], param['points'])

        cv2.imshow('Camera Feed', param['frame'])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit the application
            break
        elif key == ord('c'):
            # Clear all points
            param['points'].clear()
            print("Cleared all stored points.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
