import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open the camera.")
        return

    cv2.namedWindow('Camera Feed')
    show_pixel = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        if show_pixel:
            height, width, _ = frame.shape
            x, y = width // 2, height // 2
            pixel = frame[y, x]
            text = f"BGR: {pixel}"
            cv2.putText(frame, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            show_pixel = not show_pixel  # Toggle display

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
