import cv2
import cv2.aruco as aruco

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    arucoParam = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)

    return corners, ids

# Initialize the camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    corners, ids = findArucoMarkers(frame)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
