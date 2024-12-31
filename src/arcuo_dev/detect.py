import cv2
import cv2.aruco as aruco

# Load the uploaded image
image_path = "marker_1.png"  # Replace with the path to your uploaded image
image = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# List of possible dictionaries
dictionaries = [
    aruco.DICT_4X4_50, aruco.DICT_4X4_100, aruco.DICT_4X4_250, aruco.DICT_4X4_1000,
    aruco.DICT_5X5_50, aruco.DICT_5X5_100, aruco.DICT_5X5_250, aruco.DICT_5X5_1000,
    aruco.DICT_6X6_50, aruco.DICT_6X6_100, aruco.DICT_6X6_250, aruco.DICT_6X6_1000,
    aruco.DICT_7X7_50, aruco.DICT_7X7_100, aruco.DICT_7X7_250, aruco.DICT_7X7_1000,
]

for dictionary_id in dictionaries:
    aruco_dict = aruco.getPredefinedDictionary(dictionary_id)
    aruco_detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    if ids is not None and len(ids) > 0:
        print(f"Detected markers with dictionary {dictionary_id}: {ids.flatten()}")
        break
else:
    print("No markers detected with any dictionary.")
