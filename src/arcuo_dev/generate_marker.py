#https://chev.me/arucogen/

import cv2
import cv2.aruco as aruco

def generateArucoMarker(marker_id, marker_size=200, dictionary=aruco.DICT_4X4_250):
    aruco_dict = aruco.getPredefinedDictionary(dictionary)

    marker = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    file_name = f"marker_{marker_id}.png"
    cv2.imwrite(file_name, marker)
    print(f"Marker {marker_id} saved as {file_name}")

# Generate a marker with ID 1 as for OSL
generateArucoMarker(marker_id=1)


"""
>>> import cv2.aruco as aruco
>>> print(dir(aruco))
['ArucoDetector', 'Board', 'CORNER_REFINE_APRILTAG',
'CORNER_REFINE_CONTOUR', 'CORNER_REFINE_NONE', 'CORNER_REFINE_SUBPIX',
'CharucoBoard', 'CharucoDetector', 'CharucoParameters', 'DICT_4X4_100',
'DICT_4X4_1000', 'DICT_4X4_250', 'DICT_4X4_50', 'DICT_5X5_100', 'DICT_5X5_1000',
'DICT_5X5_250', 'DICT_5X5_50', 'DICT_6X6_100', 'DICT_6X6_1000', 'DICT_6X6_250',
'DICT_6X6_50', 'DICT_7X7_100', 'DICT_7X7_1000', 'DICT_7X7_250', 'DICT_7X7_50',
'DICT_APRILTAG_16H5', 'DICT_APRILTAG_16h5', 'DICT_APRILTAG_25H9', 'DICT_APRILTAG_25h9',
'DICT_APRILTAG_36H10', 'DICT_APRILTAG_36H11', 'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11',
'DICT_ARUCO_MIP_36H12', 'DICT_ARUCO_MIP_36h12', 'DICT_ARUCO_ORIGINAL', 'DetectorParameters',
'Dictionary', 'Dictionary_getBitsFromByteList', 'Dictionary_getByteListFromBits',
'GridBoard', 'RefineParameters', '__doc__', '__file__', '__loader__', '__name__',
'__package__', '__path__', '__spec__', '_native', 'drawDetectedCornersCharuco',
'drawDetectedDiamonds', 'drawDetectedMarkers', 'extendDictionary',
'generateImageMarker', 'getPredefinedDictionary']


"""