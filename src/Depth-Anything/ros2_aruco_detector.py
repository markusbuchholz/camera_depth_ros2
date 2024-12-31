#!/usr/bin/env python3

#https://chev.me/arucogen/

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray  
from cv_bridge import CvBridge, CvBridgeError
import cv2
import cv2.aruco as aruco

class ArucoSubscriber(Node):
    def __init__(self):
        super().__init__('aruco_subscriber')
        
        # Create a subscription to the 'video_frames' topic
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.image_callback,
            10  # QoS history depth
        )
        self.subscription  # prevent unused variable warning

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize the publisher for marker locations
        self.marker_publisher = self.create_publisher(Float64MultiArray, '/marker_localizer', 10)
        
        # Debug: Print OpenCV version
        self.get_logger().info(f"OpenCV Version: {cv2.__version__}")

        # Check if aruco module exists
        if hasattr(cv2, 'aruco'):
            self.get_logger().info("cv2.aruco module is available.")
            # List all attributes in aruco
            aruco_attrs = dir(aruco)
            self.get_logger().info(f"cv2.aruco attributes: {aruco_attrs}")

            # Check for ArucoDetector
            if hasattr(aruco, 'ArucoDetector'):
                self.get_logger().info("ArucoDetector class is available.")
            else:
                self.get_logger().error("ArucoDetector class is NOT available. Check OpenCV installation.")
                rclpy.shutdown()
                return
        else:
            self.get_logger().error("cv2.aruco module is NOT available.")
            rclpy.shutdown()
            return

        # Set up ArUco dictionary and parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters()

        # Initialize the ArucoDetector
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.get_logger().info('Aruco Subscriber Node has been started.')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # Detect ArUco markers in the image using ArucoDetector
        try:
            corners, ids, rejected = self.detector.detectMarkers(cv_image)
        except Exception as e:
            self.get_logger().error(f'ArUco Detection Error: {e}')
            return

        # If markers are detected, draw them on the image
        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(cv_image, corners, ids)
            self.get_logger().info(f'Detected ArUco markers with IDs: {ids.flatten()}')

            # Iterate through each detected marker
            for i, marker_id in enumerate(ids.flatten()):
                # Get the corners of the marker
                marker_corners = corners[i].reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = marker_corners

                # Calculate the center of the marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                # Print the location on the terminal
                self.get_logger().info(f"Marker ID {marker_id} - Center: (X: {cX}, Y: {cY})")

                # Create a Float64MultiArray message
                location_msg = Float64MultiArray()
                location_msg.data = [float(cX), float(cY)]
                
                # Publish the location
                self.marker_publisher.publish(location_msg)
        else:
            self.get_logger().info('No ArUco markers detected.')

        # Display the image with detected markers
        cv2.imshow('Aruco Detection', cv_image)
        cv2.waitKey(1)  # Necessary for OpenCV to display the image

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    aruco_subscriber = ArucoSubscriber()

    try:
        rclpy.spin(aruco_subscriber)
    except KeyboardInterrupt:
        aruco_subscriber.get_logger().info('Aruco Subscriber node stopped cleanly.')
    except Exception as e:
        aruco_subscriber.get_logger().error(f'Exception in Aruco Subscriber node: {e}')
    finally:
        aruco_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
