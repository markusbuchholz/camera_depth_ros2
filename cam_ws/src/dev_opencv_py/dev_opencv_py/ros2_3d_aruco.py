#!/usr/bin/env python3

# https://chev.me/arucogen/

import os
import yaml
import numpy as np
import cv2
import cv2.aruco as aruco

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge, CvBridgeError

class ArucoSubscriber(Node):
    def __init__(self):
        super().__init__('aruco_subscriber')

        # — Load camera calibration from camera_calib.yaml —
        calib_file = os.path.join(os.path.dirname(__file__), 'camera_calib.yaml')
        if not os.path.isfile(calib_file):
            self.get_logger().error(f"Calibration file not found: {calib_file}")
            rclpy.shutdown()
            return
        with open(calib_file, 'r') as f:
            calib = yaml.safe_load(f)

        cm = calib['camera_matrix']['data']
        dc = calib['distortion_coefficients']['data']
        self.camera_matrix = np.array(cm, dtype=np.float64).reshape(3,3)
        self.dist_coeffs   = np.array(dc, dtype=np.float64).reshape(-1,1)
        self.marker_length = float(calib.get('marker_length_m', 0.05))  # metres

        # — ROS interfaces —
        self.bridge = CvBridge()
        self.create_subscription(Image, 'video_frames', self.image_callback, 10)
        self.marker_pub = self.create_publisher(Float64MultiArray, '/marker_localizer', 10)

        # — ArUco setup —
        # Dictionary fallback
        if hasattr(aruco, 'getPredefinedDictionary'):
            self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        else:
            # older API: class name you saw in dir(aruco)
            self.aruco_dict = aruco.Dictionary(aruco.DICT_4X4_250)

        # DetectorParameters fallback
        if hasattr(aruco, 'DetectorParameters_create'):
            self.aruco_params = aruco.DetectorParameters_create()
        else:
            self.aruco_params = aruco.DetectorParameters()

        # ArucoDetector (you have this)
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.get_logger().info(f"OpenCV {cv2.__version__} | Marker = {self.marker_length} m")
        self.get_logger().info("Aruco Subscriber Node started")

    def image_callback(self, msg: Image):
        # Convert ROS image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self.get_logger().info("No ArUco markers detected.")
        else:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # — Pose estimation —
            try:
                # use built-in if available
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners,
                    self.marker_length,
                    self.camera_matrix,
                    self.dist_coeffs
                )
            except AttributeError:
                # manual solvePnP fallback
                objp = np.array([
                    [-self.marker_length/2,  self.marker_length/2, 0.0],
                    [ self.marker_length/2,  self.marker_length/2, 0.0],
                    [ self.marker_length/2, -self.marker_length/2, 0.0],
                    [-self.marker_length/2, -self.marker_length/2, 0.0],
                ], dtype=np.float32)
                rvecs, tvecs = [], []
                for corner in corners:
                    imgp = corner.reshape(4,2).astype(np.float32)
                    _, rvec, tvec = cv2.solvePnP(objp, imgp,
                                                 self.camera_matrix,
                                                 self.dist_coeffs)
                    rvecs.append(rvec)
                    tvecs.append(tvec)
                rvecs = np.array(rvecs)
                tvecs = np.array(tvecs)

            # — Draw & publish —
            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i]
                tvec = tvecs[i]
                x, y, z = tvec.flatten()
                self.get_logger().info(
                    f"ID {marker_id}: X={x:.3f} m  Y={y:.3f} m  Z={z:.3f} m"
                )

                # draw 3D axes
                if hasattr(aruco, 'drawAxis'):
                    aruco.drawAxis(frame,
                                   self.camera_matrix,
                                   self.dist_coeffs,
                                   rvec, tvec,
                                   self.marker_length * 0.5)
                else:
                    # fallback: cv2.drawFrameAxes
                    cv2.drawFrameAxes(frame,
                                      self.camera_matrix,
                                      self.dist_coeffs,
                                      rvec, tvec,
                                      self.marker_length * 0.5)

                # publish full 3D position
                out = Float64MultiArray()
                out.data = [float(x), float(y), float(z)]
                self.marker_pub.publish(out)

        cv2.imshow('Aruco 3D Pose', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
