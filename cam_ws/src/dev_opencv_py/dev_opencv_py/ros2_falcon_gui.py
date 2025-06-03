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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy


class ArucoSubscriber(Node):
    def __init__(self):
        super().__init__('aruco_subscriber')

        # ────────── Camera calibration ──────────
        calib_file = os.path.join(os.path.dirname(__file__), 'camera_calib.yaml')
        if not os.path.isfile(calib_file):
            self.get_logger().error(f"Calibration file not found: {calib_file}")
            rclpy.shutdown()
            return
        with open(calib_file, 'r') as f:
            calib = yaml.safe_load(f)
        cm = calib['camera_matrix']['data']
        dc = calib['distortion_coefficients']['data']
        self.camera_matrix = np.array(cm, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs   = np.array(dc, dtype=np.float64).reshape(-1, 1)
        self.marker_length = float(calib.get('marker_length_m', 0.05))  # metres

        # ────────── ROS interfaces ──────────
        self.bridge = CvBridge()
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, 'video_frames', self.image_callback, qos)
        self.arm_goal_pub = self.create_publisher(Float64MultiArray, '/arm_goal', 10)

        # ────────── ArUco setup ──────────
        self.aruco_dict = (aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
                           if hasattr(aruco, 'getPredefinedDictionary')
                           else aruco.Dictionary(aruco.DICT_4X4_250))
        self.aruco_params = (aruco.DetectorParameters_create()
                             if hasattr(aruco, 'DetectorParameters_create')
                             else aruco.DetectorParameters())
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # ────────── GUI setup ──────────
        self.window_name = 'Aruco 3D Pose'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)   # fixed size
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # ────────── Interaction state ──────────
        self.dot_pos     = None      # (x, y) in original image coords
        self.rect_origin = None      # top-left of square anchored on the marker
        self.mpp_x = self.mpp_y = None
        self.last_tvec   = None      # [X, Y, Z] from pose estimation
        self.frame_h     = 0         # height of last camera frame

        # ────────── Display constants ──────────
        self.DISPLAY_W = 1200
        self.DISPLAY_H = 800
        self.BUTTON_H  = 80          # button-bar height (px)

        # scale factors (updated every frame)
        self.scale_x = self.scale_y = 1.0
        self.canvas_h = 0

        self.get_logger().info(f"OpenCV {cv2.__version__}")
        self.get_logger().info("Aruco Subscriber Node started")

    # ──────────────────────────────────────────
    #  Image callback
    # ──────────────────────────────────────────
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        h, w = frame.shape[:2]
        self.frame_h = h
        canvas = np.zeros((h + self.BUTTON_H, w, 3), dtype=frame.dtype)
        canvas[0:h] = frame

        # ───── ArUco detect & pose ─────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None and len(ids):
            aruco.drawDetectedMarkers(canvas[0:h], corners, ids)

            # ---------- robust pose estimation ----------
            if hasattr(aruco, 'estimatePoseSingleMarkers'):
                # full contrib build available
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length,
                    self.camera_matrix, self.dist_coeffs)
            else:
                # fallback for stripped builds
                objp = np.array([
                    [-self.marker_length/2,  self.marker_length/2, 0.0],
                    [ self.marker_length/2,  self.marker_length/2, 0.0],
                    [ self.marker_length/2, -self.marker_length/2, 0.0],
                    [-self.marker_length/2, -self.marker_length/2, 0.0],
                ], dtype=np.float32)

                rvecs_list, tvecs_list = [], []
                for c in corners:
                    imgp = c.reshape(4, 2).astype(np.float32)
                    flag = (cv2.SOLVEPNP_IPPE_SQUARE
                            if hasattr(cv2, 'SOLVEPNP_IPPE_SQUARE')
                            else cv2.SOLVEPNP_ITERATIVE)
                    _, rv, tv = cv2.solvePnP(
                        objp, imgp,
                        self.camera_matrix, self.dist_coeffs,
                        flags=flag)
                    rvecs_list.append(rv)
                    tvecs_list.append(tv)
                rvecs = np.array(rvecs_list)
                tvecs = np.array(tvecs_list)
            # ---------- end pose estimation ----------

            # rectangle & scaling
            marker_c = corners[0].reshape(4, 2)
            tl, tr, br, bl = marker_c
            x0, y0 = int(tl[0]), int(tl[1])
            self.rect_origin = (x0, y0)
            pixel_w = np.linalg.norm(tr - tl)
            pixel_h = np.linalg.norm(bl - tl)
            self.mpp_x = self.marker_length / pixel_w
            self.mpp_y = self.marker_length / pixel_h
            self.last_tvec = tvecs[0].flatten()

            # overlays
            cv2.rectangle(canvas[0:h], (x0, y0), (x0 + 300, y0 + 300), (255, 0, 0), 2)
            if hasattr(aruco, 'drawAxis'):
                aruco.drawAxis(canvas[0:h], self.camera_matrix, self.dist_coeffs,
                               rvecs[0], tvecs[0], self.marker_length * 0.5)
            else:  # OpenCV ≥ 5
                cv2.drawFrameAxes(canvas[0:h], self.camera_matrix, self.dist_coeffs,
                                  rvecs[0], tvecs[0], self.marker_length * 0.5)
            if self.dot_pos is not None:
                cv2.circle(canvas[0:h], self.dot_pos, 5, (0, 0, 255), -1)
        else:
            self.rect_origin = self.last_tvec = None

        # ───── Buttons ─────
        send_tl, send_br   = (50,  h + 10), (150, h + 60)
        clear_tl, clear_br = (200, h + 10), (300, h + 60)
        cv2.rectangle(canvas, send_tl,  send_br,  (200, 200, 200), -1)
        cv2.rectangle(canvas, clear_tl, clear_br, (200, 200, 200), -1)
        cv2.putText(canvas, 'Send',  (send_tl[0]  + 10, send_tl[1]  + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(canvas, 'Clear', (clear_tl[0] + 10, clear_tl[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # scale factors (window is fixed to DISPLAY_W × DISPLAY_H)
        self.canvas_h, canvas_w = canvas.shape[:2]
        self.scale_x = canvas_w      / self.DISPLAY_W
        self.scale_y = self.canvas_h / self.DISPLAY_H

        # show frame
        display = cv2.resize(canvas, (self.DISPLAY_W, self.DISPLAY_H),
                             interpolation=cv2.INTER_LINEAR)
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)

    # ──────────────────────────────────────────
    #  Mouse callback
    # ──────────────────────────────────────────
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        fx = int(x * self.scale_x)
        fy = int(y * self.scale_y)

        # Send button
        if 50 <= fx <= 150 and self.frame_h + 10 <= fy <= self.frame_h + 60:
            self.handle_send()
            return
        # Clear button
        if 200 <= fx <= 300 and self.frame_h + 10 <= fy <= self.frame_h + 60:
            self.dot_pos = None
            return
        # Dot placement (inside 300×300 square)
        if fy < self.frame_h and self.rect_origin is not None:
            x0, y0 = self.rect_origin
            if x0 <= fx <= x0 + 300 and y0 <= fy <= y0 + 300:
                self.dot_pos = (fx, fy)

    # ──────────────────────────────────────────
    def handle_send(self):
        if self.dot_pos is None or self.last_tvec is None or self.rect_origin is None:
            self.get_logger().warn("Nothing valid to send.")
            return
        x0, y0 = self.rect_origin
        dx = self.dot_pos[0] - x0
        dy = self.dot_pos[1] - y0
        wx = float(self.last_tvec[0] + dx * self.mpp_x)
        wy = float(self.last_tvec[1] + dy * self.mpp_y)
        wz = float(self.last_tvec[2])

        msg = Float64MultiArray()
        msg.data = [wx, wy, wz]
        self.arm_goal_pub.publish(msg)
        self.get_logger().info(f"Published /arm_goal: [{wx:.3f}, {wy:.3f}, {wz:.3f}]")
        self.dot_pos = None  # reset after send

    # ──────────────────────────────────────────
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


# ──────────────────────────────────────────────
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
