#!/usr/bin/env python3

import threading
import queue
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

import cv2
import cv2.aruco as aruco


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # Bounded queue (max 2 frames) so we never backlog
        self.frame_queue = queue.Queue(maxsize=2)
        self.bridge = CvBridge()

        # Publisher for (x, y) centers
        self.pub = self.create_publisher(Float64MultiArray, '/marker_localizer', 10)

        # Subscriber only enqueues frames
        self.create_subscription(Image, 'video_frames', self.enqueue_frame, 10)

        # --- Use old-style DetectorParameters() ---
        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters()  # <- old API
        self.detector     = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Start a daemon thread to do all the heavy lifting
        self.worker = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker.start()

        self.get_logger().info('ArucoDetectorNode started with old DetectorParameters().')

    def enqueue_frame(self, msg: Image):
        try:
            # Convert once to BGR8
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge conversion failed: {e}')
            return

        # Drop oldest if queue full, then enqueue new
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def processing_loop(self):
        while rclpy.ok():
            frame = self.frame_queue.get()  # blocks until a frame arrives

            t0 = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            detect_ms = (time.time() - t0) * 1000.0

            if ids is not None and len(ids):
                aruco.drawDetectedMarkers(frame, corners, ids)
                centers = []
                for i, mid in enumerate(ids.flatten()):
                    c = corners[i].reshape((4, 2))
                    cX = float((c[0, 0] + c[2, 0]) * 0.5)
                    cY = float((c[0, 1] + c[2, 1]) * 0.5)
                    centers.append((cX, cY))

                    msg = Float64MultiArray(data=[cX, cY])
                    self.pub.publish(msg)

                self.get_logger().info(
                    f"Detected IDs {ids.flatten().tolist()} "
                    f"in {detect_ms:.1f} ms → centers {centers}"
                )
            else:
                self.get_logger().debug(f"No markers detected ({detect_ms:.1f} ms)")

            # Display (remove if headless)
            cv2.imshow('Aruco', frame)
            cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()

    # Let ROS callbacks and our worker thread truly run in parallel
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down on KeyboardInterrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import threading
import queue
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

import cv2
import cv2.aruco as aruco


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # Bounded queue (max 2 frames) so we never backlog
        self.frame_queue = queue.Queue(maxsize=2)
        self.bridge = CvBridge()

        # Publisher for (x, y) centers
        self.pub = self.create_publisher(Float64MultiArray, '/marker_localizer', 10)

        # Subscriber only enqueues frames
        self.create_subscription(Image, 'video_frames', self.enqueue_frame, 10)

        # --- Use old-style DetectorParameters() ---
        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters()  # <- old API
        self.detector     = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Start a daemon thread to do all the heavy lifting
        self.worker = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker.start()

        self.get_logger().info('ArucoDetectorNode started with old DetectorParameters().')

    def enqueue_frame(self, msg: Image):
        try:
            # Convert once to BGR8
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge conversion failed: {e}')
            return

        # Drop oldest if queue full, then enqueue new
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def processing_loop(self):
        while rclpy.ok():
            frame = self.frame_queue.get()  # blocks until a frame arrives

            t0 = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            detect_ms = (time.time() - t0) * 1000.0

            if ids is not None and len(ids):
                aruco.drawDetectedMarkers(frame, corners, ids)
                centers = []
                for i, mid in enumerate(ids.flatten()):
                    c = corners[i].reshape((4, 2))
                    cX = float((c[0, 0] + c[2, 0]) * 0.5)
                    cY = float((c[0, 1] + c[2, 1]) * 0.5)
                    centers.append((cX, cY))

                    msg = Float64MultiArray(data=[cX, cY])
                    self.pub.publish(msg)

                self.get_logger().info(
                    f"Detected IDs {ids.flatten().tolist()} "
                    f"in {detect_ms:.1f} ms → centers {centers}"
                )
            else:
                self.get_logger().debug(f"No markers detected ({detect_ms:.1f} ms)")

            # Display (remove if headless)
            cv2.imshow('Aruco', frame)
            cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()

    # Let ROS callbacks and our worker thread truly run in parallel
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down on KeyboardInterrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
