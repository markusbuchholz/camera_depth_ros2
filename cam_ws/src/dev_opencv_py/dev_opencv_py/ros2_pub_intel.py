#!/usr/bin/env python3

"""
running on the bottle - Jetson
python3 cam_pub_faster.py --camera 6    # or /dev/video6, etc.
"""

import sys, argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FastImagePublisher(Node):
    def __init__(self, source):
        super().__init__('fast_image_publisher')
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(Image, 'video_frames', qos)

        # Open specified index or device path, force V4L2
        self.cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f'Unable to open camera source: {source}')
            sys.exit(1)

        # --- configure for YUYV 640×480@30fps ---
        fourcc = cv2.VideoWriter_fourcc(*'YUY2')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        # drop stale frames for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # ---------------------------------------

        self.br = CvBridge()
        self.get_logger().info(f'Publishing from camera source: {source}')

    def run(self):
        while rclpy.ok() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning('Frame capture failed')
                continue

            # convert YUYV→BGR if needed (gray or single-channel)
            if frame.ndim == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

            # publish as bgr8
            msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', required=True,
                        help='Camera index (e.g. 6) or device path (e.g. /dev/video6)')
    args = parser.parse_args()

    # allow numeric indices
    try:
        source = int(args.camera)
    except ValueError:
        source = args.camera

    rclpy.init()
    node = FastImagePublisher(source)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
