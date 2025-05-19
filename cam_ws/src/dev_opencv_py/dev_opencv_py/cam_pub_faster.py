import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class FastImagePublisher(Node):
    def __init__(self):
        super().__init__('fast_image_publisher')

        # Use a best-effort QoS for lower latency
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.publisher_ = self.create_publisher(Image, 'video_frames', qos)

        # Initialize VideoCapture with default camera. Use any available backend
        self.cap = cv2.VideoCapture(0, cv2.CAP_ANY)

        if not self.cap.isOpened():
            self.get_logger().error('Unable to open camera')
            rclpy.shutdown()

        # Reduce internal buffer to drop stale frames
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Optionally reduce resolution for higher fps
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.br = CvBridge()

    def run(self):
        # Main loop: capture and publish as fast as possible
        while rclpy.ok() and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning('Frame capture failed')
                continue

            # Convert and publish
            try:
                msg = self.br.cv2_to_imgmsg(frame, encoding='bgr8')
                self.publisher_.publish(msg)
            except Exception as e:
                self.get_logger().error(f'Publish failed: {e}')

            # Allow ROS to process internal callbacks
            rclpy.spin_once(self, timeout_sec=0)

    def destroy_node(self):
        # Cleanup
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FastImagePublisher()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
