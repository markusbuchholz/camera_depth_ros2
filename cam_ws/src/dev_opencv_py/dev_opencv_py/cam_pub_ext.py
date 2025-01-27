import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge  
import cv2  
import os  

class ImagePublisher(Node):
    """
    Create an ImagePublisher class, which is a subclass of the Node class.
    """
    def __init__(self):
        """
        Class constructor to set up the node
        """
        super().__init__('image_publisher')
           
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
           
        timer_period = 0.1  # seconds
           
        self.timer = self.create_timer(timer_period, self.timer_callback)
              
        camera_device = os.getenv('CAMERA_DEVICE', '/dev/video2')
        self.get_logger().info(f'Attempting to open camera at {camera_device}')
        
        self.cap = cv2.VideoCapture(camera_device)
              
        if not self.cap.isOpened():
            self.get_logger().error(f'Unable to open the camera at {camera_device}. Shutting down node.')
            rclpy.shutdown()
       
        self.br = CvBridge()
        
    def timer_callback(self):

        ret, frame = self.cap.read()
               
        if ret:
            try:
                image_message = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher_.publish(image_message)
                self.get_logger().info('Publishing video frame')
            except Exception as e:
                self.get_logger().error(f'Failed to convert and publish image: {e}')
        else:
            self.get_logger().warning('Failed to capture video frame')
   
    def destroy_node(self):

        self.cap.release()
        super().destroy_node()
   
def main(args=None):

    rclpy.init(args=args)
    
    image_publisher = None
    try:
        image_publisher = ImagePublisher()
        
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        if image_publisher is not None:
            image_publisher.get_logger().info('Image Publisher node interrupted by user.')
    except Exception as e:
        if image_publisher is not None:
            image_publisher.get_logger().error(f'An unexpected error occurred: {e}')
    finally:
        if image_publisher is not None:
            image_publisher.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
   
if __name__ == '__main__':
    main()
