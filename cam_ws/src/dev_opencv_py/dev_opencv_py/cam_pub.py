# Import the necessary libraries
import rclpy  # Python Client Library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
  
class ImagePublisher(Node):
    """
    Create an ImagePublisher class, which is a subclass of the Node class.
    """
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_publisher')
           
        # Create the publisher. This publisher will publish an Image
        # to the video_frames topic. The queue size is 10 messages.
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
           
        # We will publish a message every 0.1 seconds
        timer_period = 0.1  # seconds
           
        # Create the timer
        self.timer = self.create_timer(timer_period, self.timer_callback)
              
        # Create a VideoCapture object
        # The argument '0' gets the default webcam.
        self.cap = cv2.VideoCapture(0)
              
        # Check if the camera opened successfully
        if not self.cap.isOpened():
            self.get_logger().error('Unable to open the camera.')
            rclpy.shutdown()
       
        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()
        
    def timer_callback(self):
        """
        Callback function.
        This function gets called every 0.1 seconds.
        """
        # Capture frame-by-frame
        ret, frame = self.cap.read()
               
        if ret:
            try:
                # Publish the image.
                # Specify encoding to ensure compatibility
                image_message = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
                self.publisher_.publish(image_message)
                self.get_logger().info('Publishing video frame')
            except Exception as e:
                self.get_logger().error(f'Failed to convert and publish image: {e}')
        else:
            self.get_logger().warning('Failed to capture video frame')
   
    def destroy_node(self):
        """
        Override the destroy_node method to release resources.
        """
        self.cap.release()
        super().destroy_node()
   
def main(args=None):
    """
    Main function to initialize and spin the node.
    """
    # Initialize the rclpy library
    rclpy.init(args=args)
   
    # Create the node
    image_publisher = ImagePublisher()
   
    try:
        # Spin the node so the callback function is called.
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        image_publisher.get_logger().info('Image Publisher node interrupted by user.')
    finally:
        # Destroy the node explicitly
        image_publisher.destroy_node()
        # Shutdown the ROS client library for Python
        rclpy.shutdown()
   
if __name__ == '__main__':
    main()
