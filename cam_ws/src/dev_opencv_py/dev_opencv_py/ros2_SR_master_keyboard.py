#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import threading
import sys
import termios
import tty
import select

class MasterNode(Node):
    def __init__(self):
        super().__init__('master_node')

        # Initialize goal_achived state
        self.goal_achived = False
        self.goal_lock = threading.Lock()

        # Initialize counts
        self.active_count = 0
        self.wait_count = 0

        # Create the /goal_achived service
        self.srv = self.create_service(Trigger, '/goal_achived', self.handle_goal_achived)
        self.get_logger().info('Service /goal_achived created and ready.')

        # Create a timer to handle counting every second
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Start the keyboard listener in a separate thread
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.keyboard_thread.start()

    def handle_goal_achived(self, request, response):
        with self.goal_lock:
            response.success = self.goal_achived
            response.message = f'Goal achieved: {self.goal_achived}'
        return response

    def timer_callback(self):
        with self.goal_lock:
            if self.goal_achived:
                self.wait_count += 1
                self.get_logger().info(f'wait {self.wait_count}')
            else:
                self.active_count += 1
                self.get_logger().info(f'active {self.active_count}')

    def keyboard_listener(self):
        """
        Listens for keyboard inputs:
        - Press 'S' or 's' to set /goal_achived to True.
        - Press 'R' or 'r' to set /goal_achived to False.
        - Press 'Q' or 'q' to quit the program.
        """
        # Set terminal to raw mode to capture key presses without Enter
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            self.get_logger().info("Press 'R' to set /goal_achived to True, 'S' to set to False, 'Q' to quit.")
            while rclpy.ok():
                dr, dw, de = select.select([sys.stdin], [], [], 0.1)
                if dr:
                    key = sys.stdin.read(1)
                    key = key.upper()
                    if key == 'R':
                        with self.goal_lock:
                            if not self.goal_achived:
                                self.goal_achived = True
                                self.wait_count = 0  # Reset wait count
                                self.get_logger().info("/goal_achived set to True")
                            else:
                                self.get_logger().info("/goal_achived is already True")
                    elif key == 'S':
                        with self.goal_lock:
                            if self.goal_achived:
                                self.goal_achived = False
                                self.active_count = 0  # Reset active count
                                self.get_logger().info("/goal_achived set to False")
                            else:
                                self.get_logger().info("/goal_achived is already False")
                    elif key == 'Q':
                        self.get_logger().info("Quitting Master Node.")
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def main(args=None):
    rclpy.init(args=args)
    master_node = MasterNode()
    try:
        rclpy.spin(master_node)
    except KeyboardInterrupt:
        master_node.get_logger().info('Master Node stopped cleanly')
    finally:
        master_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
