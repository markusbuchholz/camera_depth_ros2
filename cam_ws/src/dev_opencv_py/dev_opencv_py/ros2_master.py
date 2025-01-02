#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import threading
import time

class CounterServer(Node):
    def __init__(self):
        super().__init__('counter_server')

        # Create the /goal_achived service
        self.srv = self.create_service(Trigger, '/goal_achived', self.handle_goal_achived)
        self.get_logger().info('Service /goal_achived created and ready.')

        # Initialize the goal_achived state
        self.goal_achived = False
        self.lock = threading.Lock()

        # Start the counting loop in a separate thread
        self.thread = threading.Thread(target=self.counting_loop, daemon=True)
        self.thread.start()

    def handle_goal_achived(self, request, response):
        # Handle service requests by returning the current goal_achived status
        with self.lock:
            response.success = self.goal_achived
            response.message = f'Goal achieved: {self.goal_achived}'
        return response

    def counting_loop(self):
        while rclpy.ok():
            # Phase 1: Count to 10 with "checking_pos X"
            for i in range(10):
                if not rclpy.ok():
                    return
                self.get_logger().info(f'checking_pos {i+1}/10')
                time.sleep(1)

            # After counting to 10, set goal_achived to True
            with self.lock:
                self.goal_achived = True
            self.get_logger().info('Goal achieved set to True')

            # Phase 2: Count to 15 with "watching_rov X"
            for i in range(15):
                if not rclpy.ok():
                    return
                self.get_logger().info(f'watching_rov {i+1}/15')
                time.sleep(1)

            # After counting to 15, set goal_achived to False
            with self.lock:
                self.goal_achived = False
            self.get_logger().info('Goal achieved set to False')

def main(args=None):
    rclpy.init(args=args)
    server = CounterServer()
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Counter Server Node stopped cleanly')
    finally:
        server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
