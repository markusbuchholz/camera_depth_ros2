#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class CounterClient(Node):
    def __init__(self):
        super().__init__('counter_client')

        # Create a client for the /goal_achived service
        self.client = self.create_client(Trigger, '/goal_achived')
        self.get_logger().info('Client for /goal_achived service created.')

        # Wait until the service is available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /goal_achived service to become available...')

        self.get_logger().info('/goal_achived service is now available.')

        # Initialize counters
        self.active_count = 0
        self.wait_count = 0

        # State flag
        self.is_waiting = False

        # Create a timer that triggers every second
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # Prepare the service request
        req = Trigger.Request()

        # Call the service asynchronously
        future = self.client.call_async(req)

        # Define a callback for the service response
        future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        try:
            response = future.result()
            goal_achived = response.success

            if goal_achived:
                if not self.is_waiting:
                    self.get_logger().info('Goal achieved! Switching to wait mode.')
                self.is_waiting = True
                self.wait_count += 1
                self.get_logger().info(f'wait {self.wait_count}')
            else:
                if self.is_waiting:
                    self.get_logger().info('Goal not achieved. Switching to active mode.')
                self.is_waiting = False
                self.active_count += 1
                self.get_logger().info(f'active {self.active_count}')

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    client = CounterClient()
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        client.get_logger().info('Counter Client Node stopped cleanly')
    finally:
        client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
