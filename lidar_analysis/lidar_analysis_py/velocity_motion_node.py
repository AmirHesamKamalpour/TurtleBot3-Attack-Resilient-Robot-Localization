#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class VelocityMotionNode(Node):
    def __init__(self):
        super().__init__('velocity_motion_node')

        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.linear_velocity = 0.2
        self.angular_velocity = 0.5
        self.duration = 5.0

        self.start_time = self.get_clock().now()
        self.timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(
            self.timer_period,
            self.timer_callback
        )

        self.stopped = False

        self.get_logger().info(
            f'Starting motion: v={self.linear_velocity} m/s, '
            f'w={self.angular_velocity} rad/s for {self.duration} seconds'
        )

    def timer_callback(self):
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.start_time).nanoseconds / 1e9

        cmd = Twist()

        if elapsed_time < self.duration:
            cmd.linear.x = self.linear_velocity
            cmd.angular.z = self.angular_velocity
            self.publisher.publish(cmd)

            self.get_logger().info(
                f'Moving... elapsed time: {elapsed_time:.2f} s'
            )

        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.publisher.publish(cmd)

            if not self.stopped:
                self.get_logger().info('Motion finished. Robot stopped.')
                self.stopped = True


def main(args=None):
    rclpy.init(args=args)

    node = VelocityMotionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_cmd = Twist()
        node.publisher.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
