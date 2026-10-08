#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import csv
import matplotlib.pyplot as plt


class LidarNoiseNode(Node):
    def __init__(self, test_distance, samples_needed=100):
        super().__init__('lidar_noise_node')

        self.sigma_base = 0.01
        self.alpha = 0.02

        self.test_distance = test_distance
        self.samples_needed = samples_needed
        self.results = []

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.publisher = self.create_publisher(
            LaserScan,
            '/scan_noisy',
            10
        )

        self.get_logger().info(
            f'Collecting noisy LiDAR data at distance {self.test_distance} m'
        )

    def scan_callback(self, msg: LaserScan):
        if len(self.results) >= self.samples_needed:
            return

        noisy_msg = LaserScan()
        noisy_msg.header = msg.header
        noisy_msg.angle_min = msg.angle_min
        noisy_msg.angle_max = msg.angle_max
        noisy_msg.angle_increment = msg.angle_increment
        noisy_msg.time_increment = msg.time_increment
        noisy_msg.scan_time = msg.scan_time
        noisy_msg.range_min = msg.range_min
        noisy_msg.range_max = msg.range_max
        noisy_msg.intensities = msg.intensities

        ranges = np.array(msg.ranges, dtype=np.float32)
        noisy_ranges = []

        for d in ranges:
            if np.isfinite(d) and msg.range_min <= d <= msg.range_max:
                sigma_i = self.sigma_base + self.alpha * d
                noise = np.random.normal(0.0, sigma_i)
                d_noisy = d + noise
                d_noisy = np.clip(d_noisy, msg.range_min, msg.range_max)
                noisy_ranges.append(float(d_noisy))
            else:
                noisy_ranges.append(float(d))

        noisy_msg.ranges = noisy_ranges

        self.publisher.publish(noisy_msg)

        angles = msg.angle_min + np.arange(len(noisy_msg.ranges)) * msg.angle_increment
        noisy_ranges_array = np.array(noisy_msg.ranges)

        front_mask = (angles >= -0.0873) & (angles <= 0.0873)
        front_ranges = noisy_ranges_array[front_mask]

        front_ranges = front_ranges[np.isfinite(front_ranges)]
        front_ranges = front_ranges[
            (front_ranges >= msg.range_min) & (front_ranges <= msg.range_max)
        ]

        if len(front_ranges) > 0:
            mean = np.mean(front_ranges)
            var = np.var(front_ranges)

            self.results.append((self.test_distance, mean, var))

            self.get_logger().info(
                f'Distance: {self.test_distance} m, '
                f'Mean: {mean:.3f}, '
                f'Variance: {var:.6f}'
            )


def save_results(all_results, csv_path='/root/tb3_projects_ws/lidar_noise_noisy.csv'):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Distance', 'Mean', 'Variance'])
        writer.writerows(all_results)

    print(f'Results saved to {csv_path}')


def plot_variance(all_results):
    if len(all_results) == 0:
        print("No results to plot!")
        return

    distances = sorted(list(set([r[0] for r in all_results])))
    mean_variances = []

    for d in distances:
        vars_at_d = [r[2] for r in all_results if r[0] == d]
        mean_variances.append(np.mean(vars_at_d))

    plt.plot(distances, mean_variances, marker='o')
    plt.xlabel('Distance to Wall (m)')
    plt.ylabel('Variance of Noisy LiDAR Data (m²)')
    plt.title('Noisy LiDAR Variance vs Distance')
    plt.grid(True)
    plt.show()


def main(args=None):
    rclpy.init(args=args)

    test_distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    samples_needed = 100
    all_results = []

    for dist in test_distances:
        input(f"Place the robot {dist} meters from the wall and press Enter to start...")

        node = LidarNoiseNode(
            test_distance=dist,
            samples_needed=samples_needed
        )

        try:
            while len(node.results) < node.samples_needed:
                rclpy.spin_once(node)
        except KeyboardInterrupt:
            pass
        finally:
            all_results.extend(node.results)
            node.destroy_node()

    save_results(all_results)
    plot_variance(all_results)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
