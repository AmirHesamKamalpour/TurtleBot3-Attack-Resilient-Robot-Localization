#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import csv
import matplotlib.pyplot as plt

class LidarSubscriber(Node):
    def __init__(self, test_distance, samples_needed=100):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.test_distance = test_distance
        self.results = []  # list of tuples: (distance, mean, variance)
        self.samples_needed = samples_needed

    def scan_callback(self, msg: LaserScan):
        if len(self.results) >= self.samples_needed:
            return  # stop collecting after enough samples

        # Extract ±5° front rays
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        front_indices = np.where((angles >= -0.0873) & (angles <= 0.0873))
        front_ranges = np.array(msg.ranges)[front_indices]

        if len(front_ranges) > 0:
            mean = np.mean(front_ranges)
            var = np.var(front_ranges)
            self.get_logger().info(
                f'Distance: {self.test_distance} m, Mean: {mean:.3f}, Variance: {var:.3f}'
            )
            self.results.append((self.test_distance, mean, var))

def save_results(all_results, csv_path='/root/tb3_projects_ws/lidar_noise_all.csv'):
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
    plt.ylabel('Variance of LiDAR Noise (m²)')
    plt.title('LiDAR Noise Variance vs Distance')
    plt.grid(True)
    plt.show()

def main(args=None):
    rclpy.init(args=args)

    test_distances = [0.5, 1.0,1.5, 2.0,2.5, 3.0]
    samples_needed = 100
    all_results = []

    for dist in test_distances:
        input(f"Place the robot {dist} meters from the wall and press Enter to start...")
        node = LidarSubscriber(dist, samples_needed)

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
