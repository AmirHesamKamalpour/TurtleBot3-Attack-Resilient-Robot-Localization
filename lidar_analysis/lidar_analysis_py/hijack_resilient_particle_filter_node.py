import csv
import os
import time
from datetime import datetime
from pathlib import Path as FsPath

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, Path as NavPath
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64
from visualization_msgs.msg import Marker, MarkerArray

from .pf.map_utils import load_occupancy_map
from .pf.motion_model import odometry_delta, apply_odometry_motion, wrap_angle
from .pf.observation_model import laser_log_likelihood
from .pf.resampling import normalize_log_weights, effective_sample_size, systematic_resample
from .pf.metrics import weighted_pose_mean, weighted_pose_spread, pose_error


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)


def odom_msg_to_pose(msg: Odometry):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return np.array([p.x, p.y, quaternion_to_yaw(q)], dtype=np.float64)


def transform_truth_pose_to_map(raw_pose, x_offset, y_offset, yaw_offset):
    """Convert Gazebo/world truth pose into the PF map frame.

    x_map = R(yaw_offset) * x_truth + translation
    yaw_map = yaw_truth + yaw_offset
    """
    c = np.cos(yaw_offset)
    ss = np.sin(yaw_offset)

    x = c * raw_pose[0] - ss * raw_pose[1] + x_offset
    y = ss * raw_pose[0] + c * raw_pose[1] + y_offset
    yaw = wrap_angle(raw_pose[2] + yaw_offset)

    return np.array([x, y, yaw], dtype=np.float64)


def pose_from_xyyaw(x, y, yaw):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = 0.0
    pose.orientation.z = float(np.sin(yaw / 2.0))
    pose.orientation.w = float(np.cos(yaw / 2.0))
    return pose


class HijackResilientParticleFilterNode(Node):
    def __init__(self):
        super().__init__("hijack_resilient_particle_filter_localization")

        default_map = str(FsPath.home() / "tb3_projects_ws/maps/explore_house.yaml")
        default_log_dir = str(FsPath.home() / "tb3_projects_ws/log/pf_hijack_runs")

        self.declare_parameter("map_yaml_path", default_map)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_topic", "/odom_manipulated")
        self.declare_parameter("scan_topic", "/scan_manipulated")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("use_odom_as_truth", False)
        self.declare_parameter("true_odom_topic", "/ground_truth_odom")

        # Transform raw ground-truth odometry into map coordinates.
        # Useful when Gazebo publishes truth in world frame but PF estimates in map frame.
        self.declare_parameter("truth_x_offset", 2.229376)
        self.declare_parameter("truth_y_offset", 0.415668)
        self.declare_parameter("truth_yaw_offset", -0.016549)
        self.declare_parameter("enable_truth_calibration_from_initialpose", True)
        self.declare_parameter("truth_calibration_topic", "/initialpose")

        self.declare_parameter("n_particles", 1500)
        self.declare_parameter("seed", -1)

        self.declare_parameter("alpha1", 0.02)
        self.declare_parameter("alpha2", 0.02)
        self.declare_parameter("alpha3", 0.04)
        self.declare_parameter("alpha4", 0.02)

        self.declare_parameter("beam_step", 8)
        self.declare_parameter("max_beams", 80)
        self.declare_parameter("sigma_hit", 0.15)
        self.declare_parameter("z_hit", 0.95)
        self.declare_parameter("z_rand", 0.05)
        self.declare_parameter("max_range", 3.5)
        self.declare_parameter("max_likelihood_dist_m", 2.0)

        self.declare_parameter("resample_neff_fraction", 0.5)
        self.declare_parameter("roughening_xy_std", 0.003)
        self.declare_parameter("roughening_yaw_std", 0.003)

        # MCMC Metropolis-Hastings move after resampling.
        self.declare_parameter("mcmc_enabled", True)
        self.declare_parameter("mcmc_iterations", 2)
        self.declare_parameter("mcmc_proposal_xy_std", 0.04)
        self.declare_parameter("mcmc_proposal_yaw_std", 0.08)
        self.declare_parameter("mcmc_likelihood_scale", 10.0)
        self.declare_parameter("mcmc_min_clearance_m", 0.05)

        self.declare_parameter("converged_std_xy", 0.08)
        self.declare_parameter("converged_std_yaw", 0.10)
        self.declare_parameter("converged_required_updates", 30)

        # Prevent false convergence during simulation.
        self.declare_parameter("min_scan_updates_before_convergence", 80)
        self.declare_parameter("converged_position_error", 0.35)
        self.declare_parameter("converged_yaw_error", 0.35)

        self.declare_parameter("max_path_length", 5000)
        self.declare_parameter("path_publish_min_distance", 0.02)

        self.declare_parameter("log_dir", default_log_dir)

        # Hijacking assignment parameters.  The PF must only use manipulated data.
        self.declare_parameter("enable_attack_detection", True)
        self.declare_parameter("detection_requires_initial_convergence", True)
        self.declare_parameter("stop_robot_on_convergence", False)
        self.declare_parameter("require_truth_for_convergence", False)

        self.declare_parameter("blind_valid_fraction_threshold", 0.08)
        self.declare_parameter("scan_recovered_valid_fraction", 0.35)
        self.declare_parameter("scan_frozen_delta_threshold", 0.0008) # 0.006
        self.declare_parameter("scan_changed_delta_threshold", 0.0275) # 0.035
        self.declare_parameter("scan_frozen_required_updates", 3) # 3
        self.declare_parameter("scan_changed_required_updates", 2)

        self.declare_parameter("odom_frozen_trans_threshold", 0.004)
        self.declare_parameter("odom_frozen_yaw_threshold", 0.004)
        self.declare_parameter("odom_frozen_required_updates", 3)
        self.declare_parameter("cmd_motion_linear_threshold", 0.015)
        self.declare_parameter("cmd_motion_angular_threshold", 0.05)
        self.declare_parameter("cmd_recent_timeout_sec", 1.2)

        self.declare_parameter("mode1_local_radius_m", 1.10) #0.45
        self.declare_parameter("mode2_local_radius_m", 1.10)
        self.declare_parameter("local_radius_per_cmd_meter", 0.35)
        self.declare_parameter("max_local_recovery_radius_m", 3.5)
        self.declare_parameter("mode1_local_yaw_std", 0.40)
        self.declare_parameter("mode2_local_yaw_std", 1.20)
        self.declare_parameter("local_recovery_candidate_factor", 8)
        self.declare_parameter("local_recovery_likelihood_scale", 14.0)

        self.map_yaml_path = self.get_parameter("map_yaml_path").value
        self.map_frame = self.get_parameter("map_frame").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.use_odom_as_truth = bool(self.get_parameter("use_odom_as_truth").value)
        self.true_odom_topic = self.get_parameter("true_odom_topic").value

        self.truth_x_offset = float(self.get_parameter("truth_x_offset").value)
        self.truth_y_offset = float(self.get_parameter("truth_y_offset").value)
        self.truth_yaw_offset = float(self.get_parameter("truth_yaw_offset").value)
        self.enable_truth_calibration_from_initialpose = bool(
            self.get_parameter("enable_truth_calibration_from_initialpose").value
        )
        self.truth_calibration_topic = self.get_parameter("truth_calibration_topic").value

        self.n_particles = int(self.get_parameter("n_particles").value)
        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.alphas = np.array(
            [
                float(self.get_parameter("alpha1").value),
                float(self.get_parameter("alpha2").value),
                float(self.get_parameter("alpha3").value),
                float(self.get_parameter("alpha4").value),
            ],
            dtype=np.float64,
        )

        max_likelihood_dist_m = float(self.get_parameter("max_likelihood_dist_m").value)
        self.map = load_occupancy_map(
            self.map_yaml_path,
            max_likelihood_dist_m=max_likelihood_dist_m,
        )

        self.particles = self.map.sample_free_particles(self.n_particles, self.rng)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / float(self.n_particles)

        self.prev_odom_pose = None
        self.raw_true_pose = None
        self.true_pose = None
        self.latest_estimate = None
        self.latest_spread = None
        self.latest_neff = float(self.n_particles)
        self.latest_processing_time_ms = 0.0
        self.latest_mcmc_acceptance_rate = 0.0

        self.converged_counter = 0
        self.converged = False
        self.scan_update_count = 0
        self.latest_convergence_progress = 0.0
        self.latest_scan_match_score = float('nan')

        # Hijacking detector state.  None of this subscribes to /actual_mode; that
        # topic is referee truth and must not be used by the student algorithm.
        self.initial_converged_once = False
        self.attack_active = False
        self.detected_mode = 0
        self.attack_detection_time = None
        self.recovery_start_time = None
        self.latest_recovery_time_sec = float("nan")
        self.recovery_strategy = "none"
        self.global_recovery_started = False

        self.latest_cmd_linear = 0.0
        self.latest_cmd_angular = 0.0
        self.latest_cmd_time = None
        self.attack_cmd_distance = 0.0
        self.last_attack_integral_time = None

        self.latest_odom_delta_m = 0.0
        self.latest_odom_delta_yaw = 0.0
        self.odom_frozen_count = 0

        self.last_scan_signature = None
        self.latest_scan_valid_fraction = 1.0
        self.latest_scan_delta = float("inf")
        self.scan_frozen_count = 0
        self.scan_changed_count = 0
        self.total_scan_callbacks = 0

        self.estimated_path = NavPath()
        self.true_path = NavPath()
        self.estimated_path.header.frame_id = self.map_frame
        self.true_path.header.frame_id = self.map_frame
        self.last_estimated_path_pose = None
        self.last_true_path_pose = None

        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = ReliabilityPolicy.RELIABLE

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.particle_pose_pub = self.create_publisher(PoseArray, "/mcmc_pf/particle_poses", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/mcmc_pf/markers", 10)
        self.estimated_pose_pub = self.create_publisher(PoseStamped, "/mcmc_pf/estimated_pose", 10)
        self.true_pose_pub = self.create_publisher(PoseStamped, "/mcmc_pf/true_pose", 10)
        self.estimated_path_pub = self.create_publisher(NavPath, "/mcmc_pf/estimated_path", 10)
        self.true_path_pub = self.create_publisher(NavPath, "/mcmc_pf/true_path", 10)

        self.position_error_pub = self.create_publisher(Float64, "/mcmc_pf/error/position", 10)
        self.yaw_error_pub = self.create_publisher(Float64, "/mcmc_pf/error/yaw", 10)
        self.neff_pub = self.create_publisher(Float64, "/mcmc_pf/neff", 10)
        self.processing_time_pub = self.create_publisher(Float64, "/mcmc_pf/processing_time_ms", 10)
        self.mcmc_acceptance_pub = self.create_publisher(Float64, "/mcmc_pf/mcmc_acceptance_rate", 10)
        self.converged_pub = self.create_publisher(Bool, "/mcmc_pf/converged", 10)
        self.convergence_progress_pub = self.create_publisher(Float64, "/mcmc_pf/convergence_progress", 10)
        self.scan_match_score_pub = self.create_publisher(Float64, "/mcmc_pf/scan_match_score", 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10,
        )

        if not self.use_odom_as_truth:
            self.true_odom_sub = self.create_subscription(
                Odometry,
                self.true_odom_topic,
                self.true_odom_callback,
                10,
            )

            if self.enable_truth_calibration_from_initialpose:
                self.initialpose_sub = self.create_subscription(
                    PoseWithCovarianceStamped,
                    self.truth_calibration_topic,
                    self.initialpose_callback,
                    10,
                )
                self.get_logger().info(
                    f"Truth calibration enabled. Use RViz 2D Pose Estimate on "
                    f"{self.truth_calibration_topic} to align ground truth to map."
                )

        self.map_timer = self.create_timer(2.0, self.publish_map)

        self.csv_file, self.csv_writer = self.open_log_file()

        self.get_logger().info(f"Loaded map: {self.map_yaml_path}")
        self.get_logger().info(f"Map image: {self.map.image_path}")
        self.get_logger().info(f"Particles: {self.n_particles}")
        self.get_logger().info(f"Subscribing to odom: {self.odom_topic}")
        self.get_logger().info(f"Subscribing to scan: {self.scan_topic}")
        self.get_logger().info("Hijack-resilient MCMC particle filter node started.")

    def open_log_file(self):
        log_dir = os.path.expanduser(self.get_parameter("log_dir").value)
        os.makedirs(log_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"mcmc_pf_run_{stamp}.csv")

        f = open(path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_sec",
                "x_est",
                "y_est",
                "yaw_est",
                "x_true",
                "y_true",
                "yaw_true",
                "position_error",
                "yaw_error",
                "neff",
                "std_x",
                "std_y",
                "std_yaw",
                "converged",
                "processing_time_ms",
                "convergence_progress",
                "scan_match_score",
                "mcmc_acceptance_rate",
                "detected_mode",
                "attack_active",
                "recovery_strategy",
                "recovery_time_sec",
                "scan_valid_fraction",
                "scan_delta",
                "odom_frozen_count",
            ]
        )

        self.get_logger().info(f"Logging PF metrics to: {path}")
        return f, writer

    def publish_map(self):
        msg = self.map.to_occupancy_grid_msg(
            self.get_clock().now().to_msg(),
            self.map_frame,
        )
        self.map_pub.publish(msg)

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def cmd_vel_callback(self, msg: Twist):
        self.latest_cmd_linear = float(msg.linear.x)
        self.latest_cmd_angular = float(msg.angular.z)
        self.latest_cmd_time = self.now_sec()

    def command_motion_expected(self):
        if self.latest_cmd_time is None:
            return False

        if self.now_sec() - self.latest_cmd_time > float(self.get_parameter("cmd_recent_timeout_sec").value):
            return False

        lin = abs(self.latest_cmd_linear)
        ang = abs(self.latest_cmd_angular)
        return (
            lin > float(self.get_parameter("cmd_motion_linear_threshold").value)
            or ang > float(self.get_parameter("cmd_motion_angular_threshold").value)
        )

    def integrate_attack_command_distance(self):
        now = self.now_sec()
        if self.last_attack_integral_time is None:
            self.last_attack_integral_time = now
            return

        dt = max(0.0, min(0.5, now - self.last_attack_integral_time))
        self.last_attack_integral_time = now

        if self.attack_active:
            self.attack_cmd_distance += abs(float(self.latest_cmd_linear)) * dt

    def odom_callback(self, msg: Odometry):
        curr_pose = odom_msg_to_pose(msg)

        if self.prev_odom_pose is None:
            self.prev_odom_pose = curr_pose
            if self.use_odom_as_truth:
                self.true_pose = curr_pose
            return

        delta = odometry_delta(self.prev_odom_pose, curr_pose)
        self.latest_odom_delta_m = abs(float(delta[1]))
        self.latest_odom_delta_yaw = abs(float(delta[0])) + abs(float(delta[2]))

        odom_trans_thresh = float(self.get_parameter("odom_frozen_trans_threshold").value)
        odom_yaw_thresh = float(self.get_parameter("odom_frozen_yaw_threshold").value)

        if (
            self.command_motion_expected()
            and self.latest_odom_delta_m < odom_trans_thresh
            and self.latest_odom_delta_yaw < odom_yaw_thresh
        ):
            self.odom_frozen_count += 1
        else:
            self.odom_frozen_count = 0

        self.particles = apply_odometry_motion(
            self.particles,
            delta,
            self.alphas,
            self.rng,
        )

        self.prev_odom_pose = curr_pose

        if self.use_odom_as_truth:
            self.true_pose = curr_pose

    def true_odom_callback(self, msg: Odometry):
        raw_pose = odom_msg_to_pose(msg)
        self.raw_true_pose = raw_pose
        self.true_pose = transform_truth_pose_to_map(
            raw_pose,
            self.truth_x_offset,
            self.truth_y_offset,
            self.truth_yaw_offset,
        )

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        if self.raw_true_pose is None:
            self.get_logger().warn(
                "Received /initialpose, but no /ground_truth_odom sample has arrived yet."
            )
            return

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        clicked_pose = np.array(
            [p.x, p.y, quaternion_to_yaw(q)],
            dtype=np.float64,
        )

        raw = self.raw_true_pose

        yaw_offset = wrap_angle(clicked_pose[2] - raw[2])
        c = np.cos(yaw_offset)
        ss = np.sin(yaw_offset)

        x_rot = c * raw[0] - ss * raw[1]
        y_rot = ss * raw[0] + c * raw[1]

        x_offset = clicked_pose[0] - x_rot
        y_offset = clicked_pose[1] - y_rot

        self.truth_x_offset = float(x_offset)
        self.truth_y_offset = float(y_offset)
        self.truth_yaw_offset = float(yaw_offset)

        self.true_pose = transform_truth_pose_to_map(
            raw,
            self.truth_x_offset,
            self.truth_y_offset,
            self.truth_yaw_offset,
        )

        self.get_logger().info(
            "Calibrated ground truth -> map transform from RViz /initialpose:\n"
            f"  truth_x_offset   := {self.truth_x_offset:.6f}\n"
            f"  truth_y_offset   := {self.truth_y_offset:.6f}\n"
            f"  truth_yaw_offset := {self.truth_yaw_offset:.6f}"
        )


    def particle_validity_mask(self, particles):
        rows, cols, valid = self.map.world_to_map(particles[:, 0], particles[:, 1])

        legal = np.zeros(particles.shape[0], dtype=bool)
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            return legal

        r = rows[valid_idx]
        c = cols[valid_idx]

        clearance = float(self.get_parameter("mcmc_min_clearance_m").value)
        free_enough = self.map.free[r, c] & (self.map.distance_m[r, c] >= clearance)

        legal[valid_idx] = free_enough
        return legal

    def mcmc_move_particles(self, scan_msg):
        if not bool(self.get_parameter("mcmc_enabled").value):
            self.latest_mcmc_acceptance_rate = 0.0
            return

        iterations = int(self.get_parameter("mcmc_iterations").value)
        if iterations <= 0:
            self.latest_mcmc_acceptance_rate = 0.0
            return

        xy_std = float(self.get_parameter("mcmc_proposal_xy_std").value)
        yaw_std = float(self.get_parameter("mcmc_proposal_yaw_std").value)
        likelihood_scale = float(self.get_parameter("mcmc_likelihood_scale").value)

        total_accepts = 0
        total_trials = self.n_particles * iterations

        for _ in range(iterations):
            current_particles = self.particles.copy()

            proposals = current_particles.copy()
            proposals[:, 0] += self.rng.normal(0.0, xy_std, size=self.n_particles)
            proposals[:, 1] += self.rng.normal(0.0, xy_std, size=self.n_particles)
            proposals[:, 2] = wrap_angle(
                proposals[:, 2] + self.rng.normal(0.0, yaw_std, size=self.n_particles)
            )

            legal = self.particle_validity_mask(proposals)

            current_log_target = laser_log_likelihood(
                current_particles,
                scan_msg,
                self.map,
                beam_step=int(self.get_parameter("beam_step").value),
                max_beams=int(self.get_parameter("max_beams").value),
                sigma_hit=float(self.get_parameter("sigma_hit").value),
                z_hit=float(self.get_parameter("z_hit").value),
                z_rand=float(self.get_parameter("z_rand").value),
                max_range=float(self.get_parameter("max_range").value),
            )

            proposal_log_target = laser_log_likelihood(
                proposals,
                scan_msg,
                self.map,
                beam_step=int(self.get_parameter("beam_step").value),
                max_beams=int(self.get_parameter("max_beams").value),
                sigma_hit=float(self.get_parameter("sigma_hit").value),
                z_hit=float(self.get_parameter("z_hit").value),
                z_rand=float(self.get_parameter("z_rand").value),
                max_range=float(self.get_parameter("max_range").value),
            )

            # Symmetric Gaussian proposal, so q(x|x') / q(x'|x) = 1.
            # Accept if log(u) < log target(proposal) - log target(current).
            log_alpha = likelihood_scale * (proposal_log_target - current_log_target)
            log_alpha[~legal] = -np.inf

            accept = np.log(self.rng.random(self.n_particles) + 1e-300) < np.minimum(0.0, log_alpha)

            self.particles[accept] = proposals[accept]
            total_accepts += int(np.sum(accept))

        self.latest_mcmc_acceptance_rate = float(total_accepts) / float(max(1, total_trials))

        msg = Float64()
        msg.data = self.latest_mcmc_acceptance_rate
        self.mcmc_acceptance_pub.publish(msg)


    def scan_signature(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        if len(ranges) == 0:
            return np.zeros(0, dtype=np.float64), 0.0

        step = max(1, int(self.get_parameter("beam_step").value))
        sample = ranges[::step]
        max_range = min(float(msg.range_max), float(self.get_parameter("max_range").value))
        valid = (
            np.isfinite(sample)
            & (sample >= float(msg.range_min))
            & (sample <= max_range)
        )
        valid_fraction = float(np.mean(valid)) if len(valid) else 0.0

        signature = sample.copy()
        signature[~valid] = max_range + 1.0
        signature = np.clip(signature, 0.0, max_range + 1.0)
        return signature, valid_fraction

    def update_scan_change_counters(self, msg: LaserScan):
        signature, valid_fraction = self.scan_signature(msg)
        self.latest_scan_valid_fraction = valid_fraction

        if self.last_scan_signature is None or len(signature) != len(self.last_scan_signature):
            delta = float("inf")
        else:
            delta = float(np.median(np.abs(signature - self.last_scan_signature)))

        self.last_scan_signature = signature
        self.latest_scan_delta = delta

        frozen_thresh = float(self.get_parameter("scan_frozen_delta_threshold").value)
        changed_thresh = float(self.get_parameter("scan_changed_delta_threshold").value)
        blind_thresh = float(self.get_parameter("blind_valid_fraction_threshold").value)

        if valid_fraction > blind_thresh and np.isfinite(delta) and delta < frozen_thresh:
            self.scan_frozen_count += 1
        else:
            self.scan_frozen_count = 0

        if valid_fraction > blind_thresh and np.isfinite(delta) and delta > changed_thresh:
            self.scan_changed_count += 1
        else:
            self.scan_changed_count = 0

        return {
            "valid_fraction": valid_fraction,
            "delta": delta,
            "blind": valid_fraction < blind_thresh,
            "frozen": self.scan_frozen_count >= int(self.get_parameter("scan_frozen_required_updates").value),
            "changed": self.scan_changed_count >= int(self.get_parameter("scan_changed_required_updates").value),
            "odom_frozen": self.odom_frozen_count >= int(self.get_parameter("odom_frozen_required_updates").value),
            "motion_expected": self.command_motion_expected(),
        }

    def detection_gate_open(self):
        if not bool(self.get_parameter("enable_attack_detection").value):
            return False
        if not bool(self.get_parameter("detection_requires_initial_convergence").value):
            return True
        return bool(self.initial_converged_once)

    def mode_name(self, mode):
        return {
            1: "Blind Push / temporary blindness",
            2: "Lifted Robot / odometry frozen",
            3: "Kidnapping and teleportation",
        }.get(int(mode), "Unknown")

    def declare_attack(self, mode: int):
        if self.attack_active:
            return

        self.attack_active = True
        self.detected_mode = int(mode)
        self.attack_detection_time = self.now_sec()
        self.recovery_start_time = None
        self.latest_recovery_time_sec = float("nan")
        self.recovery_strategy = "pending"
        self.attack_cmd_distance = 0.0
        self.last_attack_integral_time = self.attack_detection_time
        self.global_recovery_started = False
        self.converged = False
        self.converged_counter = 0

        self.get_logger().info(
            f"ANOMALY DETECTED: Mode {mode} ({self.mode_name(mode)}). "
            f"features: valid_scan={self.latest_scan_valid_fraction:.3f}, "
            f"scan_delta={self.latest_scan_delta:.4f}, "
            f"odom_frozen_count={self.odom_frozen_count}, "
            f"scan_frozen_count={self.scan_frozen_count}, "
            f"scan_changed_count={self.scan_changed_count}"
        )

        if mode == 3:
            self.get_logger().info(
                "Mode 3 identified before recovery: no trustworthy odometry or scan path remains. "
                "Starting GLOBAL recovery now."
            )
            self.global_recovery()

    def reset_convergence_for_recovery(self):
        self.converged = False
        self.converged_counter = 0
        self.scan_update_count = 0
        self.latest_convergence_progress = 0.0

    def sample_local_free_particles(self, center, n_particles, radius, yaw_std):
        samples = []
        remaining = int(n_particles)
        attempts = 0

        while remaining > 0 and attempts < 40:
            attempts += 1
            batch = max(remaining * 3, 200)
            angle = self.rng.uniform(-np.pi, np.pi, size=batch)
            rad = float(radius) * np.sqrt(self.rng.random(batch))
            x = center[0] + rad * np.cos(angle)
            y = center[1] + rad * np.sin(angle)
            yaw = wrap_angle(center[2] + self.rng.normal(0.0, float(yaw_std), size=batch))
            candidates = np.column_stack((x, y, yaw)).astype(np.float64)
            legal = self.particle_validity_mask(candidates)
            if np.any(legal):
                accepted = candidates[legal]
                take = min(remaining, accepted.shape[0])
                samples.append(accepted[:take])
                remaining -= take

        if samples:
            out = np.vstack(samples)
        else:
            out = np.zeros((0, 3), dtype=np.float64)

        if out.shape[0] < n_particles:
            # Last-resort fallback keeps the node alive if the local disk lands in
            # walls.  It is only used for the missing tail, not as the main strategy.
            filler = self.map.sample_free_particles(n_particles - out.shape[0], self.rng)
            out = np.vstack((out, filler))

        return out[:n_particles].astype(np.float64)

    def scan_is_usable_for_recovery(self, msg: LaserScan):
        signature, valid_fraction = self.scan_signature(msg)
        return valid_fraction >= float(self.get_parameter("scan_recovered_valid_fraction").value)

    def local_recovery(self, mode: int, scan_msg: LaserScan):
        if self.latest_estimate is None:
            center = weighted_pose_mean(self.particles, self.weights)
        else:
            center = self.latest_estimate.copy()

        if mode == 1:
            base_radius = float(self.get_parameter("mode1_local_radius_m").value)
            yaw_std = float(self.get_parameter("mode1_local_yaw_std").value)
            strategy = "local_blind_push"
        else:
            base_radius = float(self.get_parameter("mode2_local_radius_m").value)
            yaw_std = float(self.get_parameter("mode2_local_yaw_std").value)
            strategy = "local_lifted_robot_scan_guided"

        radius = base_radius + float(self.get_parameter("local_radius_per_cmd_meter").value) * self.attack_cmd_distance
        radius = float(np.clip(radius, base_radius, float(self.get_parameter("max_local_recovery_radius_m").value)))

        factor = max(1, int(self.get_parameter("local_recovery_candidate_factor").value))
        n_candidates = max(self.n_particles, self.n_particles * factor)
        candidates = self.sample_local_free_particles(center, n_candidates, radius, yaw_std)

        if self.scan_is_usable_for_recovery(scan_msg):
            log_w = laser_log_likelihood(
                candidates,
                scan_msg,
                self.map,
                beam_step=int(self.get_parameter("beam_step").value),
                max_beams=int(self.get_parameter("max_beams").value),
                sigma_hit=float(self.get_parameter("sigma_hit").value),
                z_hit=float(self.get_parameter("z_hit").value),
                z_rand=float(self.get_parameter("z_rand").value),
                max_range=float(self.get_parameter("max_range").value),
            )
            likelihood_scale = float(self.get_parameter("local_recovery_likelihood_scale").value)
            probs = normalize_log_weights(likelihood_scale * log_w)
            indices = self.rng.choice(candidates.shape[0], size=self.n_particles, replace=True, p=probs)
            self.particles = candidates[indices].copy()
        else:
            indices = self.rng.choice(candidates.shape[0], size=self.n_particles, replace=True)
            self.particles = candidates[indices].copy()

        self.weights.fill(1.0 / float(self.n_particles))
        self.recovery_strategy = strategy
        self.recovery_start_time = self.now_sec()
        self.reset_convergence_for_recovery()

        self.get_logger().info(
            f"RECOVERY STARTED for Mode {mode}: LOCAL recovery. "
            f"radius={radius:.2f} m, cmd_distance_est={self.attack_cmd_distance:.2f} m, "
            f"scan_guided={self.scan_is_usable_for_recovery(scan_msg)}"
        )

    def global_recovery(self):
        self.particles = self.map.sample_free_particles(self.n_particles, self.rng)
        self.weights.fill(1.0 / float(self.n_particles))
        self.recovery_strategy = "global_teleport"
        self.recovery_start_time = self.now_sec()
        self.global_recovery_started = True
        self.reset_convergence_for_recovery()
        self.get_logger().info("RECOVERY STARTED for Mode 3: GLOBAL localization over the whole map.")

    def finish_attack_if_recovered_sensor_state(self, features, scan_msg: LaserScan):
        if not self.attack_active:
            return False

        mode = self.detected_mode
        recovered_valid_scan = features["valid_fraction"] >= float(self.get_parameter("scan_recovered_valid_fraction").value)
        odom_not_frozen = not features["odom_frozen"]
        scan_not_frozen = not features["frozen"]

        if mode == 1 and recovered_valid_scan:
            self.get_logger().info("Mode 1 ended: LiDAR is valid again. Starting local recovery.")
            self.local_recovery(1, scan_msg)
            self.attack_active = False
            return True

        if mode == 2 and recovered_valid_scan and odom_not_frozen:
            self.get_logger().info("Mode 2 ended: odometry is updating again. Starting local recovery.")
            self.local_recovery(2, scan_msg)
            self.attack_active = False
            return True

        if mode == 3 and recovered_valid_scan and scan_not_frozen:
            if not self.global_recovery_started:
                self.global_recovery()
            self.get_logger().info("Mode 3 sensor stream is live again. Continuing global recovery until convergence.")
            self.attack_active = False
            return True

        return False

    def update_attack_detector(self, msg: LaserScan):
        self.total_scan_callbacks += 1
        features = self.update_scan_change_counters(msg)
        self.integrate_attack_command_distance()

        # If already inside a detected attack, look only for the end condition.
        if self.attack_active:
            self.finish_attack_if_recovered_sensor_state(features, msg)
            if self.detected_mode == 3 and features["frozen"]:
                return "skip_scan_update"
            if self.detected_mode == 1 and features["blind"]:
                return "skip_scan_update"
            return "continue"

        if not self.detection_gate_open():
            return "continue"

        # Mode 1: LiDAR blindness.  The simulator changes all scan ranges to INF,
        # so the finite-valid beam fraction collapses while odometry can still move.
        if features["blind"]:
            self.declare_attack(1)
            return "skip_scan_update"

        # Mode 3: teleport freezes both odometry and scan.  Detect this before any
        # recovery, then immediately spread globally.
        if features["odom_frozen"] and features["frozen"] and features["motion_expected"]:
            self.declare_attack(3)
            return "skip_scan_update"

        # Mode 2: odometry is frozen while LiDAR remains live/changing.
        if features["odom_frozen"] and features["changed"] and not features["blind"]:
            self.declare_attack(2)
            return "continue"

        return "continue"

    def publish_scan_score(self):
        score_msg = Float64()
        score_msg.data = float(self.latest_scan_match_score)
        self.scan_match_score_pub.publish(score_msg)


    def scan_callback(self, msg: LaserScan):
        if self.prev_odom_pose is None:
            return

        step_start_time = time.perf_counter()

        detector_action = self.update_attack_detector(msg)
        if detector_action == "skip_scan_update":
            self.latest_processing_time_ms = (time.perf_counter() - step_start_time) * 1000.0
            self.publish_processing_time()
            return

        self.scan_update_count += 1

        log_prior = np.log(self.weights + 1e-300)

        log_obs = laser_log_likelihood(
            self.particles,
            msg,
            self.map,
            beam_step=int(self.get_parameter("beam_step").value),
            max_beams=int(self.get_parameter("max_beams").value),
            sigma_hit=float(self.get_parameter("sigma_hit").value),
            z_hit=float(self.get_parameter("z_hit").value),
            z_rand=float(self.get_parameter("z_rand").value),
            max_range=float(self.get_parameter("max_range").value),
        )

        self.latest_scan_match_score = float(np.max(log_obs))
        self.publish_scan_score()

        self.weights = normalize_log_weights(log_prior + log_obs)

        estimate = weighted_pose_mean(self.particles, self.weights)
        spread = weighted_pose_spread(self.particles, self.weights, estimate)
        neff = effective_sample_size(self.weights)

        self.latest_estimate = estimate
        self.latest_spread = spread
        self.latest_neff = neff

        position_error = float("nan")
        yaw_error = float("nan")

        if self.true_pose is not None:
            position_error, yaw_error = pose_error(estimate, self.true_pose)

        progress = self.compute_convergence_progress(spread, position_error, yaw_error)
        self.latest_convergence_progress = progress

        self.publish_error_topics(position_error, yaw_error, neff, progress)
        self.update_convergence(spread, position_error, yaw_error)

        if neff < self.n_particles * float(self.get_parameter("resample_neff_fraction").value):
            self.resample_particles()

            # MCMC Metropolis-Hastings move step after resampling.
            self.mcmc_move_particles(msg)

            # After resample-move, weights are still uniform, but particles may have moved.
            estimate = weighted_pose_mean(self.particles, self.weights)
            spread = weighted_pose_spread(self.particles, self.weights, estimate)

            if self.true_pose is not None:
                position_error, yaw_error = pose_error(estimate, self.true_pose)

        stamp = self.get_clock().now().to_msg()
        self.update_paths(estimate, stamp)
        self.publish_pose_outputs(estimate, stamp)
        self.publish_visualization(estimate, spread, position_error, yaw_error, stamp)

        self.latest_processing_time_ms = (time.perf_counter() - step_start_time) * 1000.0
        self.publish_processing_time()

        self.write_log(estimate, position_error, yaw_error, neff, spread)

        if self.converged and bool(self.get_parameter("stop_robot_on_convergence").value):
            self.cmd_pub.publish(Twist())

    def resample_particles(self):
        indices = systematic_resample(self.weights, self.rng)
        self.particles = self.particles[indices].copy()
        self.weights.fill(1.0 / float(self.n_particles))

        xy_std = float(self.get_parameter("roughening_xy_std").value)
        yaw_std = float(self.get_parameter("roughening_yaw_std").value)

        self.particles[:, 0] += self.rng.normal(0.0, xy_std, size=self.n_particles)
        self.particles[:, 1] += self.rng.normal(0.0, xy_std, size=self.n_particles)
        self.particles[:, 2] = wrap_angle(
            self.particles[:, 2] + self.rng.normal(0.0, yaw_std, size=self.n_particles)
        )

    def update_convergence(self, spread, position_error, yaw_error):
        xy_thresh = float(self.get_parameter("converged_std_xy").value)
        yaw_thresh = float(self.get_parameter("converged_std_yaw").value)
        required = int(self.get_parameter("converged_required_updates").value)
        min_updates = int(self.get_parameter("min_scan_updates_before_convergence").value)

        pos_err_thresh = float(self.get_parameter("converged_position_error").value)
        yaw_err_thresh = float(self.get_parameter("converged_yaw_error").value)

        spread_ok = (
            spread[0] < xy_thresh
            and spread[1] < xy_thresh
            and spread[2] < yaw_thresh
        )

        enough_updates = self.scan_update_count >= min_updates

        if self.true_pose is not None or bool(self.get_parameter("require_truth_for_convergence").value):
            truth_error_ok = (
                np.isfinite(position_error)
                and np.isfinite(yaw_error)
                and position_error < pos_err_thresh
                and abs(yaw_error) < yaw_err_thresh
            )
        else:
            truth_error_ok = True

        live_sensor_ok = (
            not self.attack_active
            and self.latest_scan_valid_fraction >= float(self.get_parameter("scan_recovered_valid_fraction").value)
        )

        convergence_candidate = spread_ok and enough_updates and truth_error_ok and live_sensor_ok

        if convergence_candidate:
            self.converged_counter += 1
        else:
            self.converged_counter = 0

        new_converged = self.converged_counter >= required

        if new_converged and not self.converged:
            self.initial_converged_once = True
            if self.recovery_start_time is not None and not np.isfinite(self.latest_recovery_time_sec):
                self.latest_recovery_time_sec = self.now_sec() - self.recovery_start_time
                self.get_logger().info(
                    f"RECONVERGED after Mode {self.detected_mode} using {self.recovery_strategy}: "
                    f"recovery_time={self.latest_recovery_time_sec:.2f} s"
                )
            else:
                self.get_logger().info("Particle filter converged; attack detector is armed.")

        self.converged = new_converged

        msg = Bool()
        msg.data = bool(self.converged)
        self.converged_pub.publish(msg)

    def publish_processing_time(self):
        msg = Float64()
        msg.data = float(self.latest_processing_time_ms)
        self.processing_time_pub.publish(msg)

    def publish_error_topics(self, position_error, yaw_error, neff, progress):
        msg = Float64()
        msg.data = float(neff)
        self.neff_pub.publish(msg)

        msg = Float64()
        msg.data = float(progress)
        self.convergence_progress_pub.publish(msg)

        if np.isfinite(position_error):
            msg = Float64()
            msg.data = float(position_error)
            self.position_error_pub.publish(msg)

        if np.isfinite(yaw_error):
            msg = Float64()
            msg.data = float(yaw_error)
            self.yaw_error_pub.publish(msg)

    def publish_pose_outputs(self, estimate, stamp):
        est_msg = PoseStamped()
        est_msg.header.stamp = stamp
        est_msg.header.frame_id = self.map_frame
        est_msg.pose = pose_from_xyyaw(estimate[0], estimate[1], estimate[2])
        self.estimated_pose_pub.publish(est_msg)

        if self.true_pose is not None:
            true_msg = PoseStamped()
            true_msg.header.stamp = stamp
            true_msg.header.frame_id = self.map_frame
            true_msg.pose = pose_from_xyyaw(
                self.true_pose[0],
                self.true_pose[1],
                self.true_pose[2],
            )
            self.true_pose_pub.publish(true_msg)

    def compute_convergence_progress(self, spread, position_error, yaw_error):
        xy_thresh = float(self.get_parameter("converged_std_xy").value)
        yaw_thresh = float(self.get_parameter("converged_std_yaw").value)
        min_updates = int(self.get_parameter("min_scan_updates_before_convergence").value)

        pos_err_thresh = float(self.get_parameter("converged_position_error").value)
        yaw_err_thresh = float(self.get_parameter("converged_yaw_error").value)

        update_score = min(1.0, self.scan_update_count / max(1.0, float(min_updates)))

        spread_score = min(
            xy_thresh / max(float(spread[0]), 1e-6),
            xy_thresh / max(float(spread[1]), 1e-6),
            yaw_thresh / max(float(spread[2]), 1e-6),
            1.0,
        )

        if np.isfinite(position_error) and np.isfinite(yaw_error):
            error_score = min(
                pos_err_thresh / max(float(position_error), 1e-6),
                yaw_err_thresh / max(abs(float(yaw_error)), 1e-6),
                1.0,
            )
        elif bool(self.get_parameter("require_truth_for_convergence").value):
            error_score = 0.0
        else:
            error_score = 1.0

        progress = 100.0 * min(update_score, spread_score, error_score)
        return float(np.clip(progress, 0.0, 100.0))

    def append_pose_to_path(self, path_msg, pose3, stamp, last_pose):
        min_dist = float(self.get_parameter("path_publish_min_distance").value)
        max_len = int(self.get_parameter("max_path_length").value)

        if last_pose is not None:
            if np.hypot(pose3[0] - last_pose[0], pose3[1] - last_pose[1]) < min_dist:
                return last_pose

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.map_frame
        ps.pose = pose_from_xyyaw(pose3[0], pose3[1], pose3[2])

        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.map_frame
        path_msg.poses.append(ps)

        if len(path_msg.poses) > max_len:
            path_msg.poses = path_msg.poses[-max_len:]

        return np.array(pose3, dtype=np.float64)

    def update_paths(self, estimate, stamp):
        self.last_estimated_path_pose = self.append_pose_to_path(
            self.estimated_path,
            estimate,
            stamp,
            self.last_estimated_path_pose,
        )
        self.estimated_path_pub.publish(self.estimated_path)

        if self.true_pose is not None:
            self.last_true_path_pose = self.append_pose_to_path(
                self.true_path,
                self.true_pose,
                stamp,
                self.last_true_path_pose,
            )
            self.true_path_pub.publish(self.true_path)


    def publish_visualization(self, estimate, spread, position_error, yaw_error, stamp):
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.map_frame

        max_show = min(self.n_particles, 1500)
        step = max(1, self.n_particles // max_show)

        for p in self.particles[::step]:
            pose_array.poses.append(pose_from_xyyaw(p[0], p[1], p[2]))

        self.particle_pose_pub.publish(pose_array)

        markers = MarkerArray()

        particle_marker = Marker()
        particle_marker.header.stamp = stamp
        particle_marker.header.frame_id = self.map_frame
        particle_marker.ns = "pf_particles"
        particle_marker.id = 0
        particle_marker.type = Marker.POINTS
        particle_marker.action = Marker.ADD
        particle_marker.pose.orientation.w = 1.0
        particle_marker.scale.x = 0.035
        particle_marker.scale.y = 0.035
        particle_marker.color.r = 1.0
        particle_marker.color.g = 1.0
        particle_marker.color.b = 0.0
        particle_marker.color.a = 0.45

        for p in self.particles[::step]:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = 0.02
            particle_marker.points.append(pt)

        markers.markers.append(particle_marker)

        est_marker = Marker()
        est_marker.header.stamp = stamp
        est_marker.header.frame_id = self.map_frame
        est_marker.ns = "pf_estimate"
        est_marker.id = 1
        est_marker.type = Marker.ARROW
        est_marker.action = Marker.ADD
        est_marker.pose = pose_from_xyyaw(estimate[0], estimate[1], estimate[2])
        est_marker.scale.x = 0.35
        est_marker.scale.y = 0.06
        est_marker.scale.z = 0.06
        est_marker.color.r = 0.0
        est_marker.color.g = 1.0
        est_marker.color.b = 0.0
        est_marker.color.a = 1.0
        markers.markers.append(est_marker)

        if self.true_pose is not None:
            true_marker = Marker()
            true_marker.header.stamp = stamp
            true_marker.header.frame_id = self.map_frame
            true_marker.ns = "pf_true"
            true_marker.id = 2
            true_marker.type = Marker.ARROW
            true_marker.action = Marker.ADD
            true_marker.pose = pose_from_xyyaw(
                self.true_pose[0],
                self.true_pose[1],
                self.true_pose[2],
            )
            true_marker.scale.x = 0.35
            true_marker.scale.y = 0.06
            true_marker.scale.z = 0.06
            true_marker.color.r = 0.0
            true_marker.color.g = 0.35
            true_marker.color.b = 1.0
            true_marker.color.a = 1.0
            markers.markers.append(true_marker)

            error_line = Marker()
            error_line.header.stamp = stamp
            error_line.header.frame_id = self.map_frame
            error_line.ns = "pf_error"
            error_line.id = 3
            error_line.type = Marker.LINE_LIST
            error_line.action = Marker.ADD
            error_line.pose.orientation.w = 1.0
            error_line.scale.x = 0.035
            error_line.color.r = 1.0
            error_line.color.g = 0.0
            error_line.color.b = 0.0
            error_line.color.a = 1.0

            p_true = Point()
            p_true.x = float(self.true_pose[0])
            p_true.y = float(self.true_pose[1])
            p_true.z = 0.05

            p_est = Point()
            p_est.x = float(estimate[0])
            p_est.y = float(estimate[1])
            p_est.z = 0.05

            error_line.points.append(p_true)
            error_line.points.append(p_est)
            markers.markers.append(error_line)

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = self.map_frame
        text.ns = "pf_text"
        text.id = 4
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(estimate[0])
        text.pose.position.y = float(estimate[1])
        text.pose.position.z = 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.16
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0

        if np.isfinite(position_error):
            text.text = (
                f"pos err: {position_error:.3f} m\n"
                f"yaw err: {yaw_error:.3f} rad\n"
                f"std: {spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f}\n"
                f"N_eff: {self.latest_neff:.1f}\n"
                f"converged: {self.converged}"
            )
        else:
            text.text = (
                f"std: {spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f}\n"
                f"N_eff: {self.latest_neff:.1f}\n"
                f"converged: {self.converged}"
            )

        markers.markers.append(text)

        progress = float(self.latest_convergence_progress)
        fill = np.clip(progress / 100.0, 0.0, 1.0)

        bar_bg = Marker()
        bar_bg.header.stamp = stamp
        bar_bg.header.frame_id = self.map_frame
        bar_bg.ns = "pf_progress_bar"
        bar_bg.id = 5
        bar_bg.type = Marker.CUBE
        bar_bg.action = Marker.ADD
        bar_bg.pose.position.x = float(estimate[0])
        bar_bg.pose.position.y = float(estimate[1] - 0.70)
        bar_bg.pose.position.z = 0.80
        bar_bg.pose.orientation.w = 1.0
        bar_bg.scale.x = 1.0
        bar_bg.scale.y = 0.08
        bar_bg.scale.z = 0.08
        bar_bg.color.r = 0.20
        bar_bg.color.g = 0.20
        bar_bg.color.b = 0.20
        bar_bg.color.a = 0.80
        markers.markers.append(bar_bg)

        bar_fill = Marker()
        bar_fill.header.stamp = stamp
        bar_fill.header.frame_id = self.map_frame
        bar_fill.ns = "pf_progress_bar"
        bar_fill.id = 6
        bar_fill.type = Marker.CUBE
        bar_fill.action = Marker.ADD
        bar_fill.pose.position.x = float(estimate[0] - 0.5 + 0.5 * fill)
        bar_fill.pose.position.y = float(estimate[1] - 0.70)
        bar_fill.pose.position.z = 0.82
        bar_fill.pose.orientation.w = 1.0
        bar_fill.scale.x = max(0.001, float(fill))
        bar_fill.scale.y = 0.09
        bar_fill.scale.z = 0.09
        bar_fill.color.r = 0.0
        bar_fill.color.g = 1.0
        bar_fill.color.b = 0.0
        bar_fill.color.a = 0.90
        markers.markers.append(bar_fill)

        progress_text = Marker()
        progress_text.header.stamp = stamp
        progress_text.header.frame_id = self.map_frame
        progress_text.ns = "pf_progress_bar"
        progress_text.id = 7
        progress_text.type = Marker.TEXT_VIEW_FACING
        progress_text.action = Marker.ADD
        progress_text.pose.position.x = float(estimate[0])
        progress_text.pose.position.y = float(estimate[1] - 0.70)
        progress_text.pose.position.z = 1.00
        progress_text.pose.orientation.w = 1.0
        progress_text.scale.z = 0.14
        progress_text.color.r = 1.0
        progress_text.color.g = 1.0
        progress_text.color.b = 1.0
        progress_text.color.a = 1.0
        progress_text.text = f"Convergence: {progress:.1f}%"
        markers.markers.append(progress_text)

        self.marker_pub.publish(markers)

    def write_log(self, estimate, position_error, yaw_error, neff, spread):
        t = self.get_clock().now().nanoseconds * 1e-9

        if self.true_pose is None:
            true_values = [float("nan"), float("nan"), float("nan")]
        else:
            true_values = [
                float(self.true_pose[0]),
                float(self.true_pose[1]),
                float(self.true_pose[2]),
            ]

        self.csv_writer.writerow(
            [
                f"{t:.6f}",
                f"{estimate[0]:.6f}",
                f"{estimate[1]:.6f}",
                f"{estimate[2]:.6f}",
                f"{true_values[0]:.6f}",
                f"{true_values[1]:.6f}",
                f"{true_values[2]:.6f}",
                f"{position_error:.6f}",
                f"{yaw_error:.6f}",
                f"{neff:.6f}",
                f"{spread[0]:.6f}",
                f"{spread[1]:.6f}",
                f"{spread[2]:.6f}",
                int(self.converged),
                f"{self.latest_processing_time_ms:.6f}",
                f"{self.latest_convergence_progress:.6f}",
                f"{self.latest_scan_match_score:.6f}",
                f"{self.latest_mcmc_acceptance_rate:.6f}",
                int(self.detected_mode),
                int(self.attack_active),
                self.recovery_strategy,
                f"{self.latest_recovery_time_sec:.6f}",
                f"{self.latest_scan_valid_fraction:.6f}",
                f"{self.latest_scan_delta:.6f}",
                int(self.odom_frozen_count),
            ]
        )
        self.csv_file.flush()

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HijackResilientParticleFilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
