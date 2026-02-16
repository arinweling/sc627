#!/usr/bin/env python3
"""
2D Lidar-based SLAM using Modified Hausdorff Distance (MHD)
Based on: "Robust 2D lidar-based SLAM in arboreal environments without IMU/GNSS"
Author: Implementation for ROS2 and Gazebo
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
import tf2_ros
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import time


class ExtendedKalmanFilter:
    """EKF for pose refinement (x, y, theta)"""
    
    def __init__(self, initial_pose, process_noise, measurement_noise):
        self.state = np.array(initial_pose)  # [x, y, theta]
        self.P = np.eye(3) * 0.1  # Initial covariance
        self.Q = np.diag(process_noise)  # Process noise covariance
        self.R = np.diag(measurement_noise)  # Measurement noise covariance
        
    def predict(self, u, dt):
        """
        Prediction step with motion model
        u = [v, omega] (linear and angular velocity)
        """
        v, omega = u
        theta = self.state[2]
        
        # State transition (differential drive model)
        if abs(omega) < 1e-6:  # Straight line motion
            self.state[0] += v * np.cos(theta) * dt
            self.state[1] += v * np.sin(theta) * dt
        else:
            self.state[0] += (v/omega) * (np.sin(theta + omega*dt) - np.sin(theta))
            self.state[1] += (v/omega) * (-np.cos(theta + omega*dt) + np.cos(theta))
            self.state[2] += omega * dt
        
        # Normalize angle
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
        # Jacobian of motion model
        A = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1, v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
        
        # Covariance prediction
        self.P = A @ self.P @ A.T + self.Q
        
        return self.state.copy()
    
    def update(self, z):
        """
        Update step with measurement from scan matching
        z = [x_measured, y_measured, theta_measured]
        """
        # Measurement model (direct observation)
        H = np.eye(3)
        
        # Innovation
        y = z - self.state
        y[2] = np.arctan2(np.sin(y[2]), np.cos(y[2]))  # Normalize angle difference
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.state = self.state + K @ y
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
        
        # Covariance update
        self.P = (np.eye(3) - K @ H) @ self.P
        
        return self.state.copy()


class ScanMatcher:
    """Scan matching using Modified Hausdorff Distance"""
    
    def __init__(self, k_best=10, max_iterations=50):
        self.k_best = k_best  # Number of best matches for MHD
        self.max_iterations = max_iterations
        
    def modified_hausdorff_distance(self, scan_points, map_dt, map_resolution, map_origin):
        """
        Compute Modified Hausdorff Distance between scan and map
        Uses distance transform for efficiency
        """
        if len(scan_points) == 0:
            return float('inf')
        
        # Convert world coordinates to grid coordinates
        grid_points = self.world_to_grid(scan_points, map_resolution, map_origin)
        
        # Get valid points (within map bounds)
        valid_mask = (grid_points[:, 0] >= 0) & (grid_points[:, 0] < map_dt.shape[1]) & \
                     (grid_points[:, 1] >= 0) & (grid_points[:, 1] < map_dt.shape[0])
        
        valid_points = grid_points[valid_mask]
        
        if len(valid_points) == 0:
            return float('inf')
        
        # Get distances from distance transform
        distances = []
        for point in valid_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= y < map_dt.shape[0] and 0 <= x < map_dt.shape[1]:
                distances.append(map_dt[y, x])
        
        if len(distances) == 0:
            return float('inf')
        
        # Modified Hausdorff: mean of k smallest distances
        distances = np.array(distances)
        k = min(self.k_best, len(distances))
        
        if k >= len(distances):
            # If k is equal to or greater than array length, just use all distances
            k_smallest = distances
        else:
            # Use partition to get k smallest elements
            k_smallest = np.partition(distances, k-1)[:k]
        
        return np.mean(k_smallest)
    
    def world_to_grid(self, points, resolution, origin):
        """Convert world coordinates to grid coordinates"""
        grid_points = np.zeros_like(points)
        grid_points[:, 0] = (points[:, 0] - origin[0]) / resolution
        grid_points[:, 1] = (points[:, 1] - origin[1]) / resolution
        return grid_points.astype(int)
    
    def transform_scan(self, scan_points, pose):
        """Transform scan points by pose [x, y, theta]"""
        x, y, theta = pose
        
        # Rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Transform each point
        transformed = np.zeros_like(scan_points)
        transformed[:, 0] = cos_theta * scan_points[:, 0] - sin_theta * scan_points[:, 1] + x
        transformed[:, 1] = sin_theta * scan_points[:, 0] + cos_theta * scan_points[:, 1] + y
        
        return transformed
    
    def match_scan(self, scan_points, map_dt, map_resolution, map_origin, initial_pose):
        """
        Match scan to map using MHD optimization
        Returns optimized pose [x, y, theta]
        """
        def objective(pose):
            transformed = self.transform_scan(scan_points, pose)
            return self.modified_hausdorff_distance(transformed, map_dt, map_resolution, map_origin)
        
        # Optimization bounds
        x0, y0, theta0 = initial_pose
        bounds = [
            (x0 - 0.5, x0 + 0.5),  # ±0.5m in x
            (y0 - 0.5, y0 + 0.5),  # ±0.5m in y
            (theta0 - np.pi/4, theta0 + np.pi/4)  # ±45 degrees
        ]
        
        # Optimize
        result = minimize(
            objective,
            initial_pose,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iterations}
        )
        
        return result.x, result.fun


class MHDSLAM(Node):
    """Main SLAM node using MHD scan matching and EKF"""
    
    def __init__(self):
        super().__init__('mhd_slam')
        
        # Parameters
        self.declare_parameter('map_resolution', 0.05)  # meters per pixel
        self.declare_parameter('map_width', 200)  # grid cells
        self.declare_parameter('map_height', 200)  # grid cells
        self.declare_parameter('k_best_matches', 10)
        self.declare_parameter('scan_min_range', 0.12)
        self.declare_parameter('scan_max_range', 3.5)
        self.declare_parameter('min_movement_update', 0.05)  # meters
        self.declare_parameter('min_rotation_update', 0.05)  # radians
        self.declare_parameter('process_noise', [0.005, 0.005, 0.005])  # [x, y, theta]
        self.declare_parameter('measurement_noise', [0.08, 0.08, 0.15])  # [x, y, theta]
        self.declare_parameter('occupancy_threshold', 0.5)
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('tf_publish_rate', 20.0)  # Hz - high frequency TF
        
        # Get parameters
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        k_best = self.get_parameter('k_best_matches').value
        self.scan_min_range = self.get_parameter('scan_min_range').value
        self.scan_max_range = self.get_parameter('scan_max_range').value
        self.min_movement_update = self.get_parameter('min_movement_update').value
        self.min_rotation_update = self.get_parameter('min_rotation_update').value
        process_noise = self.get_parameter('process_noise').value
        measurement_noise = self.get_parameter('measurement_noise').value
        self.occupancy_threshold = self.get_parameter('occupancy_threshold').value
        pub_rate = self.get_parameter('publish_rate').value
        tf_rate = self.get_parameter('tf_publish_rate').value
        
        # Initialize components
        self.scan_matcher = ScanMatcher(k_best=k_best)
        self.ekf = ExtendedKalmanFilter(
            initial_pose=[0.0, 0.0, 0.0],
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        
        # Map representation (occupancy grid)
        self.map_origin = [-self.map_width * self.map_resolution / 2,
                          -self.map_height * self.map_resolution / 2]
        self.occupancy_map = np.ones((self.map_height, self.map_width)) * 0.5  # Unknown = 0.5
        self.distance_transform = None
        
        # State
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.last_scan_points = None
        self.first_scan = True
        self.trajectory = []
        self.last_update_pose = np.array([0.0, 0.0, 0.0])  # Track when we last updated
        
        # Odometry for motion model
        self.last_odom = None
        self.last_time = None
        
        # QoS profile for map publisher (TRANSIENT_LOCAL for map_saver compatibility)
        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, '/slam_map', map_qos)
        self.pose_pub = self.create_publisher(PoseStamped, '/slam_pose', 10)
        self.pose_cov_pub = self.create_publisher(PoseWithCovarianceStamped, '/slam_pose_cov', 10)
        self.trajectory_pub = self.create_publisher(Path, '/slam_trajectory', 10)
        self.scan_marker_pub = self.create_publisher(Marker, '/slam_scan_points', 10)
        
        # QoS profile for sensor data (compatible with real robot)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers (using BEST_EFFORT for real robot compatibility)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, sensor_qos)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 50)
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Timers
        self.map_pub_timer = self.create_timer(1.0 / pub_rate, self.publish_map)
        self.tf_pub_timer = self.create_timer(1.0 / tf_rate, self.publish_tf)  # High-frequency TF
        
        self.get_logger().info('MHD SLAM node initialized')
        self.get_logger().info(f'Map: {self.map_width}x{self.map_height} @ {self.map_resolution}m/cell')
        self.get_logger().info(f'TF publish rate: {tf_rate} Hz')
    
    def odom_callback(self, msg):
        """Store odometry for motion model"""
        if self.last_odom is None:
            self.get_logger().info('First odometry message received')
        self.last_odom = msg
        if self.last_time is None:
            self.last_time = self.get_clock().now()
    
    def scan_callback(self, msg):
        """Process laser scan and update SLAM"""
        # Convert scan to Cartesian points
        scan_points = self.scan_to_points(msg)
        
        if len(scan_points) < 10:
            self.get_logger().warn('Too few scan points')
            return
        
        current_time = self.get_clock().now()
        
        # First scan: initialize map
        if self.first_scan:
            self.get_logger().info(f'Initializing map with {len(scan_points)} scan points')
            self.initialize_map(scan_points)
            self.first_scan = False
            self.last_scan_points = scan_points
            self.last_time = current_time
            self.get_logger().info('Map initialized successfully')
            return
        
        # EKF Prediction step
        if self.last_odom is not None and self.last_time is not None:
            dt = (current_time - self.last_time).nanoseconds / 1e9
            if dt > 0:
                # Get velocity from odometry
                v = self.last_odom.twist.twist.linear.x
                omega = self.last_odom.twist.twist.angular.z
                predicted_pose = self.ekf.predict([v, omega], dt)
                self.last_time = current_time
            else:
                predicted_pose = self.current_pose
        else:
            if self.last_odom is None:
                self.get_logger().warn('No odometry data received yet - waiting for /odom topic', throttle_duration_sec=5.0)
            predicted_pose = self.current_pose
        
        # Scan matching
        if self.distance_transform is not None:
            matched_pose, match_score = self.scan_matcher.match_scan(
                scan_points, self.distance_transform,
                self.map_resolution, self.map_origin,
                predicted_pose
            )
            
            # EKF Update step with matched pose
            corrected_pose = self.ekf.update(matched_pose)
            self.current_pose = corrected_pose
            
            self.get_logger().info(
                f'Pose: x={self.current_pose[0]:.2f}, y={self.current_pose[1]:.2f}, '
                f'theta={np.degrees(self.current_pose[2]):.1f}°, score={match_score:.4f}'
            )
        else:
            self.current_pose = predicted_pose
        
        # Only update map if robot has moved enough (reduce drift)
        pose_diff = self.current_pose - self.last_update_pose
        distance_moved = np.sqrt(pose_diff[0]**2 + pose_diff[1]**2)
        rotation_moved = abs(pose_diff[2])
        
        if distance_moved > self.min_movement_update or rotation_moved > self.min_rotation_update:
            self.update_map(scan_points, self.current_pose)
            self.last_update_pose = self.current_pose.copy()
        
        # Store trajectory
        self.trajectory.append(self.current_pose.copy())
        
        # Publish pose and trajectory
        self.publish_pose()
        self.publish_trajectory()
        self.publish_scan_visualization(scan_points)
        
        self.last_scan_points = scan_points
    
    def scan_to_points(self, scan_msg):
        """Convert LaserScan to Cartesian points in sensor frame"""
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(
            scan_msg.angle_min,
            scan_msg.angle_max,
            len(ranges)
        )
        
        # Filter valid ranges
        valid_mask = (ranges >= self.scan_min_range) & \
                     (ranges <= self.scan_max_range) & \
                     np.isfinite(ranges)
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        # Convert to Cartesian (in sensor frame, which is ~robot frame for TurtleBot3)
        points = np.zeros((len(valid_ranges), 2))
        points[:, 0] = valid_ranges * np.cos(valid_angles)
        points[:, 1] = valid_ranges * np.sin(valid_angles)
        
        return points
    
    def initialize_map(self, scan_points):
        """Initialize map with first scan"""
        self.get_logger().info(f'Initializing map at pose: {self.current_pose}')
        transformed_points = self.scan_matcher.transform_scan(scan_points, self.current_pose)
        self.get_logger().info(f'Transformed {len(transformed_points)} points to world frame')
        self.integrate_scan_to_map(transformed_points, self.current_pose)
        self.get_logger().info('Map initialized with first scan')
    
    def update_map(self, scan_points, pose):
        """Update occupancy grid map with new scan"""
        transformed_points = self.scan_matcher.transform_scan(scan_points, pose)
        self.integrate_scan_to_map(transformed_points, pose)
    
    def integrate_scan_to_map(self, world_points, robot_pose):
        """
        Integrate scan points into occupancy grid using ray tracing
        """
        robot_x, robot_y, _ = robot_pose
        
        # Convert robot position to grid coordinates
        robot_grid_x = int((robot_x - self.map_origin[0]) / self.map_resolution)
        robot_grid_y = int((robot_y - self.map_origin[1]) / self.map_resolution)
        
        valid_points = 0
        for point in world_points:
            # Convert point to grid coordinates
            grid_x = int((point[0] - self.map_origin[0]) / self.map_resolution)
            grid_y = int((point[1] - self.map_origin[1]) / self.map_resolution)
            
            # Check bounds
            if not (0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height):
                continue
            
            valid_points += 1
            # Ray tracing: mark cells along ray as free
            ray_cells = self.bresenham(robot_grid_x, robot_grid_y, grid_x, grid_y)
            
            for rx, ry in ray_cells[:-1]:  # All except endpoint
                if 0 <= rx < self.map_width and 0 <= ry < self.map_height:
                    # Update free space (Bayesian update)
                    self.occupancy_map[ry, rx] = max(0.0, self.occupancy_map[ry, rx] - 0.1)
            
            # Mark endpoint as occupied
            if 0 <= grid_x < self.map_width and 0 <= grid_y < self.map_height:
                self.occupancy_map[grid_y, grid_x] = min(1.0, self.occupancy_map[grid_y, grid_x] + 0.3)
        
        if valid_points == 0:
            self.get_logger().warn(f'No valid points in map bounds! Robot at grid ({robot_grid_x}, {robot_grid_y})')
        
        # Update distance transform
        occupied_cells = self.occupancy_map > self.occupancy_threshold
        self.distance_transform = distance_transform_edt(~occupied_cells)
    
    def bresenham(self, x0, y0, x1, y1):
        """Bresenham's line algorithm for ray tracing"""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells
    
    def publish_map(self):
        """Publish occupancy grid map"""
        if self.occupancy_map is None:
            return
        
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin[0]
        msg.info.origin.position.y = self.map_origin[1]
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Convert to ROS occupancy grid format (0-100, -1 for unknown)
        # occupancy_map is 0.0-1.0, convert to 0-100 for occupied probability
        occupancy_int = (self.occupancy_map * 100).astype(np.int8)
        
        # Mark cells that are still near 0.5 (unknown) as -1
        # Since we initialized with 0.5, anything close to that is unknown
        unknown_mask = np.abs(self.occupancy_map - 0.5) < 0.05
        occupancy_int[unknown_mask] = -1
        
        occupancy_data = occupancy_int.flatten()
        msg.data = occupancy_data.tolist()
        
        self.map_pub.publish(msg)
    
    def publish_pose(self):
        """Publish current pose"""
        # PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = self.current_pose[0]
        pose_msg.pose.position.y = self.current_pose[1]
        pose_msg.pose.position.z = 0.0
        
        q = quaternion_from_euler(0, 0, self.current_pose[2])
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        self.pose_pub.publish(pose_msg)
        
        # PoseWithCovarianceStamped
        pose_cov_msg = PoseWithCovarianceStamped()
        pose_cov_msg.header = pose_msg.header
        pose_cov_msg.pose.pose = pose_msg.pose
        
        # Fill covariance from EKF
        covariance = np.zeros(36)
        covariance[0] = self.ekf.P[0, 0]  # x-x
        covariance[1] = self.ekf.P[0, 1]  # x-y
        covariance[6] = self.ekf.P[1, 0]  # y-x
        covariance[7] = self.ekf.P[1, 1]  # y-y
        covariance[35] = self.ekf.P[2, 2]  # theta-theta
        pose_cov_msg.pose.covariance = covariance.tolist()
        
        self.pose_cov_pub.publish(pose_cov_msg)
    
    def publish_tf(self):
        """Publish TF transform at high frequency (separate from pose publishing)"""
        # SLAM publishes map -> odom transform
        # Odometry plugin publishes odom -> base_footprint
        # This creates the chain: map -> odom -> base_footprint -> base_link
        
        if self.last_odom is None:
            return
        
        # Get robot pose in odom frame from odometry
        odom_x = self.last_odom.pose.pose.position.x
        odom_y = self.last_odom.pose.pose.position.y
        odom_quat = self.last_odom.pose.pose.orientation
        _, _, odom_theta = euler_from_quaternion([odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])
        
        # current_pose is robot pose in map frame
        map_x = self.current_pose[0]
        map_y = self.current_pose[1]
        map_theta = self.current_pose[2]
        
        # Compute map->odom transform
        # map_pose = map_to_odom * odom_pose
        # Therefore: map_to_odom = map_pose * inv(odom_pose)
        dx = map_x - (odom_x * np.cos(map_theta - odom_theta) - odom_y * np.sin(map_theta - odom_theta))
        dy = map_y - (odom_x * np.sin(map_theta - odom_theta) + odom_y * np.cos(map_theta - odom_theta))
        dtheta = map_theta - odom_theta
        
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = dx
        t.transform.translation.y = dy
        t.transform.translation.z = 0.0
        
        q = quaternion_from_euler(0, 0, dtheta)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_trajectory(self):
        """Publish trajectory path"""
        if len(self.trajectory) == 0:
            return
        
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for pose in self.trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = pose[0]
            pose_stamped.pose.position.y = pose[1]
            pose_stamped.pose.position.z = 0.0
            
            q = quaternion_from_euler(0, 0, pose[2])
            pose_stamped.pose.orientation.x = q[0]
            pose_stamped.pose.orientation.y = q[1]
            pose_stamped.pose.orientation.z = q[2]
            pose_stamped.pose.orientation.w = q[3]
            
            path_msg.poses.append(pose_stamped)
        
        self.trajectory_pub.publish(path_msg)
    
    def publish_scan_visualization(self, scan_points):
        """Publish scan points as markers for visualization"""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'scan_points'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # Transform points to world frame
        world_points = self.scan_matcher.transform_scan(scan_points, self.current_pose)
        
        for point in world_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.0
            marker.points.append(p)
        
        self.scan_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = MHDSLAM()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
