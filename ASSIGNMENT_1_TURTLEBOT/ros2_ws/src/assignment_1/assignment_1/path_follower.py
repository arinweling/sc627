#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import tf2_ros
import numpy as np
from tf2_ros import TransformException


class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')
        
        # Parameters
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('k_p', 0.5)  # Proportional gain for angular control
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('path_topic', '/planned_path')  # or '/gvd_path'
        
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.k_p = self.get_parameter('k_p').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        path_topic = self.get_parameter('path_topic').value
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path, path_topic, self.path_callback, 10)
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State
        self.current_path = None
        self.current_waypoint_idx = 0
        
        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f'Path Follower initialized, listening to {path_topic}')
        
    def path_callback(self, msg):
        """Receive new path"""
        if len(msg.poses) > 0:
            self.current_path = msg
            self.current_waypoint_idx = 0
            self.get_logger().info(f'New path received with {len(msg.poses)} waypoints')
        
    def get_robot_pose(self):
        """Get robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time())
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            
            # Extract yaw from quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            
            return x, y, yaw
        except TransformException as ex:
            self.get_logger().warn(f'Could not get robot pose: {ex}')
            return None
            
    def control_loop(self):
        """Main control loop"""
        if self.current_path is None or len(self.current_path.poses) == 0:
            return
            
        # Get current robot pose
        pose = self.get_robot_pose()
        if pose is None:
            return
            
        robot_x, robot_y, robot_yaw = pose
        
        # Get current waypoint
        if self.current_waypoint_idx >= len(self.current_path.poses):
            # Path completed
            self.stop_robot()
            self.get_logger().info('Path completed!')
            self.current_path = None
            return
            
        waypoint = self.current_path.poses[self.current_waypoint_idx]
        goal_x = waypoint.pose.position.x
        goal_y = waypoint.pose.position.y
        
        # Calculate distance and angle to waypoint
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_goal = np.arctan2(dy, dx)
        angle_error = self.normalize_angle(angle_to_goal - robot_yaw)
        
        # Check if waypoint reached
        if distance < self.goal_tolerance:
            self.current_waypoint_idx += 1
            self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}/{len(self.current_path.poses)}')
            return
            
        # Compute velocities
        cmd = Twist()
        
        # Proportional control
        if abs(angle_error) > 0.2:  # Need to turn more
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_speed * np.sign(angle_error)
        else:
            cmd.linear.x = min(self.linear_speed, distance * self.k_p)
            cmd.angular.z = self.angular_speed * angle_error
            
        self.cmd_vel_pub.publish(cmd)
        
    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    follower = PathFollower()
    rclpy.spin(follower)
    follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
