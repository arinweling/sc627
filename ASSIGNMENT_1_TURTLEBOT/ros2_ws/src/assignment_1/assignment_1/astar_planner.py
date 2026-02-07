#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker
import numpy as np
from PIL import Image
import yaml
import heapq
import os
from ament_index_python.packages import get_package_share_directory
import tf2_ros
from tf2_ros import TransformException


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        
        # Declare parameters
        self.declare_parameter('diagonal_cost', 1.414)
        self.declare_parameter('straight_cost', 1.0)
        self.declare_parameter('map_yaml', 'my_gauntlet_map.yaml')
        self.declare_parameter('robot_radius', 0.12)  # Robot radius in meters (includes safety margin)
        self.declare_parameter('waypoint_spacing', 0.2)  # Minimum distance between waypoints (meters)
        self.declare_parameter('path_line_width', 0.05)
        self.declare_parameter('path_color_r', 1.0)
        self.declare_parameter('path_color_g', 0.0)
        self.declare_parameter('path_color_b', 0.0)
        
        # Get parameters
        self.diagonal_cost = self.get_parameter('diagonal_cost').value
        self.straight_cost = self.get_parameter('straight_cost').value
        self.map_yaml = self.get_parameter('map_yaml').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').value
        self.path_line_width = self.get_parameter('path_line_width').value
        self.path_color_r = self.get_parameter('path_color_r').value
        self.path_color_g = self.get_parameter('path_color_g').value
        self.path_color_b = self.get_parameter('path_color_b').value
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.marker_pub = self.create_publisher(Marker, '/path_marker', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # TF for getting robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Load map
        self.load_map()
        
        # Start and goal will be set dynamically
        self.goal = None
        
        self.get_logger().info('A* Planner initialized')
        
        # Publish map periodically
        self.timer = self.create_timer(1.0, self.publish_map)
        
    def load_map(self):
        """Load the map from yaml and pgm files"""
        pkg_path = get_package_share_directory('assignment_1')
        map_yaml_path = os.path.join(pkg_path, 'maps', self.map_yaml)
        
        # Load YAML
        with open(map_yaml_path, 'r') as f:
            map_config = yaml.safe_load(f)
        
        # Load image
        map_image_path = os.path.join(pkg_path, 'maps', map_config['image'])
        img = Image.open(map_image_path)
        img_array = np.array(img)
        
        # Convert to occupancy grid (0 = free, 100 = occupied, -1 = unknown)
        self.map_data = np.zeros_like(img_array, dtype=np.int8)
        self.map_data[img_array < 250] = 100  # Occupied
        self.map_data[img_array >= 250] = 0   # Free
        
        self.resolution = map_config['resolution']
        self.origin = map_config['origin']
        self.height, self.width = self.map_data.shape
        
        self.get_logger().info(f'Map loaded: {self.width}x{self.height}, resolution: {self.resolution}')
        
    def publish_map(self):
        """Publish the occupancy grid map"""
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'map'
        grid.info.resolution = float(self.resolution)
        grid.info.width = int(self.width)
        grid.info.height = int(self.height)
        grid.info.origin.position.x = float(self.origin[0])
        grid.info.origin.position.y = float(self.origin[1])
        grid.info.origin.position.z = 0.0
        # Set map orientation from YAML origin yaw (radians)
        yaw = float(self.origin[2]) if len(self.origin) > 2 else 0.0
        grid.info.origin.orientation.z = np.sin(yaw / 2.0)
        grid.info.origin.orientation.w = np.cos(yaw / 2.0)
        grid.data = self.map_data.flatten().tolist()
        self.map_pub.publish(grid)
        
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
        
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return x, y
        
    def get_robot_pose(self):
        """Get robot's current pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time())
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            return x, y
        except TransformException as ex:
            self.get_logger().warn(f'Could not get robot pose: {ex}')
            return None
    
    def goal_callback(self, msg):
        """Set goal position"""
        x, y = msg.pose.position.x, msg.pose.position.y
        self.goal = self.world_to_grid(x, y)
        self.get_logger().info(f'Goal set to: ({x:.2f}, {y:.2f}) -> grid ({self.goal[0]}, {self.goal[1]})')
        self.plan_path()
        
    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def is_collision_free(self, grid_x, grid_y):
        """Check if robot centered at (grid_x, grid_y) collides with obstacles"""
        robot_radius_cells = int(np.ceil(self.robot_radius / self.resolution))
        
        for dy in range(-robot_radius_cells, robot_radius_cells + 1):
            for dx in range(-robot_radius_cells, robot_radius_cells + 1):
                if dx**2 + dy**2 <= robot_radius_cells**2:  # Circular footprint
                    check_x, check_y = grid_x + dx, grid_y + dy
                    if not (0 <= check_x < self.width and 0 <= check_y < self.height):
                        return False
                    if self.map_data[check_y, check_x] == 100:
                        return False
        return True
        
    def get_neighbors(self, node):
        """Get valid neighbors (8-connectivity) with robot footprint collision checking"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.is_collision_free(nx, ny):  # Check robot footprint
                    neighbors.append((nx, ny))
        return neighbors
        
    def astar(self, start, goal):
        """A* pathfinding algorithm"""
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current == goal:
                break
                
            for next_node in self.get_neighbors(current):
                # Cost is higher for diagonal moves
                move_cost = self.diagonal_cost if abs(next_node[0] - current[0]) + abs(next_node[1] - current[1]) == 2 else self.straight_cost
                new_cost = cost_so_far[current] + move_cost
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        # Reconstruct path
        if goal not in came_from:
            return None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
        
    def plan_path(self):
        """Plan path from start to goal"""
        if self.goal is None:
            return
        
        # Get current robot pose as start
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().error('Cannot get robot pose for planning')
            return
        
        start_x, start_y = robot_pose
        self.start = self.world_to_grid(start_x, start_y)
            
        self.get_logger().info('Planning path...')
        path = self.astar(self.start, self.goal)
        
        if path is None:
            self.get_logger().error('No path found!')
            return
            
        self.get_logger().info(f'Path found with {len(path)} waypoints')
        self.publish_path(path)
        self.publish_path_marker(path)
        
    def publish_path(self, path):
        """Publish path as nav_msgs/Path with reduced waypoint density"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        # Always include first waypoint
        if len(path) > 0:
            grid_x, grid_y = path[0]
            x, y = self.grid_to_world(grid_x, grid_y)
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
            last_x, last_y = x, y
        
        # Add waypoints that are at least waypoint_spacing apart
        for grid_x, grid_y in path[1:]:
            x, y = self.grid_to_world(grid_x, grid_y)
            distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            
            if distance >= self.waypoint_spacing:
                pose = PoseStamped()
                pose.header.stamp = path_msg.header.stamp
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
                last_x, last_y = x, y
        
        # Always include final waypoint (goal)
        if len(path) > 1:
            grid_x, grid_y = path[-1]
            x, y = self.grid_to_world(grid_x, grid_y)
            # Only add if not already added
            if np.sqrt((x - last_x)**2 + (y - last_y)**2) > 0.01:
                pose = PoseStamped()
                pose.header.stamp = path_msg.header.stamp
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)
        
    def publish_path_marker(self, path):
        """Publish path as visualization marker"""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'astar_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = self.path_line_width
        marker.color.r = self.path_color_r
        marker.color.g = self.path_color_g
        marker.color.b = self.path_color_b
        marker.color.a = 1.0
        
        for grid_x, grid_y in path:
            x, y = self.grid_to_world(grid_x, grid_y)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1
            marker.points.append(point)
            
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    planner = AStarPlanner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
