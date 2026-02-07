#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker
import numpy as np
from PIL import Image
from scipy import ndimage
import yaml
import os
from collections import deque
from ament_index_python.packages import get_package_share_directory
import tf2_ros
from tf2_ros import TransformException


class GVDPlanner(Node):
    def __init__(self):
        super().__init__('gvd_planner')
        
        # Declare parameters
        self.declare_parameter('skeleton_threshold', 3.5)
        self.declare_parameter('skeleton_cost_multiplier', 0.2)
        self.declare_parameter('diagonal_cost', 1.414)
        self.declare_parameter('straight_cost', 1.0)
        self.declare_parameter('search_radius', 50)
        self.declare_parameter('map_yaml', 'my_gauntlet_map.yaml')
        self.declare_parameter('waypoint_spacing', 0.2)  # Minimum distance between waypoints (meters)
        self.declare_parameter('path_line_width', 0.05)
        self.declare_parameter('path_color_r', 0.0)
        self.declare_parameter('path_color_g', 1.0)
        self.declare_parameter('path_color_b', 0.0)
        self.declare_parameter('skeleton_point_size', 0.03)
        self.declare_parameter('skeleton_color_r', 0.0)
        self.declare_parameter('skeleton_color_g', 1.0)
        self.declare_parameter('skeleton_color_b', 1.0)
        
        # Get parameters
        self.skeleton_threshold = self.get_parameter('skeleton_threshold').value
        self.skeleton_cost_multiplier = self.get_parameter('skeleton_cost_multiplier').value
        self.diagonal_cost = self.get_parameter('diagonal_cost').value
        self.straight_cost = self.get_parameter('straight_cost').value
        self.search_radius = self.get_parameter('search_radius').value
        self.map_yaml = self.get_parameter('map_yaml').value
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').value
        self.path_line_width = self.get_parameter('path_line_width').value
        self.path_color_r = self.get_parameter('path_color_r').value
        self.path_color_g = self.get_parameter('path_color_g').value
        self.path_color_b = self.get_parameter('path_color_b').value
        self.skeleton_point_size = self.get_parameter('skeleton_point_size').value
        self.skeleton_color_r = self.get_parameter('skeleton_color_r').value
        self.skeleton_color_g = self.get_parameter('skeleton_color_g').value
        self.skeleton_color_b = self.get_parameter('skeleton_color_b').value
        
        # Publishers
        self.path_pub = self.create_publisher(Path, '/gvd_path', 10)
        self.marker_pub = self.create_publisher(Marker, '/gvd_path_marker', 10)
        self.skeleton_pub = self.create_publisher(Marker, '/gvd_skeleton', 10)
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # TF for getting robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Load map
        self.load_map()
        
        # Compute GVD using brushfire
        self.compute_gvd()
        
        # Start and goal will be set dynamically
        self.goal = None
        
        self.get_logger().info('GVD Planner initialized')
        
        # Publish skeleton periodically
        self.timer = self.create_timer(1.0, self.publish_skeleton)
        
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
        
        # Convert to binary map (0 = occupied, 1 = free)
        self.binary_map = np.zeros_like(img_array, dtype=np.uint8)
        self.binary_map[img_array >= 250] = 1  # Free space
        
        # Convert to occupancy grid for publishing
        self.map_data = np.zeros_like(img_array, dtype=np.int8)
        self.map_data[img_array < 250] = 100  # Occupied
        self.map_data[img_array >= 250] = 0   # Free
        
        self.resolution = map_config['resolution']
        self.origin = map_config['origin']
        self.height, self.width = self.map_data.shape
        
        self.get_logger().info(f'Map loaded: {self.width}x{self.height}, resolution: {self.resolution}')
        
    def compute_gvd(self):
        """Compute Generalized Voronoi Diagram using Brushfire algorithm"""
        self.get_logger().info('Computing GVD with Brushfire algorithm...')
        
        # Step 1: Compute distance transform (brushfire from obstacles)
        self.distance_map = ndimage.distance_transform_edt(self.binary_map)
        
        max_dist = np.max(self.distance_map)
        self.get_logger().info(f'Distance map computed. Min: {np.min(self.distance_map):.2f}, Max: {max_dist:.2f}')
        
        # Step 2: Create binary mask of areas far from obstacles
        safe_zone = (self.distance_map >= self.skeleton_threshold).astype(np.uint8)
        
        # Step 3: Apply morphological skeletonization to safe zone for connectivity
        from skimage.morphology import skeletonize
        self.skeleton = skeletonize(safe_zone).astype(np.uint8)
        
        skeleton_points = np.sum(self.skeleton)
        self.get_logger().info(f'GVD skeleton computed with threshold {self.skeleton_threshold} pixels, {skeleton_points} skeleton points')
        
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
        
    def find_nearest_skeleton_point(self, point):
        """Find nearest point on skeleton to given point"""
        min_dist = float('inf')
        nearest = None
        
        px, py = point
        
        for dy in range(-self.search_radius, self.search_radius + 1):
            for dx in range(-self.search_radius, self.search_radius + 1):
                x, y = px + dx, py + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.skeleton[y, x] == 1:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = (x, y)
                            
        return nearest if nearest else point
        
    def get_skeleton_neighbors(self, node):
        """Get neighbors that are ONLY on skeleton"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                # Only allow skeleton points as neighbors
                if self.skeleton[ny, nx] == 1:
                    neighbors.append((nx, ny))
        return neighbors
        
    def astar_on_skeleton(self, start, goal):
        """A* algorithm on skeleton points only"""
        import heapq
        
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current == goal:
                break
                
            for next_node in self.get_skeleton_neighbors(current):
                # Cost is just euclidean distance
                move_cost = self.diagonal_cost if abs(next_node[0] - current[0]) + abs(next_node[1] - current[1]) == 2 else self.straight_cost
                    
                new_cost = cost_so_far[current] + move_cost
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    # A* heuristic: g + h
                    priority = new_cost + np.sqrt((next_node[0] - goal[0])**2 + (next_node[1] - goal[1])**2)
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
        """Plan path using GVD: straight to skeleton, follow skeleton, straight to goal"""
        if self.goal is None:
            return
        
        # Get current robot pose as start
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().error('Cannot get robot pose for planning')
            return
        
        start_x, start_y = robot_pose
        self.start = self.world_to_grid(start_x, start_y)
        self.get_logger().info(f'Planning path from robot pose ({start_x:.2f}, {start_y:.2f}) to goal')
        
        # Find nearest skeleton points to start and goal
        start_skeleton = self.find_nearest_skeleton_point(self.start)
        goal_skeleton = self.find_nearest_skeleton_point(self.goal)
        
        self.get_logger().info(f'Start skeleton: {start_skeleton}, Goal skeleton: {goal_skeleton}')
        
        # Plan on skeleton ONLY
        skeleton_path = self.astar_on_skeleton(start_skeleton, goal_skeleton)
        
        if skeleton_path is None:
            self.get_logger().error('No path found on skeleton!')
            return
            
        # Build full path: start -> skeleton entry -> skeleton path -> skeleton exit -> goal
        full_path = [self.start] + skeleton_path + [self.goal]
        
        self.get_logger().info(f'Path found with {len(full_path)} waypoints')
        self.publish_path(full_path)
        self.publish_path_marker(full_path)
        
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
        marker.ns = 'path'
        marker.id = 1
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

    def publish_skeleton(self):
        """Publish skeleton as visualization marker"""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.ns = 'skeleton'
        marker.id = 2
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = self.skeleton_point_size
        marker.scale.y = self.skeleton_point_size
        marker.color.r = self.skeleton_color_r
        marker.color.g = self.skeleton_color_g
        marker.color.b = self.skeleton_color_b
        marker.color.a = 1.0
        
        for y in range(self.height):
            for x in range(self.width):
                if self.skeleton[y, x] == 1:
                    wx, wy = self.grid_to_world(x, y)
                    point = Point()
                    point.x = wx
                    point.y = wy
                    point.z = 0.05
                    marker.points.append(point)
                    
        self.skeleton_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    planner = GVDPlanner()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
