#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize
import yaml
import heapq
import os
from ament_index_python.packages import get_package_share_directory
import tf2_ros
from tf2_ros import TransformException


class PlannerServer(Node):
    def __init__(self):
        super().__init__('planner_server')
        
        # Declare parameters
        self.declare_parameter('planner_type', 'both')  # 'astar', 'gvd', or 'both'
        self.declare_parameter('diagonal_cost', 1.414)
        self.declare_parameter('straight_cost', 1.0)
        self.declare_parameter('robot_radius', 0.12)
        self.declare_parameter('waypoint_spacing', 0.2)
        
        # A* specific
        self.declare_parameter('astar_path_line_width', 0.15)
        self.declare_parameter('astar_path_color_r', 1.0)
        self.declare_parameter('astar_path_color_g', 0.0)
        self.declare_parameter('astar_path_color_b', 0.0)
        
        # GVD specific
        self.declare_parameter('skeleton_threshold', 2.0)
        self.declare_parameter('search_radius', 50)
        self.declare_parameter('gvd_path_line_width', 0.15)
        self.declare_parameter('gvd_path_color_r', 0.0)
        self.declare_parameter('gvd_path_color_g', 1.0)
        self.declare_parameter('gvd_path_color_b', 0.0)
        self.declare_parameter('skeleton_point_size', 0.03)
        self.declare_parameter('skeleton_color_r', 0.0)
        self.declare_parameter('skeleton_color_g', 1.0)
        self.declare_parameter('skeleton_color_b', 1.0)
        
        # Get parameters
        self.planner_type = self.get_parameter('planner_type').value
        self.diagonal_cost = self.get_parameter('diagonal_cost').value
        self.straight_cost = self.get_parameter('straight_cost').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.waypoint_spacing = self.get_parameter('waypoint_spacing').value
        
        # A* parameters
        self.astar_path_line_width = self.get_parameter('astar_path_line_width').value
        self.astar_path_color_r = self.get_parameter('astar_path_color_r').value
        self.astar_path_color_g = self.get_parameter('astar_path_color_g').value
        self.astar_path_color_b = self.get_parameter('astar_path_color_b').value
        
        # GVD parameters
        self.skeleton_threshold = self.get_parameter('skeleton_threshold').value
        self.search_radius = self.get_parameter('search_radius').value
        self.gvd_path_line_width = self.get_parameter('gvd_path_line_width').value
        self.gvd_path_color_r = self.get_parameter('gvd_path_color_r').value
        self.gvd_path_color_g = self.get_parameter('gvd_path_color_g').value
        self.gvd_path_color_b = self.get_parameter('gvd_path_color_b').value
        self.skeleton_point_size = self.get_parameter('skeleton_point_size').value
        self.skeleton_color_r = self.get_parameter('skeleton_color_r').value
        self.skeleton_color_g = self.get_parameter('skeleton_color_g').value
        self.skeleton_color_b = self.get_parameter('skeleton_color_b').value
        
        # Publishers
        if self.planner_type in ['astar', 'both']:
            self.astar_path_pub = self.create_publisher(Path, '/planned_path', 10)
            self.astar_marker_pub = self.create_publisher(Marker, '/path_marker', 10)
            
        if self.planner_type in ['gvd', 'both']:
            self.gvd_path_pub = self.create_publisher(Path, '/gvd_path', 10)
            self.gvd_marker_pub = self.create_publisher(Marker, '/gvd_path_marker', 10)
            self.skeleton_pub = self.create_publisher(Marker, '/gvd_skeleton', 10)
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # TF for getting robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Map data - will be populated from /map topic
        self.map_received = False
        self.binary_map = None
        self.map_data = None
        self.skeleton = None
        self.distance_map = None
        
        # Goal will be set dynamically
        self.goal = None
        
        self.get_logger().info(f'Planner Server initialized - waiting for map from /map topic')
        
        # Publish skeleton periodically if GVD is enabled
        if self.planner_type in ['gvd', 'both']:
            self.timer = self.create_timer(1.0, self.publish_skeleton)
        
    def map_callback(self, msg):
        """Receive map from /map topic"""
        if self.map_received:
            return  # Only process map once
        
        self.get_logger().info('Received map from /map topic')
        
        # Extract map metadata
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.0]
        
        # Convert OccupancyGrid data to numpy array
        map_data_array = np.array(msg.data, dtype=np.int8).reshape((self.height, self.width))
        
        # Store as map_data (0 = free, 100 = occupied, -1 = unknown)
        self.map_data = map_data_array
        
        # Convert to binary map for GVD (0 = occupied, 1 = free)
        # OccupancyGrid: -1 = unknown, 0 = free, 100 = occupied
        self.binary_map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.binary_map[map_data_array == 0] = 1  # Free space
        self.binary_map[map_data_array == -1] = 0  # Treat unknown as occupied for safety
        # map_data == 100 stays 0 (occupied)
        
        self.get_logger().info(f'Map received: {self.width}x{self.height}, resolution: {self.resolution}')
        self.get_logger().info(f'Origin: [{self.origin[0]:.2f}, {self.origin[1]:.2f}]')
        
        # Compute GVD if needed
        if self.planner_type in ['gvd', 'both']:
            self.compute_gvd()
        
        self.map_received = True
        self.get_logger().info(f'Planner Server ready with {self.planner_type} planner(s)')
        
    def compute_gvd(self):
        """Compute Generalized Voronoi Diagram using distance transform and skeletonization"""
        # Distance transform (Euclidean distance to nearest obstacle)
        self.distance_map = ndimage.distance_transform_edt(self.binary_map)
        
        # Create skeleton using skeletonization
        safe_zone = (self.distance_map > self.skeleton_threshold).astype(np.uint8)
        self.skeleton = skeletonize(safe_zone).astype(np.uint8)
        
        skeleton_points = np.sum(self.skeleton)
        self.get_logger().info(f'GVD computed: {skeleton_points} skeleton points')
        
    def publish_skeleton(self):
        """Publish skeleton as point markers"""
        if not self.map_received or self.skeleton is None:
            return
            
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
        """Set goal position and plan paths"""
        if not self.map_received:
            self.get_logger().warn('Cannot plan path - map not received yet')
            return
        
        x, y = msg.pose.position.x, msg.pose.position.y
        self.goal = self.world_to_grid(x, y)
        self.get_logger().info(f'Goal set to: ({x:.2f}, {y:.2f}) -> grid ({self.goal[0]}, {self.goal[1]})')
        
        # Get robot pose
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            self.get_logger().error('Cannot get robot pose for planning')
            return
        
        start_x, start_y = robot_pose
        self.start = self.world_to_grid(start_x, start_y)
        
        # Plan with A* if enabled
        if self.planner_type in ['astar', 'both']:
            self.plan_astar()
            
        # Plan with GVD if enabled
        if self.planner_type in ['gvd', 'both']:
            self.plan_gvd()
    
    # ==================== A* PLANNER ====================
    
    def is_collision_free(self, grid_x, grid_y):
        """Check if robot centered at (grid_x, grid_y) collides with obstacles"""
        robot_radius_cells = int(np.ceil(self.robot_radius / self.resolution))
        
        for dy in range(-robot_radius_cells, robot_radius_cells + 1):
            for dx in range(-robot_radius_cells, robot_radius_cells + 1):
                if dx**2 + dy**2 <= robot_radius_cells**2:
                    check_x, check_y = grid_x + dx, grid_y + dy
                    if not (0 <= check_x < self.width and 0 <= check_y < self.height):
                        return False
                    if self.map_data[check_y, check_x] == 100:
                        return False
        return True
        
    def get_neighbors_astar(self, node):
        """Get valid neighbors for A* (8-connectivity with robot footprint collision checking)"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.is_collision_free(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors
        
    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        
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
                
            for next_node in self.get_neighbors_astar(current):
                move_cost = self.diagonal_cost if abs(next_node[0] - current[0]) + abs(next_node[1] - current[1]) == 2 else self.straight_cost
                new_cost = cost_so_far[current] + move_cost
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        if goal not in came_from:
            return None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
        
    def plan_astar(self):
        """Plan path using A*"""
        self.get_logger().info('Planning A* path...')
        path = self.astar(self.start, self.goal)
        
        if path is None:
            self.get_logger().error('A*: No path found!')
            return
            
        self.get_logger().info(f'A* path found with {len(path)} waypoints')
        self.publish_path(path, 'astar')
        
    # ==================== GVD PLANNER ====================
    
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
        
    def get_neighbors_gvd(self, node):
        """Get neighbors that are ONLY on skeleton"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.skeleton[ny, nx] == 1:
                    neighbors.append((nx, ny))
        return neighbors
        
    def astar_on_skeleton(self, start, goal):
        """A* search restricted to skeleton points only"""
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            
            if current == goal:
                break
                
            for next_node in self.get_neighbors_gvd(current):
                move_cost = self.diagonal_cost if abs(next_node[0] - current[0]) + abs(next_node[1] - current[1]) == 2 else self.straight_cost
                new_cost = cost_so_far[current] + move_cost
                
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(next_node, goal)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        if goal not in came_from:
            return None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
        
    def plan_gvd(self):
        """Plan path using GVD"""
        start_skeleton = self.find_nearest_skeleton_point(self.start)
        goal_skeleton = self.find_nearest_skeleton_point(self.goal)
        
        self.get_logger().info(f'Planning GVD path from skeleton {start_skeleton} to {goal_skeleton}')
        
        skeleton_path = self.astar_on_skeleton(start_skeleton, goal_skeleton)
        
        if skeleton_path is None:
            self.get_logger().error('GVD: No path found on skeleton!')
            return
            
        full_path = [self.start] + skeleton_path + [self.goal]
        
        self.get_logger().info(f'GVD path found with {len(full_path)} waypoints')
        self.publish_path(full_path, 'gvd')
    
    # ==================== PATH PUBLISHING ====================
    
    def publish_path(self, path, planner_name):
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
            if np.sqrt((x - last_x)**2 + (y - last_y)**2) > 0.01:
                pose = PoseStamped()
                pose.header.stamp = path_msg.header.stamp
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
        
        # Publish to appropriate topic
        if planner_name == 'astar':
            self.astar_path_pub.publish(path_msg)
            self.publish_path_marker(path, 'astar')
        else:  # gvd
            self.gvd_path_pub.publish(path_msg)
            self.publish_path_marker(path, 'gvd')
            
    def publish_path_marker(self, path, planner_name):
        """Publish path as visualization marker"""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        if planner_name == 'astar':
            marker.ns = 'astar_path'
            marker.id = 0
            marker.scale.x = self.astar_path_line_width
            marker.color.r = self.astar_path_color_r
            marker.color.g = self.astar_path_color_g
            marker.color.b = self.astar_path_color_b
        else:  # gvd
            marker.ns = 'gvd_path'
            marker.id = 1
            marker.scale.x = self.gvd_path_line_width
            marker.color.r = self.gvd_path_color_r
            marker.color.g = self.gvd_path_color_g
            marker.color.b = self.gvd_path_color_b
            
        marker.color.a = 1.0
        
        for grid_x, grid_y in path:
            x, y = self.grid_to_world(grid_x, grid_y)
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1
            marker.points.append(point)
        
        if planner_name == 'astar':
            self.astar_marker_pub.publish(marker)
        else:
            self.gvd_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    planner = PlannerServer()
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
