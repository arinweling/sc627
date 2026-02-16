#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
import numpy as np
from PIL import Image
import yaml
import os


class MapCorrector(Node):
    def __init__(self):
        super().__init__('map_corrector')
        
        # Declare parameters
        self.declare_parameter('map_yaml_file', '')  # Path to map YAML file
        self.declare_parameter('rotation_angle', 0.0)  # Rotation in degrees (0, 90, 180, 270)
        self.declare_parameter('flip_horizontal', False)
        self.declare_parameter('flip_vertical', False)
        self.declare_parameter('output_topic', '/map')  # Publish corrected map
        
        # Get parameters
        map_yaml_file = self.get_parameter('map_yaml_file').value
        self.rotation_angle = self.get_parameter('rotation_angle').value
        self.flip_horizontal = self.get_parameter('flip_horizontal').value
        self.flip_vertical = self.get_parameter('flip_vertical').value
        self.output_topic = self.get_parameter('output_topic').value
        
        if not map_yaml_file:
            self.get_logger().error('map_yaml_file parameter is required!')
            return
        
        # Publisher
        self.map_pub = self.create_publisher(OccupancyGrid, self.output_topic, 10)
        
        self.get_logger().info(f'Map Corrector initialized')
        self.get_logger().info(f'  Map file: {map_yaml_file}')
        self.get_logger().info(f'  Rotation: {self.rotation_angle}°')
        self.get_logger().info(f'  Flip H: {self.flip_horizontal}, Flip V: {self.flip_vertical}')
        self.get_logger().info(f'  Output topic: {self.output_topic}')
        
        # Load and publish map
        self.load_and_publish_map(map_yaml_file)
        
        # Create timer to republish map periodically (latched topic simulation)
        self.timer = self.create_timer(1.0, self.publish_map)
        
    def load_and_publish_map(self, yaml_file):
        """Load map from YAML file, correct orientation, and prepare for publishing"""
        try:
            # Read YAML file
            with open(yaml_file, 'r') as f:
                map_config = yaml.safe_load(f)
            
            # Get map directory for relative image path
            map_dir = os.path.dirname(yaml_file)
            image_file = os.path.join(map_dir, map_config['image'])
            
            # Load image
            img = Image.open(image_file)
            img_array = np.array(img)
            
            # IMPORTANT: PIL loads images with origin at top-left (image coordinates)
            # ROS maps have origin at bottom-left (map coordinates)
            # We must flip vertically to convert from image to map coordinates
            img_array = np.flipud(img_array)
            
            # Get parameters from YAML
            resolution = map_config.get('resolution', 0.05)
            origin = map_config.get('origin', [0.0, 0.0, 0.0])
            negate = map_config.get('negate', 0)
            occupied_thresh = map_config.get('occupied_thresh', 0.65)
            free_thresh = map_config.get('free_thresh', 0.25)
            
            # Convert image to occupancy grid values
            # Standard map convention (Cartographer):
            # - White (255) = free space → occupancy 0
            # - Black (0) = obstacles → occupancy 100
            # - Gray (intermediate) = unknown → occupancy 100
            
            # PGM values are 0-255, normalize to 0-1
            img_normalized = img_array.astype(float) / 255.0
            
            # Debug: print normalized value statistics
            self.get_logger().info(f'Image normalized stats:')
            self.get_logger().info(f'  Min: {np.min(img_normalized):.3f}, Max: {np.max(img_normalized):.3f}')
            self.get_logger().info(f'  Mean: {np.mean(img_normalized):.3f}, Median: {np.median(img_normalized):.3f}')
            unique_vals, counts = np.unique(img_normalized, return_counts=True)
            self.get_logger().info(f'  Unique values: {len(unique_vals)}')
            for val, count in zip(unique_vals[:10], counts[:10]):  # Print first 10
                self.get_logger().info(f'    {val:.3f}: {count} pixels')
            
            # Negate if needed (inverts white/black meaning)
            if negate:
                img_normalized = 1.0 - img_normalized
            
            # Convert to occupancy grid: -1 (unknown), 0 (free), 100 (occupied)
            # Cartographer uses specific pixel values:
            # - 254 (0.996) = free space
            # - 205 (0.804) = unknown
            # - 0 (0.000) = occupied
            # 
            # Use simple threshold approach for Cartographer maps:
            occupancy_grid = np.full_like(img_array, -1, dtype=np.int8)
            
            occupancy_grid[img_normalized > occupied_thresh] = 0      # Free (white, ~254)
            occupancy_grid[img_normalized < free_thresh] = 100    # Occupied (black, ~0)
            # Gray pixels (~205, 0.8) stay as -1 (unknown)
            
            self.get_logger().info(f'Occupancy grid stats:')
            unique, counts = np.unique(occupancy_grid, return_counts=True)
            for val, count in zip(unique, counts):
                label = {-1: 'unknown', 0: 'free', 100: 'occupied'}.get(val, 'other')
                self.get_logger().info(f'  {label} ({val}): {count} pixels')
            
            # Apply transformations
            corrected_map = self.correct_map_orientation(occupancy_grid)
            
            # Create OccupancyGrid message
            self.map_msg = OccupancyGrid()
            self.map_msg.header.frame_id = 'map'
            
            # Update dimensions if rotated by 90 or 270
            original_height, original_width = img_array.shape
            if self.rotation_angle in [90, 270]:
                self.map_msg.info.width = original_height
                self.map_msg.info.height = original_width
            else:
                self.map_msg.info.width = original_width
                self.map_msg.info.height = original_height
            
            self.map_msg.info.resolution = resolution
            
            # Set origin
            self.map_msg.info.origin = Pose()
            self.map_msg.info.origin.position.x = origin[0]
            self.map_msg.info.origin.position.y = origin[1]
            self.map_msg.info.origin.position.z = origin[2] if len(origin) > 2 else 0.0
            self.map_msg.info.origin.orientation.w = 1.0
            
            # Flatten and set data
            self.map_msg.data = corrected_map.flatten().tolist()
            
            self.get_logger().info(f'Map loaded successfully: {self.map_msg.info.width}x{self.map_msg.info.height}')
            self.get_logger().info(f'Resolution: {resolution}, Origin: {origin}')
            
        except Exception as e:
            self.get_logger().error(f'Error loading map: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def publish_map(self):
        """Publish the map periodically"""
        if hasattr(self, 'map_msg'):
            self.map_msg.header.stamp = self.get_clock().now().to_msg()
            self.map_pub.publish(self.map_msg)
    
    def correct_map_orientation(self, map_data):
        """Apply rotation and flipping to map data"""
        corrected = map_data.copy()
        
        # Apply horizontal flip
        if self.flip_horizontal:
            corrected = np.fliplr(corrected)
            
        # Apply vertical flip
        if self.flip_vertical:
            corrected = np.flipud(corrected)
            
        # Apply rotation (counter-clockwise)
        if self.rotation_angle == 90:
            corrected = np.rot90(corrected, k=1)  # 90° CCW
        elif self.rotation_angle == 180:
            corrected = np.rot90(corrected, k=2)  # 180°
        elif self.rotation_angle == 270:
            corrected = np.rot90(corrected, k=3)  # 270° CCW (90° CW)
            
        return corrected


def main(args=None):
    rclpy.init(args=args)
    corrector = MapCorrector()
    rclpy.spin(corrector)
    corrector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
