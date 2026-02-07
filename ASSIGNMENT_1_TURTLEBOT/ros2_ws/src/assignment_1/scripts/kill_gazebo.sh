#!/bin/bash
# Kill all Gazebo processes

echo "Killing Gazebo processes..."
pkill -9 gazebo
pkill -9 gzserver
pkill -9 gzclient

echo "Done! All Gazebo processes terminated."
