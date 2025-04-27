#!/bin/bash

source ~/catkin_ws/src/grounding_env/bin/activate

source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

rosrun grounding_dino_ros grounding_dino_node.py
