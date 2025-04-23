cmake_minimum_required(VERSION 3.0.2)
project(realsense)

find_package(catkin REQUIRED COMPONENTS
  realsense2_camera
  realsense2_description
  ddynamic_reconfigure
  roscpp
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})