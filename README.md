# robotic_ultrasound_thesis
Robotic Ultrasound Project
Overview
This repository contains the source code and configurations for the robotic ultrasound system developed as part of a Master's thesis at Delft University of Technology. The project integrates various ROS packages to facilitate robotic control, vision processing, and impedance control for ultrasound-guided procedures.

Table of Contents
Project Structure

Installation Instructions

Usage

External Dependencies

License

Project Structure
The repository follows a standard ROS workspace layout:

css
Kopiëren
Bewerken
robotic_ultrasound_project/
├── src/
│   ├── robotic_ultrasound_transportation/
│   ├── robotic_ultrasound_vision/
│   └── ...
├── CMakeLists.txt
└── package.xml
robotic_ultrasound_transportation: Handles the transportation logic and interfaces with the robotic arm.

robotic_ultrasound_vision: Manages vision processing tasks, including image acquisition and processing.

Additional ROS packages as required for the project.

The iiwa_impedance_control package is maintained separately in its own repository, as detailed in the External Dependencies section.

Installation Instructions
Prerequisites
ROS Noetic

Catkin workspace setup

Steps
Clone the repository:

bash
Kopiëren
Bewerken
cd ~/catkin_ws/src
git clone https://github.com/toinek/robotic_ultrasound_project.git
Install external dependencies:

RealSense SDK 2.0:

Follow the installation instructions provided by Intel for the RealSense SDK 2.0.

KUKA FRI:

bash
Kopiëren
Bewerken
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone git@gitlab.tudelft.nl:nickymol/kuka_fri.git
cd kuka_fri
wget https://gist.githubusercontent.com/matthias-mayr/0f947982474c1865aab825bd084e7a92/raw/244f1193bd30051ae625c8f29ed241855a59ee38/0001-Config-Disables-SIMD-march-native-by-default.patch
git am 0001-Config-Disables-SIMD-march-native-by-default.patch
./waf configure
./waf
sudo ./waf install
iiwa_ros:

bash
Kopiëren
Bewerken
cd ~/catkin_ws/src
git clone https://github.com/epfl-lasa/iiwa_ros.git
realsense-ros:

bash
Kopiëren
Bewerken
cd ~/catkin_ws/src
git clone git@github.com:IntelRealSense/realsense-ros.git
Build the workspace:

bash
Kopiëren
Bewerken
cd ~/catkin_ws
catkin_make
Usage
Source the workspace:

bash
Kopiëren
Bewerken
source devel/setup.bash
Launch the desired ROS nodes:

For example, to launch the vision processing node:

bash
Kopiëren
Bewerken
roslaunch robotic_ultrasound_vision vision_node.launch
Replace vision_node.launch with the appropriate launch file for other functionalities.

External Dependencies
This project relies on several external repositories:

RealSense SDK 2.0: For depth sensing and camera integration.

KUKA FRI: For communication with the KUKA robot.

iiwa_ros: For interfacing with the KUKA iiwa robot.

realsense-ros: For RealSense camera integration in ROS.

Please refer to the respective repositories for detailed installation and setup instructions.

License
This project is licensed under the MIT License - see the LICENSE file for details.
