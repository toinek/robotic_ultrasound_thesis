# Robotic Ultrasound Thesis Project

This repository contains the code and setup instructions for the robotic ultrasound thesis project, utilizing a KUKA robot with the iiwa_ros package, impedance control, and RealSense SDK for advanced robotic manipulation and sensing.

## Prerequisites

- **ROS**: Ensure ROS (Robot Operating System) is installed (preferably ROS Noetic or compatible version).
- **Catkin Workspace**: A catkin workspace is required to build the project.
- **RealSense SDK 2.0**: Install the RealSense SDK by following the official [installation instructions](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md).
- **Git**: Required for cloning repositories.
- **Dependencies**: Ensure all dependencies for `iiwa_ros` and other packages are installed.

## Setup Instructions

Follow these steps to set up the project in a catkin workspace:

1. **Create a Catkin Workspace**  
   Create and navigate to a catkin workspace, then create a `src` folder:
   ```bash
   mkdir <WORKSPACE_NAME> && cd <WORKSPACE_NAME>
   mkdir src
   ```

2. **Install the KUKA FRI Repository**  
   Clone the KUKA FRI repository and apply a SIMD patch to disable `-march=native`:
   ```bash
   cd src
   git clone git@gitlab.tudelft.nl:nickymol/kuka_fri.git
   cd kuka_fri
   wget https://gist.githubusercontent.com/matthias-mayr/0f947982474c1865aab825bd084e7a92/raw/244f1193bd30051ae625c8f29ed241855a59ee38/0001-Config-Disables-SIMD-march-native-by-default.patch
   git am 0001-Config-Disables-SIMD-march-native-by-default.patch
   ./waf configure
   ./waf
   sudo ./waf install
   ```

   **Important**: Do **not** run `export CXXFLAGS="-march=native -faligned-new"`, as the SIMD patch disables `-march=native` to prevent segmentation faults.

3. **Install iiwa_ros Dependencies**  
   Install the dependencies for the `iiwa_ros` package, excluding the `kuka-fri` repository (already installed). Clone the `iiwa_ros` repository:
   ```bash
   cd src
   git clone https://github.com/epfl-lasa/iiwa_ros.git
   ```

4. **Install Impedance Controller**  
   Clone the impedance control repository and checkout the `spline_trajectory` branch, which removes the need for a specific end-effector:
   ```bash
   git clone git@gitlab.tudelft.nl:nickymol/iiwa_impedance_control.git
   cd iiwa_impedance_control
   git checkout spline_trajectory
   ```

5. **Clone the Thesis Repository**  
   Clone the main thesis project repository:
   ```bash
   git clone git@github.com:toinek/robotic_ultrasound_thesis.git
   ```

6. **Build the Workspace**  
   Navigate to the root of the catkin workspace and build:
   ```bash
   cd <WORKSPACE_NAME>
   catkin_make
   ```

7. **Source the Workspace**  
   Source the workspace to make the packages available:
   ```bash
   source devel/setup.bash
   ```

## Usage

- Ensure the RealSense camera is connected and properly configured.
- Run the necessary ROS nodes from the `iiwa_ros` and `robotic_ultrasound_thesis` packages.
- Use the impedance controller for tasks requiring compliant motion, leveraging the `spline_trajectory` branch for trajectory planning.
- Refer to the `robotic_ultrasound_thesis` repository for specific scripts and configurations tailored to ultrasound tasks.

## Notes

- The `spline_trajectory` branch in `iiwa_impedance_control` is specifically modified to eliminate the dependency on a particular end-effector, making it more flexible for various setups.
- If you encounter segmentation faults, double-check that `-march=native` is not enabled in any build configurations.
- For detailed documentation on the KUKA robot or impedance control, refer to the respective repositories (`kuka_fri`, `iiwa_ros`, `iiwa_impedance_control`).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [KUKA FRI Repository](https://gitlab.tudelft.nl/nickymol/kuka_fri)
- [iiwa_ros Package](https://github.com/epfl-lasa/iiwa_ros)
- [Impedance Control Repository](https://gitlab.tudelft.nl/nickymol/iiwa_impedance_control)
- Intel RealSense SDK Team
