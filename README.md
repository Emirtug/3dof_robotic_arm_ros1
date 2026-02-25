# 3DoF RRR Robot Arm

A 3-joint robot arm for ROS1 Noetic. Runs in Gazebo simulation, real robot uses OpenCM9.04 board with AX-12A servos.

## What you need

```bash
sudo apt install ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-controller ros-noetic-position-controllers \
    ros-noetic-joint-trajectory-controller
```

Also need Dynamixel SDK for real robot:
```bash
pip install dynamixel-sdk
```

## Build

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## Run Gazebo simulation

```bash
# Terminal 1 - start gazebo
roslaunch 3dof_rrr_robot_arm gazebo.launch

# Terminal 2 - control the robot
rosrun 3dof_rrr_robot_arm basic_move.py
```

Commands in basic_move.py:
- 1: Home position
- 2-4: Different poses
- 5: Wave motion
- 6: Sinusoidal motion
- q: Quit

## Hardware

- OpenCM9.04 expansion board
- 3x AX-12A Dynamixel servos
- 12V power supply

## Files

```
├── config/gazebo_controllers.yaml  # PID gains
├── launch/gazebo.launch            # Gazebo startup
├── scripts/basic_move.py           # Control script
├── meshes/                         # STL files
└── urdf/                           # Robot description
```

## Notes

Position control with PID. Controller runs at 50Hz.
