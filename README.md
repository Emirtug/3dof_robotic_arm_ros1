# 3DoF RRR Robot Arm

A simple 3-joint robot arm package for ROS1 Noetic. Originally had some ROS2 stuff mixed in, now it's cleaned up and works properly with Noetic.

Uses Dynamixel servos for the real robot, and you can also run it in Gazebo if you just want to test things out.

## What's in the box

- MoveIt for motion planning (uses OMPL/RRTConnect)
- Gazebo simulation with ros_control
- Real hardware support via Dynamixel SDK
- Position control with PID tuning

## Hardware setup

- 3x Dynamixel servos (IDs: 1, 2, 3)
- U2D2 or USB2Dynamixel adapter
- Runs at 57600 baud

## Getting started

```bash
# grab the repo
cd ~/catkin_ws/src
git clone https://github.com/YOUR_USERNAME/3dof_rrr_robot_arm.git

# install the ros stuff you need
sudo apt install ros-noetic-moveit ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-controller ros-noetic-position-controllers \
    ros-noetic-joint-trajectory-controller

# build it
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## How to run

### Option 1: Just MoveIt + RViz (no physics, just visualization)
```bash
roslaunch 3dof_rrr_robot_arm demo.launch
```
Good for testing motion planning without worrying about physics.

### Option 2: Gazebo simulation
```bash
# Terminal 1 - start gazebo
roslaunch 3dof_rrr_robot_arm gazebo.launch

# Terminal 2 - run the control script
rosrun 3dof_rrr_robot_arm basic_move.py
```
Type 1-6 to move the robot around, q to quit.

### Option 3: Real robot with Dynamixel motors
```bash
# first, make sure you have permission to access the usb port
sudo chmod 666 /dev/ttyACM0

# install dynamixel sdk if you haven't
pip install dynamixel-sdk

# Terminal 1 - start the controller
roslaunch 3dof_rrr_robot_arm real_robot.launch

# Terminal 2 - send commands
rostopic pub /robot_command std_msgs/Float64MultiArray "data: [0.5, 0.3, 0.2]"
```

## Quick start order

1. Source your workspace: `source ~/catkin_ws/devel/setup.bash`
2. Pick your mode (simulation or real)
3. Launch it
4. Control it

## Folder structure

```
3dof_rrr_robot_arm/
├── config/           # all the yaml configs
├── launch/           # launch files
├── meshes/           # STL files for visualization
├── scripts/          # python control scripts
├── urdf/             # robot description files
├── CMakeLists.txt
└── package.xml
```

## Joint limits

| Joint | Min | Max | Speed |
|-------|-----|-----|-------|
| joint1 | -130° | 130° | 1 rad/s |
| joint2 | 0° | 90° | 1 rad/s |
| joint3 | -90° | 90° | 1 rad/s |

## Troubleshooting

**Robot not moving in Gazebo?**
- Make sure the controllers loaded: `rostopic list | grep arm_controller`
- Check for errors in the gazebo terminal

**Can't connect to Dynamixel?**
- Check USB port: `ls /dev/ttyACM*`
- Set permissions: `sudo chmod 666 /dev/ttyACM0`
- Verify motor IDs match config

**MoveIt planning fails?**
- Check if robot_description is loaded: `rosparam get /robot_description`

## Notes

This package was converted from a mixed ROS1/ROS2 setup to pure ROS1 Noetic. If something's broken, check the launch files first - that's usually where things go wrong.

## License

MIT

## Author

Emirtug
