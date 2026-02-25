# 3DoF RRR Robot Arm

3-joint robot arm for ROS1 Noetic with Gazebo simulation. 
Real robot uses OpenCM9.04 + AX-12A servos.

## Setup

```bash
sudo apt install ros-noetic-gazebo-ros-control \
    ros-noetic-joint-state-controller ros-noetic-position-controllers \
    ros-noetic-joint-trajectory-controller ros-noetic-xacro

cd ~/catkin_ws && catkin_make
source devel/setup.bash
```

## Run

```bash
# Position control (default)
roslaunch 3dof_rrr_robot_arm gazebo.launch

# Velocity control
roslaunch 3dof_rrr_robot_arm gazebo.launch controller:=velocity

# Torque control
roslaunch 3dof_rrr_robot_arm gazebo.launch controller:=effort
```

Control the robot:
```bash
rosrun 3dof_rrr_robot_arm basic_move.py
```

## Structure

```
├── config/
│   ├── robot_params.yaml           # Robot physical params
│   └── controllers/
│       ├── position_controller.yaml
│       ├── velocity_controller.yaml
│       └── effort_controller.yaml
├── launch/
│   └── gazebo.launch
├── scripts/
│   └── basic_move.py
├── urdf/
│   └── robot.urdf.xacro            # Robot description (xacro)
└── meshes/                          # STL files
```

## Controller Selection

Just change the `controller` argument - transmission hardware interface updates automatically via xacro.

| Controller | Hardware Interface | Use Case |
|------------|-------------------|----------|
| position | PositionJointInterface | Go to angle X |
| velocity | VelocityJointInterface | Spin at speed Y |
| effort | EffortJointInterface | Apply torque Z |

## Hardware

- OpenCM9.04 expansion board
- 3x AX-12A Dynamixel servos
- Motor IDs: 1, 2, 3
