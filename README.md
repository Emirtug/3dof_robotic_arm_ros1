# 3-DOF RRR Robot Arm - TUBITAK 2209-B

## Overview

A 3-DOF RRR (Revolute-Revolute-Revolute) robot arm controlled via ArUco markers with ANN-based motion learning. The system runs in Gazebo simulation and on real hardware using an OpenCM9.04 microcontroller with AX-12A Dynamixel servos.

Key features:
- **ArUco marker tracking**: Camera-based position control using fiducial markers
- **ANN motion learning**: Artificial neural network for position smoothing, prediction, and vibration reduction
- **Dual-mode operation**: Gazebo simulation, real hardware, or both in parallel
- **MQTT communication**: Decoupled architecture for flexible deployment

---

## Architecture

```
                    +------------------+
                    |  ArUco Camera    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | aruco_mqtt_     |
                    | sender.py       |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   MQTT Broker    |
                    |   (mosquitto)    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
   +--------------------+       +------------------------+
   | mqtt_robot_ann_    |       | real_robot_controller   |
   | controller.py      |       | .py                     |
   +---------+----------+       +----------+--------------+
             |                             |
             v                             v
   +--------------------+       +------------------------+
   | Gazebo Simulation  |       | serial_robot_bridge.py  |
   | (arm_controller)   |       +----------+--------------+
   +--------------------+                  |
                                            v
                                +------------------------+
                                | OpenCM9.04 (firmware)  |
                                +----------+-------------+
                                           |
                                           v
                                +------------------------+
                                | 3x AX-12A Dynamixel     |
                                | Servos (TTL daisy-chain)|
                                +------------------------+
```

---

## Project Structure

```
3dof_rrr_robot_arm/
|
|-- CMakeLists.txt              # Catkin build configuration
|-- package.xml                 # ROS package dependencies
|-- README.md                   # This file
|
|-- config/
|   |-- robot_params.yaml       # Joint limits, hardware config, control rates
|   +-- controllers/
|       |-- position_controller.yaml
|       |-- velocity_controller.yaml
|       +-- effort_controller.yaml
|
|-- launch/
|   +-- gazebo.launch           # Spawns Gazebo with robot, controllers, robot_state_publisher
|
|-- urdf/
|   |-- robot.urdf.xacro        # Main robot model (xacro with hardware interface)
|   +-- orijinal.urdf           # Original URDF reference
|
|-- meshes/                     # 3D meshes for Gazebo visualization
|
|-- scripts/
|   |-- mqtt_robot_ann_controller.py   # Gazebo controller: ANN, smoothing, recording, playback
|   |-- aruco_mqtt_sender.py           # Reads ArUco from camera, publishes to MQTT
|   |-- real_robot_controller.py       # Real robot: MQTT to serial bridge
|   |-- serial_robot_bridge.py         # USB serial protocol for OpenCM9.04
|   |-- motion_recorder.py             # ROS bag recording/playback, ANN training data export
|   |-- ann_motion_predictor.py        # ANN model for smoothing and prediction
|   +-- robot_control_opencm9.04/
|       +-- robot_control_opencm9.04.ino   # OpenCM9.04 firmware (Arduino)
|
+-- recordings/                 # Recorded motions and exports for ANN training
    +-- exports/
```

---

## Prerequisites

| Component | Purpose |
|-----------|---------|
| ROS Noetic | Robot Operating System |
| Gazebo | Simulation environment |
| Python 3 | Scripts and controllers |
| OpenCV | ArUco marker detection |
| paho-mqtt | MQTT client |
| pyserial | Serial communication (real robot) |
| pynput | Keyboard input for controllers |
| TensorFlow / Keras | Optional; ANN training and inference |

Install MQTT broker:
```bash
sudo apt install mosquitto mosquitto-clients
```

---

## Quick Start

### Gazebo Only

```bash
# Terminal 1: Launch Gazebo and robot
roslaunch 3dof_rrr_robot_arm gazebo.launch

# Terminal 2: Start MQTT broker
mosquitto -d

# Terminal 3: Gazebo controller
rosrun 3dof_rrr_robot_arm mqtt_robot_ann_controller.py

# Terminal 4: ArUco sender (camera)
rosrun 3dof_rrr_robot_arm aruco_mqtt_sender.py
```

### Real Robot + Gazebo (Parallel)

Both simulation and real robot receive the same ArUco position stream.

```bash
# Terminal 1: Launch Gazebo
roslaunch 3dof_rrr_robot_arm gazebo.launch

# Terminal 2: MQTT broker
mosquitto -d

# Terminal 3: Gazebo controller
rosrun 3dof_rrr_robot_arm mqtt_robot_ann_controller.py

# Terminal 4: Real robot controller (OpenCM9.04 must be connected)
python3 scripts/real_robot_controller.py

# Terminal 5: ArUco sender
rosrun 3dof_rrr_robot_arm aruco_mqtt_sender.py
```

### Real Robot Only

No ROS required. Run from package root (`3dof_rrr_robot_arm/`).

```bash
# Terminal 1: MQTT broker
mosquitto -d

# Terminal 2: Real robot controller
python3 scripts/real_robot_controller.py

# Terminal 3: ArUco sender
python3 scripts/aruco_mqtt_sender.py
```

---

## Keyboard Controls

### Gazebo Controller (mqtt_robot_ann_controller.py)

| Key | Action |
|-----|--------|
| A | Toggle ANN (On/Off) |
| S | Toggle smoothing |
| P | Toggle prediction |
| R | Start/stop recording |
| Y | Playback recording |
| T | Train ANN from recordings |
| M | Show metrics/comparison |
| L | Soldering effect |
| Q | Quit |

### Real Robot Controller (real_robot_controller.py)

| Key | Action |
|-----|--------|
| S | Start/stop |
| H | Home position |
| F | Safe position |
| E | E-STOP (emergency stop) |
| R | Relax (torque disable) |
| I | Status |
| Q | Quit |

---

## ANN Training Workflow

1. Press **[R]** to start recording ArUco-driven motions
2. Move the ArUco marker; motions are logged
3. Press **[R]** again to stop
4. Press **[T]** to train the ANN on recorded data
5. Press **[A]** to enable ANN mode (smoothing and prediction)

---

## Hardware

| Item | Description |
|------|-------------|
| Microcontroller | OpenCM9.04 |
| Servos | 3x AX-12A Dynamixel |
| Connection | TTL daisy-chain (IDs: 1, 2, 3) |
| Baudrate | 1,000,000 |
| Interface | USB serial (e.g. `/dev/ttyACM0`) |

Flash the firmware before using the real robot:
1. Open `scripts/robot_control_opencm9.04/robot_control_opencm9.04.ino` in Arduino IDE
2. Select board: OpenCM9.04
3. Upload to the board

---

## Author

Emirtug Kacar
