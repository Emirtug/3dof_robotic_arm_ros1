#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MQTT Robot Controller - Robot Side
Receives position data via MQTT and moves the robot arm
"""

import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import json
import time
import math
import paho.mqtt.client as mqtt

# ============== CONFIGURATION ==============
# MQTT settings
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "robot/position"
MQTT_COMMAND_TOPIC = "robot/command"
MQTT_CLIENT_ID = f"robot_controller_{int(time.time())}"

# Robot joint limits (from URDF)
JOINT1_MIN = -2.268  # -130 deg
JOINT1_MAX = 2.268   # +130 deg
JOINT2_MIN = 0.0
JOINT2_MAX = 1.57    # 90 deg
JOINT3_MIN = 0.0
JOINT3_MAX = 1.57    # 90 deg

# ============ REFERENCE BASED CONTROL ============
# Center reference point (marker rest position)
CENTER_X = 0.0       # cm - marker center X
CENTER_Y = 0.0       # cm - marker center Y  
CENTER_Z = 15.0      # cm - marker default distance from camera

# Movement range around center (how much movement = full joint range)
# Smaller values = more sensitive, larger = less sensitive
RANGE_X = 10.0       # ±10cm from center = full joint1 range
RANGE_Y = 8.0        # ±8cm from center = full joint3 range
RANGE_Z = 10.0       # ±10cm from center (5-25cm) = full joint2 range

# Joint movement limits (how much the joint can move from home)
# This prevents the robot from moving too far
JOINT1_MOVE_LIMIT = 1.5   # rad (~86 deg each side)
JOINT2_MOVE_LIMIT = 0.6   # rad (~34 deg)
JOINT3_MOVE_LIMIT = 0.6   # rad (~34 deg)

# Direction inversion (1 = normal, -1 = inverted)
INVERT_X = -1  # Invert joint1 direction
INVERT_Y = -1  # Invert joint3 direction
INVERT_Z = -1  # Invert joint2 direction (closer = up)

# Home position (robot rest position)
HOME_POSITION = [0.0, 0.5, 0.5]  # joint1, joint2, joint3

# Safe position (when marker lost)
SAFE_POSITION = [0.0, 0.4, 0.4]

# Smoothing factor (0-1, lower = smoother but slower response)
SMOOTHING = 0.15

# Control rate (Hz) - lower = less vibration
CONTROL_RATE = 10

# Trajectory execution time (seconds) - higher = smoother movement
TRAJECTORY_TIME = 0.2

# Deadzone - ignore small position changes (cm)
DEADZONE = 0.3
# ===========================================


class MQTTRobotController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('mqtt_robot_controller', anonymous=True)
        
        # Joint trajectory publisher
        self.trajectory_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)
        
        # Joint names (must match URDF)
        self.joint_names = ['joint1', 'joint2', 'joint3']
        
        # Current and target positions
        self.current_joints = list(HOME_POSITION)
        self.target_joints = list(HOME_POSITION)
        
        # MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_connected = False
        
        # State
        self.last_message_time = time.time()
        self.message_count = 0
        self.enabled = True
        
        rospy.loginfo("MQTT Robot Controller initialized")
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            rospy.loginfo(f"Connected to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
            self.mqtt_connected = True
            # Subscribe to topics
            client.subscribe(MQTT_TOPIC)
            client.subscribe(MQTT_COMMAND_TOPIC)
            rospy.loginfo(f"Subscribed to: {MQTT_TOPIC}, {MQTT_COMMAND_TOPIC}")
        else:
            rospy.logerr(f"MQTT connection failed! Code: {rc}")
            
    def on_disconnect(self, client, userdata, flags, rc, properties=None):
        rospy.logwarn("Disconnected from MQTT broker!")
        self.mqtt_connected = False
        
    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            data = json.loads(msg.payload.decode())
            
            if msg.topic == MQTT_TOPIC:
                # Position message
                self.handle_position(data)
                
            elif msg.topic == MQTT_COMMAND_TOPIC:
                # Command message
                self.handle_command(data)
                
        except json.JSONDecodeError as e:
            rospy.logwarn(f"Invalid JSON: {e}")
        except Exception as e:
            rospy.logerr(f"Message handling error: {e}")
            
    def handle_position(self, data):
        """Process position data and update target joints"""
        if not self.enabled:
            return
            
        x = data.get('x', 0)
        y = data.get('y', 0)
        z = data.get('z', 20)
        
        # Apply deadzone - ignore very small changes
        if hasattr(self, 'last_x'):
            if abs(x - self.last_x) < DEADZONE:
                x = self.last_x
            if abs(y - self.last_y) < DEADZONE:
                y = self.last_y
            if abs(z - self.last_z) < DEADZONE:
                z = self.last_z
        
        self.last_x, self.last_y, self.last_z = x, y, z
        
        self.last_message_time = time.time()
        self.message_count += 1
        
        # ============ REFERENCE BASED MAPPING ============
        # Calculate offset from center reference point
        offset_x = x - CENTER_X
        offset_y = y - CENTER_Y
        offset_z = z - CENTER_Z
        
        # Normalize offset to -1...+1 range
        norm_x = self.clamp(offset_x / RANGE_X, -1.0, 1.0)
        norm_y = self.clamp(offset_y / RANGE_Y, -1.0, 1.0)
        norm_z = self.clamp(offset_z / RANGE_Z, -1.0, 1.0)
        
        # Apply direction inversion
        norm_x *= INVERT_X
        norm_y *= INVERT_Y
        norm_z *= INVERT_Z
        
        # Calculate joint positions as home + offset
        # Normalized value * move limit gives the movement from home
        joint1 = HOME_POSITION[0] + norm_x * JOINT1_MOVE_LIMIT
        joint2 = HOME_POSITION[1] + norm_z * JOINT2_MOVE_LIMIT
        joint3 = HOME_POSITION[2] + norm_y * JOINT3_MOVE_LIMIT
        
        # Clamp to physical limits
        joint1 = self.clamp(joint1, JOINT1_MIN, JOINT1_MAX)
        joint2 = self.clamp(joint2, JOINT2_MIN, JOINT2_MAX)
        joint3 = self.clamp(joint3, JOINT3_MIN, JOINT3_MAX)
        
        self.target_joints = [joint1, joint2, joint3]
        
        if self.message_count % 10 == 0:
            rospy.loginfo(f"RX #{self.message_count}: X:{x:.1f} Y:{y:.1f} Z:{z:.1f} | Off: X:{offset_x:.1f} Y:{offset_y:.1f} Z:{offset_z:.1f} -> J1:{math.degrees(joint1):.1f}° J2:{math.degrees(joint2):.1f}° J3:{math.degrees(joint3):.1f}°")
            
    def handle_command(self, data):
        """Process command messages"""
        command = data.get('command', '')
        
        if command == 'home':
            rospy.loginfo("Command: HOME - Moving to home position")
            self.target_joints = list(HOME_POSITION)
            
        elif command == 'safe':
            rospy.loginfo("Command: SAFE - Moving to safe position")
            self.target_joints = list(SAFE_POSITION)
            
        elif command == 'stop':
            rospy.loginfo("Command: STOP - Disabling control")
            self.enabled = False
            
        elif command == 'start':
            rospy.loginfo("Command: START - Enabling control")
            self.enabled = True
            
        elif command == 'zero':
            rospy.loginfo("Command: ZERO - Moving to zero position")
            self.target_joints = [0.0, 0.0, 0.0]
            
    def map_value(self, value, in_min, in_max, out_min, out_max):
        """Map a value from one range to another"""
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
    def clamp(self, value, min_val, max_val):
        """Clamp value to range"""
        return max(min_val, min(max_val, value))
        
    def smooth_move(self):
        """Apply smoothing to joint movements"""
        for i in range(3):
            diff = self.target_joints[i] - self.current_joints[i]
            self.current_joints[i] += diff * SMOOTHING
            
    def publish_joints(self):
        """Publish current joint positions to ROS via JointTrajectory"""
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self.current_joints
        point.velocities = [0.0, 0.0, 0.0]
        point.time_from_start = rospy.Duration(TRAJECTORY_TIME)
        
        traj.points.append(point)
        self.trajectory_pub.publish(traj)
        
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        rospy.loginfo(f"Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}...")
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            time.sleep(1)
            return self.mqtt_connected
        except Exception as e:
            rospy.logerr(f"MQTT connection error: {e}")
            return False
            
    def run(self):
        """Main control loop"""
        # Connect to MQTT
        if not self.connect_mqtt():
            rospy.logerr("Failed to connect to MQTT broker!")
            return
            
        rospy.loginfo("=" * 50)
        rospy.loginfo("  MQTT Robot Controller Active")
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"  Broker: {MQTT_BROKER}:{MQTT_PORT}")
        rospy.loginfo(f"  Topic: {MQTT_TOPIC}")
        rospy.loginfo(f"  Home: {[f'{math.degrees(j):.1f}°' for j in HOME_POSITION]}")
        rospy.loginfo("=" * 50)
        
        # Move to home position initially
        self.target_joints = list(HOME_POSITION)
        
        rate = rospy.Rate(CONTROL_RATE)  # Control loop rate
        
        while not rospy.is_shutdown():
            # Apply smoothing and publish
            self.smooth_move()
            self.publish_joints()
            
            rate.sleep()
            
        # Cleanup
        rospy.loginfo("Shutting down...")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()


if __name__ == '__main__':
    try:
        controller = MQTTRobotController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
