#!/usr/bin/env python3
"""
MQTT Dual Controller - Gazebo + Arduino Parallel Control
Controls both Gazebo simulation and real robot via Arduino simultaneously
Press SPACE for soldering effect in Gazebo

Features:
- Parallel control of Gazebo and real hardware
- Space key triggers soldering spark effect
- Visual feedback in both simulation and terminal
"""

import json
import time
import math
import signal
import sys
import threading
import serial
import serial.tools.list_ports
import paho.mqtt.client as mqtt

# ROS imports
import rospy
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import LinkStates

# Keyboard input
try:
    import pynput
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("[WARN] pynput not installed. Space key disabled.")
    print("       Install with: pip install pynput")

# ============ MQTT SETTINGS ============
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "robot/position"
MQTT_COMMAND_TOPIC = "robot/command"
MQTT_CLIENT_ID = f"robot_dual_controller_{int(time.time())}"

# ============ SERIAL SETTINGS ============
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 0.1
SERIAL_PORT = None  # Auto-detect

# ============ SAFETY SETTINGS ============
HEARTBEAT_INTERVAL = 0.5
ARDUINO_TIMEOUT = 3.0

# ============ JOINT LIMITS (radians) ============
JOINT1_MIN = -3.14
JOINT1_MAX = 3.14
JOINT2_MIN = 0.0
JOINT2_MAX = 1.57
JOINT3_MIN = 0.0
JOINT3_MAX = 1.57

# ============ REFERENCE BASED CONTROL ============
CENTER_X = 0.0
CENTER_Y = 0.0
CENTER_Z = 15.0

RANGE_X = 10.0
RANGE_Y = 8.0
RANGE_Z = 10.0

JOINT1_MOVE_LIMIT = 1.5
JOINT2_MOVE_LIMIT = 0.6
JOINT3_MOVE_LIMIT = 0.6

INVERT_X = -1
INVERT_Y = -1
INVERT_Z = -1

HOME_POSITION = [0.0, 0.5, 0.5]
SAFE_POSITION = [0.0, 0.3, 0.3]

SMOOTHING = 0.15
CONTROL_RATE = 10
DEADZONE = 0.3

# ============ SOLDERING EFFECT SETTINGS ============
SOLDER_DURATION = 0.3  # How long the spark lasts (seconds)
SOLDER_COLOR = [1.0, 0.8, 0.0, 1.0]  # Orange-yellow (RGBA)
SOLDER_SPARK_COLOR = [1.0, 1.0, 1.0, 1.0]  # White spark

# Spark particle SDF template
SPARK_SDF = '''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="solder_spark">
    <static>true</static>
    <link name="spark_link">
      <visual name="spark_visual">
        <geometry>
          <sphere>
            <radius>0.008</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>1 0.8 0 1</ambient>
          <diffuse>1 0.9 0.2 1</diffuse>
          <emissive>1 0.7 0 1</emissive>
        </material>
      </visual>
      <visual name="glow_visual">
        <geometry>
          <sphere>
            <radius>0.015</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>1 0.5 0 0.5</ambient>
          <diffuse>1 0.6 0.1 0.3</diffuse>
          <emissive>1 0.5 0 0.8</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''


class DualController:
    def __init__(self):
        print("=" * 60)
        print("  DUAL CONTROLLER - Gazebo + Arduino")
        print("  Press 'L' for soldering effect!")
        print("=" * 60)
        
        # Initialize ROS node
        rospy.init_node('mqtt_dual_controller', anonymous=True)
        
        # ========== GAZEBO SETUP ==========
        # Joint trajectory publisher for Gazebo
        self.trajectory_pub = rospy.Publisher(
            '/arm_controller/command', 
            JointTrajectory, 
            queue_size=10
        )
        
        # Marker publisher for RViz soldering effect
        self.marker_pub = rospy.Publisher(
            '/soldering_marker',
            Marker,
            queue_size=10
        )
        
        # Gazebo model services for spark effect
        rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
        rospy.wait_for_service('/gazebo/delete_model', timeout=5.0)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        
        # Subscribe to link states to get end effector (tool_tip) position
        self.end_effector_pos = [0.0, 0.0, 0.0]  # Will be updated from Gazebo
        self.end_effector_orientation = [0.0, 0.0, 0.0, 1.0]  # quaternion
        self.link_states_received = False
        self.link_states_sub = rospy.Subscriber(
            '/gazebo/link_states',
            LinkStates,
            self.link_states_callback
        )
        
        # TF buffer for getting transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.joint_names = ['joint1', 'joint2', 'joint3']
        
        # ========== ARDUINO SETUP ==========
        self.serial_port = None
        self.serial_connected = False
        self.serial_lock = threading.Lock()
        self.last_arduino_response = time.time()
        
        # ========== MQTT SETUP ==========
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        self.mqtt_connected = False
        
        # ========== STATE ==========
        self.current_joints = list(HOME_POSITION)
        self.target_joints = list(HOME_POSITION)
        self.last_x, self.last_y, self.last_z = 0.0, 0.0, CENTER_Z
        
        self.last_message_time = time.time()
        self.message_count = 0
        self.enabled = True
        self.running = True
        self.emergency_stop = False
        self.shutdown_in_progress = False
        
        # Soldering state
        self.soldering_active = False
        self.solder_start_time = 0
        self.spark_spawned = False
        self.solder_count = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Setup keyboard listener
        if KEYBOARD_AVAILABLE:
            self.keyboard_listener = keyboard.Listener(
                on_press=self.on_key_press,
                on_release=self.on_key_release
            )
            self.keyboard_listener.start()
            print("[KEYBOARD] Listener started - Press 'L' to solder!")
        
        rospy.loginfo("Dual Controller initialized")
        
    def link_states_callback(self, msg):
        """Get end effector (tool_tip) position from Gazebo"""
        try:
            # Debug: Log available links once
            if not hasattr(self, '_links_logged'):
                rospy.loginfo(f"Available Gazebo links: {msg.name}")
                self._links_logged = True
            
            # Search for tool_tip, gripper_link, or link3 in order of preference
            search_links = ['tool_tip', 'gripper_link', 'link3']
            
            for search_name in search_links:
                for i, name in enumerate(msg.name):
                    # Check if the link name contains our target (handles namespace prefixes)
                    if search_name in name:
                        pos = msg.pose[i].position
                        ori = msg.pose[i].orientation
                        self.end_effector_pos = [pos.x, pos.y, pos.z]
                        self.end_effector_orientation = [ori.x, ori.y, ori.z, ori.w]
                        self.link_states_received = True
                        return
        except Exception as e:
            rospy.logwarn_throttle(5, f"Link states error: {e}")
            
    def on_key_press(self, key):
        """Handle key press events"""
        try:
            # Use 'L' key for soldering (SPACE interferes with input)
            if hasattr(key, 'char') and key.char == 'l':
                self.start_soldering()
        except:
            pass
            
    def on_key_release(self, key):
        """Handle key release events"""
        try:
            if hasattr(key, 'char') and key.char == 'l':
                self.stop_soldering()
        except:
            pass
            
    def start_soldering(self):
        """Start soldering effect"""
        if self.soldering_active:
            return
            
        self.soldering_active = True
        self.solder_start_time = time.time()
        self.solder_count += 1
        
        print(f"\n🔥 SOLDERING #{self.solder_count} - ACTIVE 🔥")
        
        # Spawn spark in Gazebo
        self.spawn_spark()
        
        # Publish RViz marker
        self.publish_solder_marker(True)
        
    def stop_soldering(self):
        """Stop soldering effect"""
        if not self.soldering_active:
            return
            
        self.soldering_active = False
        duration = time.time() - self.solder_start_time
        
        print(f"✓ Solder #{self.solder_count} complete ({duration:.2f}s)")
        
        # Remove spark from Gazebo
        self.remove_spark()
        
        # Remove RViz marker
        self.publish_solder_marker(False)
        
    def spawn_spark(self):
        """Spawn soldering spark effect in Gazebo at end effector"""
        if self.spark_spawned:
            self.remove_spark()
            
        try:
            # Spawn spark attached to link3 with offset to tool_tip position
            # tool_tip offset from gripper_link: xyz="-0.05 0.06 0.2"
            pose = Pose()
            pose.position.x = -0.05  # Offset towards tool tip
            pose.position.y = 0.06
            pose.position.z = 0.25   # link3 to gripper + tool_tip offset
            pose.orientation.w = 1.0
            
            # Try to attach to link3 (Gazebo doesn't show virtual links)
            try:
                self.spawn_model(
                    model_name='solder_spark',
                    model_xml=SPARK_SDF,
                    robot_namespace='',
                    initial_pose=pose,
                    reference_frame='3dof_rrr_robot_arm::link3'
                )
                self.spark_spawned = True
                rospy.loginfo("Spark spawned attached to link3")
                return
            except Exception as e:
                rospy.logwarn(f"Could not attach to link3: {e}")
            
            # Fallback: spawn at current end effector world position
            if self.link_states_received:
                pose.position.x = self.end_effector_pos[0]
                pose.position.y = self.end_effector_pos[1]
                pose.position.z = self.end_effector_pos[2]
                
                self.spawn_model(
                    model_name='solder_spark',
                    model_xml=SPARK_SDF,
                    robot_namespace='',
                    initial_pose=pose,
                    reference_frame='world'
                )
                self.spark_spawned = True
                rospy.loginfo("Spark spawned at world position (fallback)")
            
        except Exception as e:
            rospy.logwarn(f"Failed to spawn spark: {e}")
            
    def remove_spark(self):
        """Remove soldering spark from Gazebo"""
        if not self.spark_spawned:
            return
            
        try:
            self.delete_model(model_name='solder_spark')
            self.spark_spawned = False
        except:
            pass
            
    def publish_solder_marker(self, active=True):
        """Publish RViz marker for soldering effect at tool_tip"""
        marker = Marker()
        marker.header.frame_id = "link3"  # Attached to link3 frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "soldering"
        marker.id = 0
        marker.type = Marker.SPHERE
        
        if active:
            marker.action = Marker.ADD
            marker.pose.position.x = -0.05  # Offset to tool_tip
            marker.pose.position.y = 0.06
            marker.pose.position.z = 0.25
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.color.r = 1.0
            marker.color.g = 0.7
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0)  # Persistent
        else:
            marker.action = Marker.DELETE
            
        self.marker_pub.publish(marker)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        if self.shutdown_in_progress:
            print("\n[WARN] Force quit!")
            sys.exit(1)
            
        self.shutdown_in_progress = True
        print(f"\n[SAFETY] Initiating safe shutdown...")
        self.running = False
        self.enabled = False
        self.safe_shutdown()
        
    def safe_shutdown(self):
        """Perform safe shutdown sequence"""
        print("[SAFETY] Moving to safe position...")
        
        # Remove any active spark
        self.remove_spark()
        
        # Send safe to Arduino
        for _ in range(3):
            if self.send_arduino_command('safe'):
                break
            time.sleep(0.2)
            
        # Send safe to Gazebo
        self.send_gazebo_trajectory(SAFE_POSITION)
        
        time.sleep(0.5)
        self.send_arduino_command('relax')
        
        print("[SAFETY] Robot in safe position")
        
    # ========== ARDUINO FUNCTIONS ==========
    def find_opencm_port(self):
        """Auto-detect OpenCM9.04 serial port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "ACM" in port.device or "USB" in port.device:
                return port.device
            if port.vid == 0x0483 and port.pid == 0x5740:
                return port.device
        return None
        
    def connect_serial(self):
        """Connect to OpenCM9.04 via Serial"""
        port = SERIAL_PORT if SERIAL_PORT else self.find_opencm_port()
        
        if not port:
            print("[SERIAL] OpenCM9.04 not found - running Gazebo only mode")
            return False
            
        try:
            with self.serial_lock:
                self.serial_port = serial.Serial(
                    port=port,
                    baudrate=SERIAL_BAUDRATE,
                    timeout=SERIAL_TIMEOUT
                )
                time.sleep(2)
                
            self.serial_connected = True
            self.last_arduino_response = time.time()
            print(f"[SERIAL] Connected to {port}")
            
            time.sleep(0.5)
            self.send_arduino_command('start')
            return True
            
        except Exception as e:
            print(f"[ERROR] Serial connection failed: {e}")
            return False
            
    def send_arduino_joints(self, joints):
        """Send joint positions to Arduino"""
        if not self.serial_connected or not self.serial_port:
            return False
            
        try:
            with self.serial_lock:
                cmd = f"J:{joints[0]:.4f},{joints[1]:.4f},{joints[2]:.4f}\n"
                self.serial_port.write(cmd.encode())
            return True
        except:
            self.serial_connected = False
            return False
            
    def send_arduino_command(self, command):
        """Send command to Arduino"""
        if not self.serial_port:
            return False
            
        try:
            with self.serial_lock:
                self.serial_port.write(f"C:{command}\n".encode())
            return True
        except:
            return False
            
    def send_arduino_heartbeat(self):
        """Send heartbeat to Arduino"""
        if not self.serial_connected:
            return
        try:
            with self.serial_lock:
                self.serial_port.write(b"H:\n")
        except:
            pass
            
    def read_arduino_responses(self):
        """Read responses from Arduino"""
        if not self.serial_connected or not self.serial_port:
            return
            
        try:
            with self.serial_lock:
                while self.serial_port.in_waiting > 0:
                    response = self.serial_port.readline().decode().strip()
                    if response:
                        self.last_arduino_response = time.time()
                        if "[ERR]" in response or "[WARN]" in response:
                            print(f"[ARDUINO] {response}")
        except:
            pass
            
    # ========== GAZEBO FUNCTIONS ==========
    def send_gazebo_trajectory(self, joints):
        """Send joint trajectory to Gazebo"""
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = joints
        point.time_from_start = rospy.Duration(0.2)
        
        msg.points = [point]
        self.trajectory_pub.publish(msg)
        
    # ========== MQTT FUNCTIONS ==========
    def on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[MQTT] Connected to {MQTT_BROKER}")
            self.mqtt_connected = True
            client.subscribe(MQTT_TOPIC)
            client.subscribe(MQTT_COMMAND_TOPIC)
        else:
            print(f"[ERROR] MQTT connection failed: {rc}")
            
    def on_mqtt_disconnect(self, client, userdata, flags, rc, properties=None):
        print("[MQTT] Disconnected!")
        self.mqtt_connected = False
        if not self.shutdown_in_progress:
            self.send_arduino_command('safe')
            self.send_gazebo_trajectory(SAFE_POSITION)
        
    def on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        if self.emergency_stop:
            return
            
        try:
            data = json.loads(msg.payload.decode())
            
            if msg.topic == MQTT_TOPIC:
                self.handle_position(data)
            elif msg.topic == MQTT_COMMAND_TOPIC:
                self.handle_command(data)
                
        except Exception as e:
            rospy.logwarn(f"Message error: {e}")
            
    def handle_position(self, data):
        """Process position data"""
        if not self.enabled:
            return
            
        x = data.get('x', 0)
        y = data.get('y', 0)
        z = data.get('z', CENTER_Z)
        
        # Deadzone
        if abs(x - self.last_x) < DEADZONE:
            x = self.last_x
        if abs(y - self.last_y) < DEADZONE:
            y = self.last_y
        if abs(z - self.last_z) < DEADZONE:
            z = self.last_z
            
        self.last_x, self.last_y, self.last_z = x, y, z
        self.last_message_time = time.time()
        self.message_count += 1
        
        # Reference based mapping
        offset_x = x - CENTER_X
        offset_y = y - CENTER_Y
        offset_z = z - CENTER_Z
        
        norm_x = max(-1.0, min(1.0, offset_x / RANGE_X))
        norm_y = max(-1.0, min(1.0, offset_y / RANGE_Y))
        norm_z = max(-1.0, min(1.0, offset_z / RANGE_Z))
        
        norm_x *= INVERT_X
        norm_y *= INVERT_Y
        norm_z *= INVERT_Z
        
        joint1 = HOME_POSITION[0] + norm_x * JOINT1_MOVE_LIMIT
        joint2 = HOME_POSITION[1] + norm_z * JOINT2_MOVE_LIMIT
        joint3 = HOME_POSITION[2] + norm_y * JOINT3_MOVE_LIMIT
        
        joint1 = max(JOINT1_MIN, min(JOINT1_MAX, joint1))
        joint2 = max(JOINT2_MIN, min(JOINT2_MAX, joint2))
        joint3 = max(JOINT3_MIN, min(JOINT3_MAX, joint3))
        
        self.target_joints = [joint1, joint2, joint3]
        
        if self.message_count % 20 == 0:
            solder_status = "welding active" if self.soldering_active else ""
            print(f"[RX] X:{x:.1f} Y:{y:.1f} Z:{z:.1f} -> "
                  f"J1:{math.degrees(joint1):.0f}° J2:{math.degrees(joint2):.0f}° J3:{math.degrees(joint3):.0f}° {solder_status}")
            
    def handle_command(self, data):
        """Process command messages"""
        command = data.get('command', '')
        
        if command == 'home':
            self.target_joints = list(HOME_POSITION)
            self.send_arduino_command('home')
        elif command == 'safe':
            self.target_joints = list(SAFE_POSITION)
            self.send_arduino_command('safe')
        elif command == 'stop':
            self.enabled = False
            self.send_arduino_command('stop')
        elif command == 'start':
            self.enabled = True
            self.send_arduino_command('start')
        elif command == 'solder':
            # Remote solder trigger
            self.start_soldering()
            threading.Timer(SOLDER_DURATION, self.stop_soldering).start()
            
    def control_loop(self):
        """Main control loop"""
        rate = 1.0 / CONTROL_RATE
        last_heartbeat = time.time()
        
        while self.running and not rospy.is_shutdown():
            try:
                # Heartbeat
                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    self.send_arduino_heartbeat()
                    last_heartbeat = time.time()
                    
                # Read Arduino
                self.read_arduino_responses()
                
                # Apply smoothing
                if not self.emergency_stop and self.enabled:
                    for i in range(3):
                        self.current_joints[i] += (self.target_joints[i] - self.current_joints[i]) * SMOOTHING
                        
                    # Send to BOTH Gazebo and Arduino
                    self.send_gazebo_trajectory(self.current_joints)
                    self.send_arduino_joints(self.current_joints)
                    
                # Update spark position if soldering and not attached to tool_tip
                # If spark is attached to tool_tip frame, it moves automatically
                # Only respawn if we're using world frame fallback
                if self.soldering_active and self.spark_spawned:
                    # Respawn every 0.3 seconds to update position (for world frame)
                    if time.time() - self.solder_start_time > 0.3:
                        # Only respawn if link_states show significant movement
                        self.remove_spark()
                        self.spawn_spark()
                        self.solder_start_time = time.time()
                        
                time.sleep(rate)
                
            except Exception as e:
                rospy.logerr(f"Control loop error: {e}")
                time.sleep(0.5)
                
    def run(self):
        """Main entry point"""
        # Connect to Arduino (optional)
        self.connect_serial()
        
        # Connect to MQTT
        try:
            print(f"[MQTT] Connecting to {MQTT_BROKER}...")
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"[ERROR] MQTT failed: {e}")
            return
            
        print("\n" + "=" * 60)
        print("  DUAL CONTROLLER RUNNING")
        print("  ─────────────────────────────────────────────")
        print("  Gazebo:  ✓ Active")
        print(f"  Arduino: {'✓ Connected' if self.serial_connected else '✗ Not connected'}")
        print("  ─────────────────────────────────────────────")
        print("  Controls:")
        print("    L key  = Trigger soldering effect (hold)")
        print("    Ctrl+C = Safe shutdown")
        print("=" * 60 + "\n")
        
        try:
            self.control_loop()
        finally:
            if not self.shutdown_in_progress:
                self.safe_shutdown()
                
            self.running = False
            
            if KEYBOARD_AVAILABLE:
                self.keyboard_listener.stop()
                
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
            if self.serial_port:
                try:
                    self.serial_port.close()
                except:
                    pass
                    
            print("[INFO] Controller stopped.")


if __name__ == "__main__":
    controller = DualController()
    controller.run()
