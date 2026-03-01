#!/usr/bin/env python3.9
"""
MQTT to Arduino (OpenCM9.04) Controller - WITH SAFETY FEATURES
Receives position data from MQTT and sends to OpenCM9.04 via Serial

Safety Features:
- Graceful shutdown on Ctrl+C, SIGTERM
- Safe position sent before exit
- Heartbeat monitoring
- Serial reconnection on disconnect
- Arduino error message handling
- Emergency stop command

Protocol: "J:joint1,joint2,joint3\n" (radians)
Commands: "C:home\n", "C:safe\n", "C:stop\n", "C:start\n", "C:estop\n"
Heartbeat: "H:\n"
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

# ============ MQTT SETTINGS ============
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "robot/position"
MQTT_COMMAND_TOPIC = "robot/command"
MQTT_CLIENT_ID = f"robot_arduino_controller_{int(time.time())}"

# ============ SERIAL SETTINGS ============
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 0.1
SERIAL_RECONNECT_INTERVAL = 2.0  # seconds

# Auto-detect OpenCM9.04 or specify manually
# SERIAL_PORT = "/dev/ttyACM0"  # Uncomment to specify manually
SERIAL_PORT = None  # Auto-detect

# ============ SAFETY SETTINGS ============
HEARTBEAT_INTERVAL = 0.5      # Send heartbeat to Arduino every 500ms
ARDUINO_TIMEOUT = 3.0         # If no response from Arduino for 3s, reconnect
SAFE_SHUTDOWN_TIMEOUT = 2.0   # Max time to wait for safe position on shutdown

# ============ JOINT LIMITS (radians) ============
JOINT1_MIN = -3.14   # -180 deg
JOINT1_MAX = 3.14    # 180 deg
JOINT2_MIN = 0.0     # 0 deg
JOINT2_MAX = 1.57    # 90 deg
JOINT3_MIN = 0.0     # 0 deg
JOINT3_MAX = 1.57    # 90 deg

# ============ REFERENCE BASED CONTROL ============
CENTER_X = 0.0       # cm
CENTER_Y = 0.0       # cm  
CENTER_Z = 15.0      # cm - marker default distance

RANGE_X = 10.0       # ±10cm = full joint1 range
RANGE_Y = 8.0        # ±8cm = full joint3 range
RANGE_Z = 10.0       # ±10cm = full joint2 range

JOINT1_MOVE_LIMIT = 1.5   # rad
JOINT2_MOVE_LIMIT = 0.6   # rad
JOINT3_MOVE_LIMIT = 0.6   # rad

# Direction inversion
INVERT_X = -1
INVERT_Y = -1
INVERT_Z = -1

# Home position
HOME_POSITION = [0.0, 0.5, 0.5]

# Safe position (lower, safer pose)
SAFE_POSITION = [0.0, 0.3, 0.3]

# Smoothing and control
SMOOTHING = 0.15
CONTROL_RATE = 10  # Hz
DEADZONE = 0.3     # cm


class MQTTArduinoController:
    def __init__(self):
        print("=" * 60)
        print("  MQTT to Arduino (OpenCM9.04) Controller")
        print("  WITH SAFETY FEATURES")
        print("=" * 60)
        
        # Serial connection
        self.serial_port = None
        self.serial_connected = False
        self.serial_lock = threading.Lock()
        self.last_arduino_response = time.time()
        
        # Current and target positions
        self.current_joints = list(HOME_POSITION)
        self.target_joints = list(HOME_POSITION)
        
        # Last position for deadzone
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = CENTER_Z
        
        # MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        self.mqtt_connected = False
        
        # State
        self.last_message_time = time.time()
        self.message_count = 0
        self.enabled = True
        self.running = True
        self.emergency_stop = False
        self.shutdown_in_progress = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("[SAFETY] Signal handlers registered (Ctrl+C = graceful shutdown)")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        if self.shutdown_in_progress:
            print("\n[WARN] Force quit requested. Exiting immediately!")
            sys.exit(1)
            
        self.shutdown_in_progress = True
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\n[SAFETY] {sig_name} received. Initiating safe shutdown...")
        
        # Stop control loop
        self.running = False
        self.enabled = False
        
        # Send safe position to Arduino
        self.safe_shutdown()
        
    def safe_shutdown(self):
        """Perform safe shutdown sequence"""
        print("[SAFETY] Sending SAFE position to robot...")
        
        # Try to send safe command multiple times
        for attempt in range(3):
            if self.send_command('safe'):
                print(f"[SAFETY] Safe command sent (attempt {attempt + 1})")
                time.sleep(0.3)  # Give time for servo to move
                break
            time.sleep(0.2)
        
        # Also send safe joint positions directly
        self.send_joints(SAFE_POSITION)
        time.sleep(0.5)
        
        # Disable torque (let servos relax)
        self.send_command('relax')
        
        print("[SAFETY] Robot moved to safe position")
        
    def find_opencm_port(self):
        """Auto-detect OpenCM9.04 serial port"""
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # OpenCM9.04 usually shows up as ttyACM* or with specific VID/PID
            if "ACM" in port.device or "USB" in port.device:
                print(f"  Found potential OpenCM9.04: {port.device} - {port.description}")
                return port.device
            # ROBOTIS OpenCM9.04 VID:PID = 0483:5740
            if port.vid == 0x0483 and port.pid == 0x5740:
                print(f"  Found OpenCM9.04: {port.device}")
                return port.device
                
        return None
        
    def connect_serial(self):
        """Connect to OpenCM9.04 via Serial"""
        port = SERIAL_PORT if SERIAL_PORT else self.find_opencm_port()
        
        if not port:
            print("[ERROR] OpenCM9.04 not found! Available ports:")
            for p in serial.tools.list_ports.comports():
                print(f"  - {p.device}: {p.description}")
            return False
            
        try:
            with self.serial_lock:
                if self.serial_port and self.serial_port.is_open:
                    self.serial_port.close()
                    
                self.serial_port = serial.Serial(
                    port=port,
                    baudrate=SERIAL_BAUDRATE,
                    timeout=SERIAL_TIMEOUT
                )
                time.sleep(2)  # Wait for Arduino reset
                
            self.serial_connected = True
            self.last_arduino_response = time.time()
            print(f"[SERIAL] Connected to {port} @ {SERIAL_BAUDRATE} baud")
            
            # Clear any pending data
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            
            # Send initial home position
            time.sleep(0.5)
            self.send_command('start')
            self.send_joints(HOME_POSITION)
            
            return True
            
        except serial.SerialException as e:
            print(f"[ERROR] Serial connection failed: {e}")
            self.serial_connected = False
            return False
            
    def reconnect_serial(self):
        """Attempt to reconnect to Arduino"""
        print("[SERIAL] Attempting to reconnect...")
        
        with self.serial_lock:
            if self.serial_port:
                try:
                    self.serial_port.close()
                except:
                    pass
                self.serial_port = None
                
        self.serial_connected = False
        time.sleep(SERIAL_RECONNECT_INTERVAL)
        return self.connect_serial()
            
    def send_joints(self, joints):
        """Send joint positions to OpenCM9.04"""
        if not self.serial_connected or not self.serial_port:
            return False
            
        if self.emergency_stop:
            return False
            
        try:
            with self.serial_lock:
                # Format: "J:j1,j2,j3\n" (radians with 4 decimal places)
                cmd = f"J:{joints[0]:.4f},{joints[1]:.4f},{joints[2]:.4f}\n"
                self.serial_port.write(cmd.encode())
            return True
        except Exception as e:
            print(f"[ERROR] Serial write failed: {e}")
            self.serial_connected = False
            return False
            
    def send_command(self, command):
        """Send command to OpenCM9.04"""
        if not self.serial_port:
            return False
            
        try:
            with self.serial_lock:
                cmd = f"C:{command}\n"
                self.serial_port.write(cmd.encode())
            print(f"[SERIAL] Sent command: {command}")
            return True
        except Exception as e:
            print(f"[ERROR] Serial write failed: {e}")
            return False
            
    def send_heartbeat(self):
        """Send heartbeat to keep Arduino watchdog happy"""
        if not self.serial_connected or not self.serial_port:
            return False
            
        try:
            with self.serial_lock:
                self.serial_port.write(b"H:\n")  # Heartbeat command
            return True
        except:
            return False
            
    def read_arduino_responses(self):
        """Read and process responses from Arduino"""
        if not self.serial_connected or not self.serial_port:
            return
            
        try:
            with self.serial_lock:
                while self.serial_port.in_waiting > 0:
                    response = self.serial_port.readline().decode().strip()
                    if response:
                        self.last_arduino_response = time.time()
                        self.process_arduino_response(response)
        except Exception as e:
            print(f"[ERROR] Serial read failed: {e}")
            
    def process_arduino_response(self, response):
        """Process response messages from Arduino"""
        if response.startswith("[ERR]"):
            print(f"[ARDUINO ERROR] {response}")
            # On critical error, trigger emergency stop
            if "VOLTAGE" in response or "OVERHEAT" in response:
                self.trigger_emergency_stop("Arduino reported critical error")
                
        elif response.startswith("[WARN]"):
            print(f"[ARDUINO WARN] {response}")
            
        elif response.startswith("[ESTOP]"):
            print(f"[ARDUINO] {response}")
            self.trigger_emergency_stop("Arduino triggered E-STOP")
            
        elif response.startswith("[OK]") or response.startswith("[CMD]"):
            # Normal status messages
            print(f"[ARDUINO] {response}")
            
        else:
            # Debug messages
            print(f"[ARDUINO] {response}")
            
    def trigger_emergency_stop(self, reason):
        """Trigger emergency stop"""
        if self.emergency_stop:
            return
            
        self.emergency_stop = True
        self.enabled = False
        
        print("=" * 60)
        print(f"  !!! EMERGENCY STOP TRIGGERED !!!")
        print(f"  Reason: {reason}")
        print("=" * 60)
        
        # Send E-STOP command to Arduino
        self.send_command('estop')
        
        print("[E-STOP] Robot stopped. Manual reset required.")
        print("[E-STOP] To reset: restart this controller")
        
    def reset_emergency_stop(self):
        """Reset emergency stop (manual action)"""
        self.emergency_stop = False
        self.enabled = True
        self.send_command('reset')
        print("[SAFETY] Emergency stop reset. Control resumed.")
            
    def on_mqtt_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
            self.mqtt_connected = True
            client.subscribe(MQTT_TOPIC)
            client.subscribe(MQTT_COMMAND_TOPIC)
            print(f"[MQTT] Subscribed to: {MQTT_TOPIC}, {MQTT_COMMAND_TOPIC}")
        else:
            print(f"[ERROR] MQTT connection failed! Code: {rc}")
            
    def on_mqtt_disconnect(self, client, userdata, flags, rc, properties=None):
        print("[MQTT] Disconnected from broker!")
        self.mqtt_connected = False
        
        # On MQTT disconnect, go to safe position
        if not self.shutdown_in_progress:
            print("[SAFETY] MQTT lost - sending safe command")
            self.send_command('safe')
        
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
                
        except json.JSONDecodeError as e:
            print(f"[WARN] Invalid JSON: {e}")
        except Exception as e:
            print(f"[ERROR] Message handling error: {e}")
            
    def handle_position(self, data):
        """Process position data and update target joints"""
        if not self.enabled or self.emergency_stop:
            return
            
        x = data.get('x', 0)
        y = data.get('y', 0)
        z = data.get('z', CENTER_Z)
        
        # Apply deadzone
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
        
        # Normalize to -1...+1
        norm_x = self.clamp(offset_x / RANGE_X, -1.0, 1.0)
        norm_y = self.clamp(offset_y / RANGE_Y, -1.0, 1.0)
        norm_z = self.clamp(offset_z / RANGE_Z, -1.0, 1.0)
        
        # Apply inversion
        norm_x *= INVERT_X
        norm_y *= INVERT_Y
        norm_z *= INVERT_Z
        
        # Calculate joint positions
        joint1 = HOME_POSITION[0] + norm_x * JOINT1_MOVE_LIMIT
        joint2 = HOME_POSITION[1] + norm_z * JOINT2_MOVE_LIMIT
        joint3 = HOME_POSITION[2] + norm_y * JOINT3_MOVE_LIMIT
        
        # Clamp to limits
        joint1 = self.clamp(joint1, JOINT1_MIN, JOINT1_MAX)
        joint2 = self.clamp(joint2, JOINT2_MIN, JOINT2_MAX)
        joint3 = self.clamp(joint3, JOINT3_MIN, JOINT3_MAX)
        
        self.target_joints = [joint1, joint2, joint3]
        
        if self.message_count % 20 == 0:
            print(f"[RX #{self.message_count}] X:{x:.1f} Y:{y:.1f} Z:{z:.1f} -> "
                  f"J1:{math.degrees(joint1):.1f}° J2:{math.degrees(joint2):.1f}° J3:{math.degrees(joint3):.1f}°")
            
    def handle_command(self, data):
        """Process command messages"""
        command = data.get('command', '')
        
        if command == 'home':
            print("[CMD] HOME - Moving to home position")
            self.target_joints = list(HOME_POSITION)
            self.send_command('home')
            
        elif command == 'safe':
            print("[CMD] SAFE - Moving to safe position")
            self.target_joints = list(SAFE_POSITION)
            self.send_command('safe')
            
        elif command == 'stop':
            print("[CMD] STOP - Disabling control")
            self.enabled = False
            self.send_command('stop')
            
        elif command == 'start':
            print("[CMD] START - Enabling control")
            if not self.emergency_stop:
                self.enabled = True
                self.send_command('start')
            else:
                print("[WARN] Cannot start - emergency stop active")
                
        elif command == 'estop':
            self.trigger_emergency_stop("Remote E-STOP command received")
            
        elif command == 'reset':
            self.reset_emergency_stop()
            
    def clamp(self, value, min_val, max_val):
        return max(min_val, min(max_val, value))
        
    def control_loop(self):
        """Main control loop - applies smoothing and sends to Arduino"""
        rate = 1.0 / CONTROL_RATE
        last_heartbeat = time.time()
        
        while self.running:
            try:
                # Check Arduino connection health
                if self.serial_connected:
                    if time.time() - self.last_arduino_response > ARDUINO_TIMEOUT:
                        print("[WARN] Arduino not responding, attempting reconnect...")
                        self.reconnect_serial()
                        
                # Send heartbeat periodically
                if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                    self.send_heartbeat()
                    last_heartbeat = time.time()
                    
                # Read Arduino responses
                self.read_arduino_responses()
                
                # Apply smoothing (only if not emergency stopped)
                if not self.emergency_stop and self.enabled:
                    for i in range(3):
                        self.current_joints[i] += (self.target_joints[i] - self.current_joints[i]) * SMOOTHING
                        
                    # Send to Arduino
                    if self.serial_connected:
                        self.send_joints(self.current_joints)
                            
                time.sleep(rate)
                
            except Exception as e:
                print(f"[ERROR] Control loop: {e}")
                time.sleep(0.5)
                
    def run(self):
        """Main entry point"""
        # Connect to Serial
        if not self.connect_serial():
            print("\n[WARN] Running without Serial connection (dry-run mode)")
            print("       Connect OpenCM9.04 and restart to enable hardware control\n")
            
        # Connect to MQTT
        try:
            print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"[ERROR] MQTT connection failed: {e}")
            return
            
        print("\n" + "=" * 60)
        print("  Controller running with SAFETY FEATURES enabled")
        print("  - Ctrl+C       : Graceful shutdown (safe position)")
        print("  - MQTT 'estop' : Emergency stop")
        print("  - MQTT 'reset' : Reset emergency stop")
        print("  - Watchdog     : Auto-safe on connection loss")
        print("=" * 60 + "\n")
        
        try:
            self.control_loop()
        finally:
            # Cleanup (signal handler may have already done this)
            if not self.shutdown_in_progress:
                self.safe_shutdown()
                
            self.running = False
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
            if self.serial_port:
                try:
                    self.serial_port.close()
                except:
                    pass
                    
            print("[INFO] Controller stopped safely.")


if __name__ == "__main__":
    controller = MQTTArduinoController()
    controller.run()
