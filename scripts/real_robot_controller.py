#!/usr/bin/env python3
"""
Real Robot Controller - OpenCM9.04 + AX-12A

Listens to ArUco position data via MQTT and sends joint commands
to the real robot via serial (OpenCM9.04).

Runs PARALLEL with mqtt_robot_ann_controller.py (Gazebo).
Both subscribe to the same MQTT topic, so ArUco controls
both Gazebo and the real robot simultaneously.

Usage:
    python3 real_robot_controller.py
    python3 real_robot_controller.py --port /dev/ttyACM1

Pipeline:
    ArUco Camera → aruco_mqtt_sender.py → MQTT
                                            ├── mqtt_robot_ann_controller.py → Gazebo
                                            └── real_robot_controller.py → OpenCM9.04 → Servos
"""

import time
import json
import argparse
import signal
import sys
import threading

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[ERR] paho-mqtt required: pip install paho-mqtt")
    sys.exit(1)

from serial_robot_bridge import SerialRobotBridge, SERIAL_AVAILABLE

if not SERIAL_AVAILABLE:
    print("[ERR] pyserial required: pip install pyserial")
    sys.exit(1)

# =============================================================================
# Configuration (same values as Gazebo controller)
# =============================================================================
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC_POSITION = "robot/position"
MQTT_TOPIC_COMMAND = "robot/command"
MESSAGE_TIMEOUT = 8.0

# Joint limits (matching URDF + OpenCM9.04)
JOINT1_MIN = -2.268
JOINT1_MAX = 2.268
JOINT2_MIN = 0.0
JOINT2_MAX = 2.0
JOINT3_MIN = 0.0
JOINT3_MAX = 2.0

# Reference-based mapping (same as Gazebo controller)
CENTER_X = 0.0
CENTER_Y = 0.0
CENTER_Z = 15.0

RANGE_X = 10.0
RANGE_Y = 8.0
RANGE_Z = 8.0

JOINT1_MOVE_LIMIT = 1.2
JOINT2_MOVE_LIMIT = 1.0
JOINT3_MOVE_LIMIT = 1.0

INVERT_X = -1
INVERT_Y = -1
INVERT_Z = -1

HOME_POSITION = [0.0, 1.0, 1.0]
SAFE_POSITION = [0.0, 0.5, 0.5]

SMOOTHING = 0.15
DEADZONE = 0.2
CONTROL_RATE = 30


class RealRobotController:
    def __init__(self, serial_port):
        self.bridge = SerialRobotBridge(port=serial_port)
        self.joint_angles = [0.0, 0.0, 0.0]
        self.running = True

        self.last_x = CENTER_X
        self.last_y = CENTER_Y
        self.last_z = CENTER_Z

        self.last_message_time = time.time()
        self.timeout_triggered = False
        self.message_count = 0

        self.mqtt_client = None
        self.mqtt_connected = False
        self.control_active = False

    def start(self):
        print("\n" + "=" * 60)
        print("  Real Robot Controller (OpenCM9.04 + AX-12A)")
        print("=" * 60)

        if not self.bridge.connect():
            print("[ERR] Cannot connect to OpenCM9.04. Exiting.")
            SerialRobotBridge.list_ports()
            return False

        self._setup_mqtt()

        print("-" * 60)
        print(f"  MQTT:   {MQTT_BROKER}:{MQTT_PORT}")
        print(f"  Serial: {self.bridge.port}")
        print(f"  Home:   {HOME_POSITION}")
        print("-" * 60)
        print("Controls:")
        print("  [S] Start/Stop    [H] Home       [F] Safe")
        print("  [E] E-STOP        [R] Relax      [I] Status")
        print("  [Q] Quit")
        print("=" * 60 + "\n")

        self._setup_keyboard()
        print("[INFO] Press [S] to start control")
        self._run_loop()
        return True

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def _setup_keyboard(self):
        def key_listener():
            try:
                from pynput import keyboard
                def on_press(key):
                    try:
                        c = key.char.lower() if hasattr(key, 'char') else None
                        if c == 's':
                            self._toggle_control()
                        elif c == 'h':
                            print("[KEY] Home")
                            self.bridge.send_command("home")
                            self._go_to_home()
                        elif c == 'f':
                            print("[KEY] Safe")
                            self._go_to_safe()
                        elif c == 'e':
                            print("[KEY] E-STOP!")
                            self.control_active = False
                            self.bridge.send_command("estop")
                        elif c == 'r':
                            print("[KEY] Relax (torque off)")
                            self.control_active = False
                            self.bridge.send_command("relax")
                        elif c == 'i':
                            self.bridge.request_status()
                        elif c == 'q':
                            self.shutdown()
                            sys.exit(0)
                    except AttributeError:
                        pass
                listener = keyboard.Listener(on_press=on_press)
                listener.start()
            except ImportError:
                print("[WARN] pynput not installed - keyboard disabled")

        threading.Thread(target=key_listener, daemon=True).start()

    def _toggle_control(self):
        self.control_active = not self.control_active
        if self.control_active:
            self.bridge.send_command("start")
            print("[CONTROL] STARTED - robot follows ArUco")
        else:
            self.bridge.send_command("stop")
            print("[CONTROL] STOPPED - robot frozen")

    # ------------------------------------------------------------------
    # MQTT
    # ------------------------------------------------------------------
    def _setup_mqtt(self):
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        self.mqtt_client.on_disconnect = self._on_disconnect

        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"[ERR] MQTT connection failed: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.mqtt_connected = True
            client.subscribe(MQTT_TOPIC_POSITION)
            client.subscribe(MQTT_TOPIC_COMMAND)
            print("[MQTT] Connected")
        else:
            print(f"[MQTT] Connection failed: rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.mqtt_connected = False
        print("[MQTT] Disconnected")

    def _on_message(self, client, userdata, msg):
        if msg.topic == MQTT_TOPIC_COMMAND:
            command = msg.payload.decode().strip().lower()
            if command == "safe":
                self._go_to_safe()
            elif command == "home":
                self._go_to_home()
            return

        if msg.topic == MQTT_TOPIC_POSITION:
            self.last_message_time = time.time()
            self.timeout_triggered = False

            try:
                data = json.loads(msg.payload.decode())
                x = data.get('x', 0)
                y = data.get('y', 0)
                z = data.get('z', 0)
                self._move_to_position((x, y, z))
                self.message_count += 1
            except (json.JSONDecodeError, KeyError):
                pass

    # ------------------------------------------------------------------
    # Reference mapping (same as Gazebo controller)
    # ------------------------------------------------------------------
    def _reference_mapping(self, position):
        x, y, z = position

        offset_x = x - CENTER_X
        offset_y = y - CENTER_Y
        offset_z = z - CENTER_Z

        norm_x = max(-1.0, min(1.0, offset_x / RANGE_X)) * INVERT_X
        norm_y = max(-1.0, min(1.0, offset_y / RANGE_Y)) * INVERT_Y
        norm_z = max(-1.0, min(1.0, offset_z / RANGE_Z)) * INVERT_Z

        joint1 = max(JOINT1_MIN, min(JOINT1_MAX, HOME_POSITION[0] + norm_x * JOINT1_MOVE_LIMIT))
        joint2 = max(JOINT2_MIN, min(JOINT2_MAX, HOME_POSITION[1] + norm_z * JOINT2_MOVE_LIMIT))
        joint3 = max(JOINT3_MIN, min(JOINT3_MAX, HOME_POSITION[2] + norm_z * JOINT3_MOVE_LIMIT))

        return [joint1, joint2, joint3]

    def _move_to_position(self, position):
        if not self.control_active:
            return

        x, y, z = position

        dx = abs(x - self.last_x)
        dy = abs(y - self.last_y)
        dz = abs(z - self.last_z)

        if dx < DEADZONE and dy < DEADZONE and dz < DEADZONE:
            return

        self.last_x, self.last_y, self.last_z = x, y, z

        target_joints = self._reference_mapping(position)

        for i in range(3):
            self.joint_angles[i] += (target_joints[i] - self.joint_angles[i]) * SMOOTHING

        self.bridge.send_joints(self.joint_angles)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def _go_to_home(self):
        print("[HOME] Moving to home position...")
        target = HOME_POSITION
        for _ in range(20):
            for i in range(3):
                self.joint_angles[i] += (target[i] - self.joint_angles[i]) * 0.2
            self.bridge.send_joints(self.joint_angles)
            time.sleep(0.05)
        self.bridge.send_command("home")
        print("[HOME] Done")

    def _go_to_safe(self):
        print("[SAFE] Moving to safe position...")
        for _ in range(20):
            for i in range(3):
                self.joint_angles[i] += (SAFE_POSITION[i] - self.joint_angles[i]) * 0.2
            self.bridge.send_joints(self.joint_angles)
            time.sleep(0.05)
        self.bridge.send_command("safe")
        print("[SAFE] Done")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _run_loop(self):
        rate_sleep = 1.0 / CONTROL_RATE

        while self.running:
            if time.time() - self.last_message_time > MESSAGE_TIMEOUT:
                if not self.timeout_triggered:
                    print("[TIMEOUT] No ArUco data - going to safe")
                    self._go_to_safe()
                    self.timeout_triggered = True

            if self.message_count > 0 and self.message_count % 300 == 0:
                print(f"[STATUS] msgs={self.message_count} joints=[{self.joint_angles[0]:.2f}, {self.joint_angles[1]:.2f}, {self.joint_angles[2]:.2f}]")

            time.sleep(rate_sleep)

    def shutdown(self):
        print("\n[SHUTDOWN] Stopping...")
        self.running = False
        self._go_to_safe()

        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

        self.bridge.disconnect()
        print("[SHUTDOWN] Done")


def main():
    parser = argparse.ArgumentParser(description="Real Robot Controller")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port (default: /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports")
    args = parser.parse_args()

    if args.list_ports:
        SerialRobotBridge.list_ports()
        return

    controller = RealRobotController(serial_port=args.port)

    def signal_handler(sig, frame):
        controller.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    controller.start()


if __name__ == "__main__":
    main()
