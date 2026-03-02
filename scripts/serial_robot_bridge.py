#!/usr/bin/env python3
"""
Serial Bridge for OpenCM9.04 + AX-12A Robot Arm

Handles USB serial communication between the Python controller
and the OpenCM9.04 board controlling Dynamixel AX-12A servos.

Protocol (matching OpenCM9.04 firmware):
  Joints:    "J:j1,j2,j3\n"  (radians)
  Commands:  "C:home\n", "C:safe\n", "C:stop\n", "C:start\n"
             "C:estop\n", "C:reset\n", "C:relax\n"
  Heartbeat: "H:\n"
  Status:    "?\n"

Usage:
    from serial_robot_bridge import SerialRobotBridge

    bridge = SerialRobotBridge(port="/dev/ttyACM0")
    bridge.connect()
    bridge.send_joints([0.0, 1.0, 1.0])
    bridge.disconnect()
"""

import time
import threading

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class SerialRobotBridge:
    DEFAULT_PORT = "/dev/ttyACM0"
    DEFAULT_BAUD = 115200
    HEARTBEAT_INTERVAL = 1.0  # seconds

    def __init__(self, port=None, baud=None):
        self.port = port or self.DEFAULT_PORT
        self.baud = baud or self.DEFAULT_BAUD
        self.ser = None
        self.connected = False

        self._heartbeat_thread = None
        self._heartbeat_running = False
        self._read_thread = None
        self._read_running = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self):
        if not SERIAL_AVAILABLE:
            print("[SERIAL] pyserial not installed. Run: pip install pyserial")
            return False

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # OpenCM9.04 resets on serial connect
            self.ser.reset_input_buffer()
            self.connected = True

            self._start_heartbeat()
            self._start_reader()

            self.send_command("start")
            print(f"[SERIAL] Connected to {self.port} @ {self.baud}")
            return True

        except serial.SerialException as e:
            print(f"[SERIAL] Connection failed: {e}")
            self.ser = None
            self.connected = False
            return False

    def disconnect(self):
        self._stop_heartbeat()
        self._stop_reader()

        if self.ser and self.ser.is_open:
            try:
                self.send_command("safe")
                time.sleep(0.3)
                self.ser.close()
            except Exception:
                pass

        self.ser = None
        self.connected = False
        print("[SERIAL] Disconnected")

    def is_connected(self):
        return self.connected and self.ser is not None and self.ser.is_open

    # ------------------------------------------------------------------
    # Send
    # ------------------------------------------------------------------
    def send_joints(self, joint_angles):
        """Send joint angles (radians) to OpenCM9.04: J:j1,j2,j3"""
        if not self.is_connected():
            return False
        try:
            cmd = f"J:{joint_angles[0]:.4f},{joint_angles[1]:.4f},{joint_angles[2]:.4f}\n"
            self.ser.write(cmd.encode())
            return True
        except Exception as e:
            print(f"[SERIAL] Send failed: {e}")
            self.connected = False
            return False

    def send_command(self, command):
        """Send control command: C:home, C:safe, C:stop, C:start, etc."""
        if not self.is_connected():
            return False
        try:
            self.ser.write(f"C:{command}\n".encode())
            return True
        except Exception as e:
            print(f"[SERIAL] Command failed: {e}")
            self.connected = False
            return False

    def send_heartbeat(self):
        if not self.is_connected():
            return
        try:
            self.ser.write(b"H:\n")
        except Exception:
            self.connected = False

    # ------------------------------------------------------------------
    # Background heartbeat
    # ------------------------------------------------------------------
    def _start_heartbeat(self):
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._heartbeat_running = False
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2)

    def _heartbeat_loop(self):
        while self._heartbeat_running and self.is_connected():
            self.send_heartbeat()
            time.sleep(self.HEARTBEAT_INTERVAL)

    # ------------------------------------------------------------------
    # Background reader (prints OpenCM9.04 debug output)
    # ------------------------------------------------------------------
    def _start_reader(self):
        self._read_running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    def _stop_reader(self):
        self._read_running = False
        if self._read_thread:
            self._read_thread.join(timeout=2)

    def _read_loop(self):
        while self._read_running and self.is_connected():
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"[OpenCM] {line}")
            except Exception:
                break
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def list_ports():
        """List available serial ports"""
        if not SERIAL_AVAILABLE:
            print("[SERIAL] pyserial not installed")
            return []
        ports = serial.tools.list_ports.comports()
        for p in ports:
            print(f"  {p.device} - {p.description}")
        return [p.device for p in ports]

    def request_status(self):
        """Request status report from OpenCM9.04"""
        if not self.is_connected():
            return
        try:
            self.ser.write(b"?\n")
        except Exception:
            pass
