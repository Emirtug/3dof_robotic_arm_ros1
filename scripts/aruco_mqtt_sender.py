#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
ArUco MQTT Sender - Operator Side
Reads ArUco marker position from camera and publishes via MQTT
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import json
import time
from datetime import datetime
import paho.mqtt.client as mqtt

# ============== CONFIGURATION ==============
# Camera settings
CAMERA_ID = 2  # USB 2.0 Camera (external), HP built-in = 2
MARKER_SIZE = 0.015              # 15mm marker size in meters
TARGET_ID = 102
FRAME_WIDTH = 640                # Camera resolution in 4:3 ratio
FRAME_HEIGHT = 480               # Reduced - for performance
FOCAL_LENGTH = 640               # Aspect ratio preserved

# MQTT settings
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "robot/position"
MQTT_COMMAND_TOPIC = "robot/command"
MQTT_CLIENT_ID = f"aruco_sender_{int(time.time())}"

# Transmission settings
SEND_RATE = 30                   # Messages per second (Hz) - Increased
SEND_INTERVAL = 1.0 / SEND_RATE
CONNECTION_TIMEOUT = 8.0         # Auto home after 8 sec disconnect

# Verification settings
VERIFICATION_TIME = 2.0          # Shortened - faster start
VERIFICATION_ZONE_SIZE = 50      # Lowered - for small resolution

# Marker lost safety settings
MARKER_LOST_TIMEOUT = 10.0       # Seconds before safety action when marker lost

# Image preprocessing filters (for better ArUco detection)
FILTER_ENABLED = True            # Master switch - ON (for detection)
FILTER_CLAHE = True              # Contrast Limited Adaptive Histogram Equalization - ON
FILTER_CLAHE_CLIP = 3.0          # CLAHE clip limit (higher = more contrast)
FILTER_CLAHE_GRID = 8            # CLAHE grid size
FILTER_DENOISE = True            # Gaussian blur for noise reduction - ON (reduces jitter)
FILTER_DENOISE_KERNEL = 3        # Blur kernel size (odd number: 3, 5, 7)
FILTER_SHARPEN = False           # Sharpening filter
FILTER_SHARPEN_AMOUNT = 1.0      # Sharpening strength (0.5-2.0)
FILTER_ADAPTIVE_THRESH = False   # Adaptive thresholding (experimental)
FILTER_SHOW_DEBUG = False        # Show filtered image in separate window
# ===========================================


class ArucoMQTTSender:
    def __init__(self):
        cx, cy = FRAME_WIDTH / 2, FRAME_HEIGHT / 2
        self.camera_matrix = np.array([
            [FOCAL_LENGTH, 0, cx],
            [0, FOCAL_LENGTH, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = aruco.DetectorParameters()
        
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 53
        self.aruco_params.adaptiveThreshWinSizeStep = 4
        self.aruco_params.minMarkerPerimeterRate = 0.02
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.08
        self.aruco_params.minCornerDistanceRate = 0.02
        self.aruco_params.minOtsuStdDev = 5.0
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        self.aruco_params.errorCorrectionRate = 0.6
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.1
        
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        self.clahe = cv2.createCLAHE(
            clipLimit=FILTER_CLAHE_CLIP, 
            tileGridSize=(FILTER_CLAHE_GRID, FILTER_CLAHE_GRID)
        )
        
        self.sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        self.filter_fps = 0
        self.last_filter_time = time.time()
        
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, MQTT_CLIENT_ID)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_connected = False
        
        self.last_send_time = 0
        self.last_successful_send = time.time()
        self.sending_enabled = True
        self.last_position = None
        self.message_count = 0
        self.home_sent_for_timeout = False
        
        self.verified = False
        self.verification_start_time = None
        self.verification_progress = 0.0
        self.last_marker_center = None
        
        self.marker_last_seen_time = time.time()
        self.marker_lost_warning_sent = False
        
        # Temporal stabilization - keep last position for a few frames when marker is lost
        self.stable_corners = None
        self.stable_id = None
        self.frames_since_seen = 0
        self.max_holdover_frames = 5  # Hold last position for 5 frames when marker is lost
        
        
    def preprocess_frame(self, gray):
        """
        Apply image preprocessing filters for better ArUco detection.
        
        Filters applied (when enabled):
        1. CLAHE - Improves contrast in different lighting conditions
        2. Gaussian Blur - Reduces noise
        3. Sharpening - Enhances edges
        4. Adaptive Threshold - Binarizes image (experimental)
        
        Returns:
            Preprocessed grayscale image
        """
        if not FILTER_ENABLED:
            return gray
        
        processed = gray.copy()
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if FILTER_CLAHE:
            processed = self.clahe.apply(processed)
        
        # 2. Denoise with Gaussian Blur
        if FILTER_DENOISE:
            kernel_size = FILTER_DENOISE_KERNEL
            if kernel_size % 2 == 0:
                kernel_size += 1  # Must be odd
            processed = cv2.GaussianBlur(processed, (kernel_size, kernel_size), 0)
        
        # 3. Sharpening
        if FILTER_SHARPEN:
            blurred = cv2.GaussianBlur(processed, (0, 0), 3)
            processed = cv2.addWeighted(
                processed, 1 + FILTER_SHARPEN_AMOUNT,
                blurred, -FILTER_SHARPEN_AMOUNT,
                0
            )
        
        # 4. Adaptive Thresholding (experimental - can help in some lighting)
        if FILTER_ADAPTIVE_THRESH:
            processed = cv2.adaptiveThreshold(
                processed, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )
        
        if FILTER_SHOW_DEBUG:
            comparison = np.hstack([gray, processed])
            cv2.putText(comparison, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, "Filtered", (FRAME_WIDTH + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Filter Debug', comparison)
        
        return processed
    
    def toggle_filter(self, filter_name):
        """Toggle a specific filter on/off"""
        global FILTER_CLAHE, FILTER_DENOISE, FILTER_SHARPEN, FILTER_ADAPTIVE_THRESH, FILTER_ENABLED
        
        if filter_name == 'clahe':
            FILTER_CLAHE = not FILTER_CLAHE
            print(f"[FILTER] CLAHE: {'ON' if FILTER_CLAHE else 'OFF'}")
        elif filter_name == 'denoise':
            FILTER_DENOISE = not FILTER_DENOISE
            print(f"[FILTER] Denoise: {'ON' if FILTER_DENOISE else 'OFF'}")
        elif filter_name == 'sharpen':
            FILTER_SHARPEN = not FILTER_SHARPEN
            print(f"[FILTER] Sharpen: {'ON' if FILTER_SHARPEN else 'OFF'}")
        elif filter_name == 'adaptive':
            FILTER_ADAPTIVE_THRESH = not FILTER_ADAPTIVE_THRESH
            print(f"[FILTER] Adaptive Thresh: {'ON' if FILTER_ADAPTIVE_THRESH else 'OFF'}")
        elif filter_name == 'all':
            FILTER_ENABLED = not FILTER_ENABLED
            print(f"[FILTER] All Filters: {'ON' if FILTER_ENABLED else 'OFF'}")
        elif filter_name == 'debug':
            global FILTER_SHOW_DEBUG
            FILTER_SHOW_DEBUG = not FILTER_SHOW_DEBUG
            print(f"[FILTER] Debug View: {'ON' if FILTER_SHOW_DEBUG else 'OFF'}")
            if not FILTER_SHOW_DEBUG:
                cv2.destroyWindow('Filter Debug')
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[MQTT] Connected to broker: {MQTT_BROKER}:{MQTT_PORT}")
            self.mqtt_connected = True
            self.last_successful_send = time.time()
            self.home_sent_for_timeout = False
        else:
            print(f"[MQTT] Connection failed! Code: {rc}")
            
    def on_disconnect(self, client, userdata, flags, rc, properties=None):
        print(f"[MQTT] Disconnected from broker!")
        self.mqtt_connected = False
        
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            time.sleep(1)
            return self.mqtt_connected
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
            return False
            
    def check_connection_timeout(self):
        """Check if connection has been lost for too long and send home command"""
        if not self.mqtt_connected:
            elapsed = time.time() - self.last_successful_send
            if elapsed >= CONNECTION_TIMEOUT and not self.home_sent_for_timeout:
                print(f"\n[SAFETY] Connection lost for {CONNECTION_TIMEOUT}s - Sending HOME command!")
                self.home_sent_for_timeout = True
                return True
        return False
        
    def check_marker_lost_timeout(self, marker_found):
        """Check if marker has been lost for too long"""
        if marker_found:
            self.marker_last_seen_time = time.time()
            self.marker_lost_warning_sent = False
            return False
        else:
            elapsed = time.time() - self.marker_last_seen_time
            if elapsed >= MARKER_LOST_TIMEOUT and not self.marker_lost_warning_sent:
                print(f"\n[SAFETY] Marker lost for {MARKER_LOST_TIMEOUT}s - Moving to SAFE position!")
                self.marker_lost_warning_sent = True
                self.send_command("safe")
                return True
            return False
            
    def get_marker_lost_elapsed(self):
        """Get time since marker was last seen"""
        return time.time() - self.marker_last_seen_time
            
    def send_position(self, x, y, z):
        """Publish position data via MQTT and print joint angles"""
        if not self.mqtt_connected:
            return False
        current_time = time.time()
        if current_time - self.last_send_time < SEND_INTERVAL:
            return False
        data = {
            "x": round(x, 2),
            "y": round(y, 2),
            "z": round(z, 2),
            "timestamp": current_time,
            "marker_id": TARGET_ID
        }
        try:
            self.mqtt_client.publish(MQTT_TOPIC, json.dumps(data))
            self.last_send_time = current_time
            self.last_successful_send = current_time
            self.message_count += 1
            self.home_sent_for_timeout = False
            return True
        except Exception as e:
            print(f"[MQTT] Send error: {e}")
            return False
            
    def send_command(self, command):
        """Send control command (start, stop, home)"""
        if not self.mqtt_connected:
            return False
            
        data = {
            "command": command,
            "timestamp": time.time()
        }
        
        try:
            self.mqtt_client.publish(MQTT_COMMAND_TOPIC, json.dumps(data))
            print(f"[MQTT] Command sent: {command}")
            return True
        except Exception as e:
            print(f"[MQTT] Command error: {e}")
            return False
            
    def is_in_verification_zone(self, marker_center):
        """Check if marker is within the center verification zone"""
        screen_center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
        distance = np.sqrt(
            (marker_center[0] - screen_center[0])**2 + 
            (marker_center[1] - screen_center[1])**2
        )
        return distance < VERIFICATION_ZONE_SIZE
        
    def update_verification(self, marker_center):
        """Update verification progress"""
        if marker_center is None:
            self.verification_start_time = None
            self.verification_progress = max(0, self.verification_progress - 0.05)
            return False
            
        if not self.is_in_verification_zone(marker_center):
            self.verification_start_time = None
            self.verification_progress = max(0, self.verification_progress - 0.02)
            return False
            
        if self.verification_start_time is None:
            self.verification_start_time = time.time()
            
        elapsed = time.time() - self.verification_start_time
        self.verification_progress = min(1.0, elapsed / VERIFICATION_TIME)
        
        if self.verification_progress >= 1.0:
            self.verified = True
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(0.1)
            print("\n" + "=" * 40)
            print("   VERIFICATION COMPLETE!")
            print("   System is now active.")
            print("=" * 40 + "\n")
            return True
            
        return False
        
    def draw_verification_screen(self, frame, marker_center=None):
        """Draw the Face ID style verification interface"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        center_x, center_y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        
        cv2.putText(frame, "MARKER VERIFICATION", (center_x - 140, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Hold marker in the target zone", (center_x - 160, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        zone_color = (0, 255, 0) if self.verification_progress > 0.5 else (0, 165, 255)
        cv2.circle(frame, (center_x, center_y), VERIFICATION_ZONE_SIZE, zone_color, 2)
        
        if self.verification_progress > 0:
            angle = int(360 * self.verification_progress)
            axes = (VERIFICATION_ZONE_SIZE + 15, VERIFICATION_ZONE_SIZE + 15)
            cv2.ellipse(frame, (center_x, center_y), axes, -90, 0, angle, (0, 255, 0), 8)
        
        cross_size = 20
        cv2.line(frame, (center_x - cross_size, center_y), 
                 (center_x + cross_size, center_y), (100, 100, 100), 1)
        cv2.line(frame, (center_x, center_y - cross_size), 
                 (center_x, center_y + cross_size), (100, 100, 100), 1)
        
        bracket_size = 30
        bracket_offset = VERIFICATION_ZONE_SIZE - 10
        corners = [
            (center_x - bracket_offset, center_y - bracket_offset),
            (center_x + bracket_offset, center_y - bracket_offset),
            (center_x - bracket_offset, center_y + bracket_offset),
            (center_x + bracket_offset, center_y + bracket_offset),
        ]
        
        for i, (cx, cy) in enumerate(corners):
            dx = bracket_size if i % 2 == 0 else -bracket_size
            dy = bracket_size if i < 2 else -bracket_size
            cv2.line(frame, (cx, cy), (cx + dx, cy), zone_color, 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy), zone_color, 2)
        
        percent = int(self.verification_progress * 100)
        percent_text = f"{percent}%"
        text_size = cv2.getTextSize(percent_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        cv2.putText(frame, percent_text, 
                    (center_x - text_size[0]//2, center_y + VERIFICATION_ZONE_SIZE + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        bar_width = 300
        bar_height = 20
        bar_x = center_x - bar_width // 2
        bar_y = FRAME_HEIGHT - 60
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (50, 50, 50), -1)
        fill_width = int(bar_width * self.verification_progress)
        if fill_width > 0:
            color = (0, int(255 * self.verification_progress), int(255 * (1 - self.verification_progress)))
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                          color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (200, 200, 200), 2)
        
        if marker_center is None:
            status = "Searching for marker..."
            status_color = (0, 0, 255)
        elif not self.is_in_verification_zone(marker_center):
            status = "Move marker to center"
            status_color = (0, 165, 255)
        else:
            status = "Hold steady..."
            status_color = (0, 255, 0)
            
        text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(frame, status, (center_x - text_size[0]//2, bar_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(frame, f"Target: Marker ID {TARGET_ID}", (10, FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
            
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer - prevent lag
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera /dev/video{CAMERA_ID}")
            return
            
        if not self.connect_mqtt():
            print("WARNING: No MQTT connection, running in view-only mode")
            
        print("=" * 60)
        print("       ArUco MQTT Sender - Operator Side")
        print("=" * 60)
        print(f"  Camera: /dev/video{CAMERA_ID}")
        print(f"  Target Marker: {TARGET_ID}")
        print(f"  MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
        print(f"  Topic: {MQTT_TOPIC}")
        print(f"  Send Rate: {SEND_RATE} Hz")
        print(f"  Safety Timeout: {CONNECTION_TIMEOUT}s")
        print("=" * 60)
        print("  Verification required before operation!")
        print("  Hold marker in center zone for 3 seconds.")
        print("-" * 60)
        print("  Filter Controls (after verification):")
        print("    [F] Toggle all filters")
        print("    [1] Toggle CLAHE")
        print("    [2] Toggle Denoise")
        print("    [3] Toggle Sharpen")
        print("    [4] Toggle Adaptive Threshold")
        print("    [D] Toggle debug view")
        print("=" * 60)
        print()
        
        while True:
            cap.grab()
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            gray_filtered = self.preprocess_frame(gray)
            
            corners, ids, _ = self.aruco_detector.detectMarkers(gray_filtered)
            
            marker_center = None
            marker_corners = None
            target_found_this_frame = False
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == TARGET_ID:
                        marker_corners = corners[i]
                        marker_center = corners[i][0].mean(axis=0).astype(int)
                        marker_center = tuple(marker_center)
                        target_found_this_frame = True
                        self.stable_corners = corners[i]
                        self.stable_id = TARGET_ID
                        self.frames_since_seen = 0
                        break
            
            # Temporal holdover: if marker not found in this frame, keep last position
            if not target_found_this_frame and self.stable_corners is not None:
                self.frames_since_seen += 1
                if self.frames_since_seen <= self.max_holdover_frames:
                    marker_corners = self.stable_corners
                    marker_center = self.stable_corners[0].mean(axis=0).astype(int)
                    marker_center = tuple(marker_center)
                    if ids is None:
                        ids = np.array([[self.stable_id]])
                        corners = [self.stable_corners]
                    else:
                        ids = np.append(ids, [[self.stable_id]], axis=0)
                        corners = list(corners) + [self.stable_corners]
                else:
                    self.stable_corners = None
            
            if not self.verified:
                if marker_corners is not None:
                    aruco.drawDetectedMarkers(frame, [marker_corners], np.array([[TARGET_ID]]))
                    cv2.circle(frame, marker_center, 8, (0, 255, 255), -1)
                
                if self.update_verification(marker_center):
                    continue
                
                frame = self.draw_verification_screen(frame, marker_center)
                
                cv2.imshow('ArUco MQTT Sender - Verification', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            if self.check_connection_timeout():
                self.send_command("home")
            
            cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 110), (0, 0, 0), -1)
            
            cv2.circle(frame, (FRAME_WIDTH - 20, 15), 8, (0, 255, 0), -1)
            cv2.putText(frame, "VERIFIED", (FRAME_WIDTH - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1)
            
            mqtt_status = "CONNECTED" if self.mqtt_connected else "DISCONNECTED"
            mqtt_color = (0, 255, 0) if self.mqtt_connected else (0, 0, 255)
            cv2.putText(frame, f"MQTT: {mqtt_status}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, mqtt_color, 1)
            
            if not self.mqtt_connected:
                elapsed = time.time() - self.last_successful_send
                remaining = max(0, CONNECTION_TIMEOUT - elapsed)
                if remaining > 0:
                    cv2.putText(frame, f"HOME in: {remaining:.1f}s", (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(frame, "HOME SENT", (10, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            
            send_status = "ACTIVE" if self.sending_enabled else "PAUSED"
            send_color = (0, 255, 0) if self.sending_enabled else (0, 165, 255)
            cv2.putText(frame, f"TX: {send_status}", (250, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, send_color, 1)
            
            cv2.putText(frame, f"Msgs: {self.message_count}", (400, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            target_marker_found = False
            unauthorized_markers = []
            if ids is not None:
                for marker_id in ids.flatten():
                    if marker_id == TARGET_ID:
                        target_marker_found = True
                    else:
                        unauthorized_markers.append(marker_id)
            
            self.check_marker_lost_timeout(target_marker_found)
            
            if unauthorized_markers:
                warning_text = f"WARNING: Unknown marker(s): {unauthorized_markers}"
                cv2.putText(frame, warning_text, (10, FRAME_HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if int(time.time() * 3) % 2 == 0:
                    cv2.rectangle(frame, (0, FRAME_HEIGHT - 35), (FRAME_WIDTH, FRAME_HEIGHT), 
                                  (0, 0, 100), -1)
                    cv2.putText(frame, warning_text, (10, FRAME_HEIGHT - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id == TARGET_ID:
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                            [corners[i]], MARKER_SIZE, self.camera_matrix, self.dist_coeffs
                        )
                        
                        tvec = tvecs[0][0]
                        x_cm = tvec[0] * 100
                        y_cm = tvec[1] * 100
                        z_cm = tvec[2] * 100
                        
                        self.last_position = (x_cm, y_cm, z_cm)
                        
                        if self.sending_enabled:
                            sent = self.send_position(x_cm, y_cm, z_cm)
                            if sent:
                                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                                print(f"[{ts}] TX -> X:{x_cm:+7.2f} Y:{y_cm:+7.2f} Z:{z_cm:+7.2f}")
                        
                        cv2.putText(frame, f"MARKER {TARGET_ID} FOUND",
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        cv2.putText(frame, f"X: {x_cm:+.1f} cm", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
                        cv2.putText(frame, f"Y: {y_cm:+.1f} cm", (180, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
                        cv2.putText(frame, f"Z: {z_cm:+.1f} cm", (350, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
                        
                        if self.sending_enabled and self.mqtt_connected:
                            cv2.circle(frame, (FRAME_WIDTH - 30, 50), 10, (0, 255, 0), -1)
                            cv2.putText(frame, "TX", (FRAME_WIDTH - 60, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs,
                                          rvecs[0], tvecs[0], MARKER_SIZE * 2)
            else:
                cv2.putText(frame, "NO MARKER DETECTED", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                if self.last_position:
                    x, y, z = self.last_position
                    cv2.putText(frame, f"Last: X:{x:+.1f} Y:{y:+.1f} Z:{z:+.1f}",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                marker_lost_elapsed = self.get_marker_lost_elapsed()
                if marker_lost_elapsed > 2.0:
                    remaining = max(0, MARKER_LOST_TIMEOUT - marker_lost_elapsed)
                    if remaining > 0:
                        bar_width = int((remaining / MARKER_LOST_TIMEOUT) * 200)
                        cv2.rectangle(frame, (10, 90), (210, 105), (50, 50, 50), -1)
                        cv2.rectangle(frame, (10, 90), (10 + bar_width, 105), (0, 165, 255), -1)
                        cv2.putText(frame, f"SAFE in: {remaining:.1f}s", (220, 103),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    else:
                        cv2.putText(frame, "!! SAFE MODE ACTIVATED !!", (10, 103),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.line(frame, (FRAME_WIDTH//2 - 20, FRAME_HEIGHT//2),
                     (FRAME_WIDTH//2 + 20, FRAME_HEIGHT//2), (50, 50, 50), 1)
            cv2.line(frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2 - 20),
                     (FRAME_WIDTH//2, FRAME_HEIGHT//2 + 20), (50, 50, 50), 1)
            
            filter_y = FRAME_HEIGHT - 25
            filter_status = f"Filters: {'ON' if FILTER_ENABLED else 'OFF'}"
            if FILTER_ENABLED:
                active = []
                if FILTER_CLAHE: active.append('C')
                if FILTER_DENOISE: active.append('D')
                if FILTER_SHARPEN: active.append('S')
                if FILTER_ADAPTIVE_THRESH: active.append('A')
                if active:
                    filter_status += f" [{'+'.join(active)}]"
            cv2.putText(frame, filter_status, (FRAME_WIDTH - 200, filter_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            cv2.imshow('ArUco MQTT Sender - [Q] Quit [SPACE] Toggle [F] Filters', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.sending_enabled = not self.sending_enabled
                status = "ACTIVE" if self.sending_enabled else "PAUSED"
                print(f"\n>>> Transmission: {status}\n")
            elif key == ord('h'):
                self.send_command("home")
            elif key == ord('s'):
                filename = f"mqtt_sender_{datetime.now().strftime('%H%M%S')}.png"
                cv2.imwrite(filename, frame)
                print(f">>> Screenshot saved: {filename}")
            elif key == ord('f'):
                self.toggle_filter('all')
            elif key == ord('1'):
                self.toggle_filter('clahe')
            elif key == ord('2'):
                self.toggle_filter('denoise')
            elif key == ord('3'):
                self.toggle_filter('sharpen')
            elif key == ord('4'):
                self.toggle_filter('adaptive')
            elif key == ord('d'):
                self.toggle_filter('debug')
        
        print("\nShutting down...")
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total messages sent: {self.message_count}")


if __name__ == '__main__':
    sender = ArucoMQTTSender()
    sender.run()
