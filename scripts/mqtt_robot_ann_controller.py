#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
MQTT Robot ANN Controller
=========================
TÜBİTAK 2209-B Projesi için ANN Entegreli Robot Kontrolcüsü

Bu controller ANN modülünü kullanarak:
1. Pozisyon smoothing (titreşim azaltma)
2. Pozisyon prediction (gecikme telafisi)
3. Karşılaştırma metrikleri (ANN vs Raw)

Ayrıca motion_recorder ile entegre çalışarak:
- Hareketleri kaydeder
- Kaydedilmiş hareketleri oynatır
- ANN eğitimi için veri toplar

Kullanım:
    roslaunch 3dof_rrr_robot_arm gazebo.launch
    rosrun 3dof_rrr_robot_arm mqtt_robot_ann_controller.py

Klavye Kontrolleri:
    [A] Toggle ANN (On/Off) - Karşılaştırma için
    [S] Toggle Smoothing
    [P] Toggle Prediction
    [R] Start/Stop Recording
    [Y] Playback Recording
    [T] Train ANN from recordings
    [M] Show metrics/comparison
    [L] Soldering effect
    [Q] Quit

Author: Emirtuğ Kacar
Date: 2024
"""

import rospy
import json
import time
import math
import threading
import numpy as np
import os
from collections import deque
from datetime import datetime

# ROS Messages
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Pose
from std_msgs.msg import String, Header
from gazebo_msgs.srv import SpawnModel, DeleteModel

# MQTT
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[WARN] paho-mqtt not installed")

# Keyboard
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("[WARN] pynput not installed")

# TF2 for transforms
try:
    import tf2_ros
    from tf2_geometry_msgs import PointStamped
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ANN Module
try:
    from ann_motion_predictor import ANNMotionPredictor, TF_AVAILABLE as ANN_TF_AVAILABLE
    ANN_AVAILABLE = True
except ImportError:
    ANN_AVAILABLE = False
    ANN_TF_AVAILABLE = False
    print("[WARN] ANN module not available")


# =============================================================================
# Configuration
# =============================================================================
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC_POSITION = "robot/position"

# ============ JOINT LIMITS (radians) - URDF ile uyumlu ============
JOINT1_MIN = -2.268  # URDF: -2.268
JOINT1_MAX = 2.268   # URDF: 2.268
JOINT2_MIN = 0.0
JOINT2_MAX = 1.57
JOINT3_MIN = 0.0
JOINT3_MAX = 1.57

# ============ REFERENCE BASED CONTROL (same as dual_controller) ============
CENTER_X = 0.0
CENTER_Y = 0.0
CENTER_Z = 10.0  # Log'a göre 8-12cm arası çalışıyorsun

RANGE_X = 10.0
RANGE_Y = 8.0
RANGE_Z = 8.0    # Daha hassas range

JOINT1_MOVE_LIMIT = 1.2   # URDF limiti 2.268
JOINT2_MOVE_LIMIT = 0.6   # Aşağı/yukarı hareket
JOINT3_MOVE_LIMIT = 0.6   # Aşağı/yukarı hareket

INVERT_X = -1
INVERT_Y = -1
INVERT_Z = -1   # DEĞİŞTİRİLDİ - Z arttıkça joint AZALSIN

HOME_POSITION = [0.0, 1.0, 1.0]  # Yukarıdan başla - aşağı inebilsin
SAFE_POSITION = [0.0, 0.5, 0.5]

# Tepki hızı parametreleri (lag azaltma)
SMOOTHING = 0.35          # Yüksek = hızlı tepki (0.25-0.5 arası)
CONTROL_RATE = 30         # Daha yüksek rate
DEADZONE = 0.2            # Düşük = daha hassas
TRAJECTORY_TIME = 0.08    # Düşük = hızlı hareket

# Model directory
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'models'
)
os.makedirs(MODEL_DIR, exist_ok=True)

# Recording directory
RECORDING_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'recordings'
)

# Spark SDF for soldering effect
SPARK_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="solder_spark">
    <static>false</static>
    <link name="spark_link">
      <gravity>false</gravity>
      <visual name="spark_visual">
        <geometry>
          <sphere><radius>0.008</radius></sphere>
        </geometry>
        <material>
          <ambient>1 0.5 0 1</ambient>
          <diffuse>1 0.7 0 1</diffuse>
          <emissive>1 0.8 0.2 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


class PerformanceMetrics:
    """Track and compare ANN vs Raw performance"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        
        # Position tracking
        self.raw_positions = deque(maxlen=window_size)
        self.ann_positions = deque(maxlen=window_size)
        self.actual_positions = deque(maxlen=window_size)
        
        # Latency
        self.raw_latencies = deque(maxlen=window_size)
        self.ann_latencies = deque(maxlen=window_size)
        
        # Smoothness (jerk)
        self.raw_jerks = deque(maxlen=window_size)
        self.ann_jerks = deque(maxlen=window_size)
        
        # Timestamps
        self.last_update = time.time()
    
    def add_sample(self, raw_pos, ann_pos, actual_pos, latency_raw=0, latency_ann=0):
        """Add a sample for comparison"""
        self.raw_positions.append(np.array(raw_pos))
        self.ann_positions.append(np.array(ann_pos))
        self.actual_positions.append(np.array(actual_pos))
        
        self.raw_latencies.append(latency_raw)
        self.ann_latencies.append(latency_ann)
        
        # Calculate jerk (smoothness) if enough samples
        if len(self.raw_positions) >= 4:
            raw_jerk = self._calculate_jerk(list(self.raw_positions)[-4:])
            ann_jerk = self._calculate_jerk(list(self.ann_positions)[-4:])
            self.raw_jerks.append(raw_jerk)
            self.ann_jerks.append(ann_jerk)
    
    def _calculate_jerk(self, positions):
        """Calculate jerk (rate of change of acceleration)"""
        if len(positions) < 4:
            return 0
        
        # Approximate jerk from 4 consecutive positions
        p0, p1, p2, p3 = positions
        
        # Velocity
        v0 = p1 - p0
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Acceleration
        a0 = v1 - v0
        a1 = v2 - v1
        
        # Jerk
        jerk = np.linalg.norm(a1 - a0)
        return jerk
    
    def get_stats(self):
        """Get comparison statistics"""
        if len(self.raw_positions) < 10:
            return None
        
        raw_positions = np.array(self.raw_positions)
        ann_positions = np.array(self.ann_positions)
        actual_positions = np.array(self.actual_positions)
        
        # Prediction error (if prediction is used)
        raw_errors = np.linalg.norm(raw_positions[1:] - actual_positions[:-1], axis=1)
        ann_errors = np.linalg.norm(ann_positions[1:] - actual_positions[:-1], axis=1)
        
        stats = {
            'samples': len(self.raw_positions),
            'raw': {
                'mean_error': np.mean(raw_errors) if len(raw_errors) > 0 else 0,
                'std_error': np.std(raw_errors) if len(raw_errors) > 0 else 0,
                'mean_latency': np.mean(self.raw_latencies) if self.raw_latencies else 0,
                'mean_jerk': np.mean(self.raw_jerks) if self.raw_jerks else 0,
            },
            'ann': {
                'mean_error': np.mean(ann_errors) if len(ann_errors) > 0 else 0,
                'std_error': np.std(ann_errors) if len(ann_errors) > 0 else 0,
                'mean_latency': np.mean(self.ann_latencies) if self.ann_latencies else 0,
                'mean_jerk': np.mean(self.ann_jerks) if self.ann_jerks else 0,
            }
        }
        
        # Improvement percentages
        if stats['raw']['mean_error'] > 0:
            stats['error_improvement'] = (
                (stats['raw']['mean_error'] - stats['ann']['mean_error']) / 
                stats['raw']['mean_error'] * 100
            )
        else:
            stats['error_improvement'] = 0
        
        if stats['raw']['mean_jerk'] > 0:
            stats['smoothness_improvement'] = (
                (stats['raw']['mean_jerk'] - stats['ann']['mean_jerk']) / 
                stats['raw']['mean_jerk'] * 100
            )
        else:
            stats['smoothness_improvement'] = 0
        
        return stats


class MQTTRobotANNController:
    """
    ANN-enhanced robot controller with recording capabilities.
    """
    
    def __init__(self):
        """Initialize the controller"""
        rospy.init_node('mqtt_robot_ann_controller', anonymous=True)
        
        # State
        self.current_position = None
        self.last_raw_position = None
        self.joint_angles = list(HOME_POSITION)  # Start at home
        self.running = True
        self.message_count = 0
        
        # Deadzone tracking
        self.last_x = CENTER_X
        self.last_y = CENTER_Y  
        self.last_z = CENTER_Z
        
        # ANN settings
        self.ann_enabled = True
        self.smoothing_enabled = True
        self.prediction_enabled = True
        
        # ANN module
        self.ann_predictor = None
        if ANN_AVAILABLE:
            self.ann_predictor = ANNMotionPredictor(sequence_length=10, prediction_horizon=1)
            self._load_ann_model()
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        
        # Recording state
        self.is_recording = False
        self.recorded_positions = []
        self.record_start_time = None
        
        # Playback state
        self.is_playing = False
        self.playback_positions = []
        self.playback_index = 0
        self.playback_thread = None
        
        # Soldering state
        self.is_soldering = False
        self.spark_spawned = False
        
        # Publishers
        self.trajectory_pub = rospy.Publisher(
            '/arm_controller/command',
            JointTrajectory,
            queue_size=1
        )
        
        # Gazebo services for spark
        try:
            rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=5.0)
            rospy.wait_for_service('/gazebo/delete_model', timeout=5.0)
            self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            self.delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            self.gazebo_available = True
        except:
            self.gazebo_available = False
            print("[WARN] Gazebo services not available")
        
        # MQTT
        self.mqtt_client = None
        self.mqtt_connected = False
        if MQTT_AVAILABLE:
            self._setup_mqtt()
        
        # Keyboard
        if KEYBOARD_AVAILABLE:
            self._setup_keyboard()
        
        # Print status
        self._print_status()
    
    def _print_status(self):
        """Print controller status"""
        print("\n" + "=" * 70)
        print("MQTT Robot ANN Controller")
        print("=" * 70)
        print(f"  ANN Module:     {'✓ Loaded' if ANN_AVAILABLE else '✗ Not available'}")
        print(f"  TensorFlow:     {'✓ Available' if ANN_TF_AVAILABLE else '✗ Using fallback'}")
        print(f"  MQTT:           {'✓ Connecting' if MQTT_AVAILABLE else '✗ Not available'}")
        print(f"  Gazebo:         {'✓ Connected' if self.gazebo_available else '✗ Not available'}")
        print("-" * 70)
        print("Controls:")
        print("  [A] ANN On/Off     [S] Smoothing     [P] Prediction")
        print("  [R] Record         [Y] Playback      [T] Train ANN")
        print("  [M] Metrics        [L] Solder        [Q] Quit")
        print("=" * 70)
        print(f"\nANN: {'ON' if self.ann_enabled else 'OFF'} | "
              f"Smooth: {'ON' if self.smoothing_enabled else 'OFF'} | "
              f"Predict: {'ON' if self.prediction_enabled else 'OFF'}")
        print("-" * 70 + "\n")
    
    def _load_ann_model(self):
        """Try to load pre-trained ANN model"""
        if not ANN_AVAILABLE:
            return
        
        # Look for most recent model
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_params.json')]
        
        if model_files:
            # Get most recent
            model_files.sort(reverse=True)
            base_name = model_files[0].replace('_params.json', '')
            model_path = os.path.join(MODEL_DIR, base_name)
            
            if self.ann_predictor.load_model(model_path):
                print(f"[ANN] Loaded model: {base_name}")
            else:
                print("[ANN] No pre-trained model loaded")
        else:
            print("[ANN] No pre-trained model found")
    
    def _setup_mqtt(self):
        """Setup MQTT connection"""
        self.mqtt_client = mqtt.Client(
            client_id=f"ann_controller_{int(time.time())}",
            protocol=mqtt.MQTTv311
        )
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
        try:
            self.mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.mqtt_connected = True
            self.mqtt_client.subscribe(MQTT_TOPIC_POSITION)
            print(f"[MQTT] Connected ✓")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        self.mqtt_connected = False
        print("[MQTT] Disconnected - Moving to SAFE position!")
        self._go_to_safe_position()
    
    def _go_to_safe_position(self):
        """Move robot to safe position"""
        print("[SAFE] Moving to safe position...")
        
        # Gradually move to safe position
        for _ in range(20):  # 20 steps
            self.joint_angles[0] += (SAFE_POSITION[0] - self.joint_angles[0]) * 0.2
            self.joint_angles[1] += (SAFE_POSITION[1] - self.joint_angles[1]) * 0.2
            self.joint_angles[2] += (SAFE_POSITION[2] - self.joint_angles[2]) * 0.2
            
            traj = JointTrajectory()
            traj.header.stamp = rospy.Time.now()
            traj.joint_names = ['joint1', 'joint2', 'joint3']
            
            point = JointTrajectoryPoint()
            point.positions = list(self.joint_angles)
            point.time_from_start = rospy.Duration(TRAJECTORY_TIME)
            traj.points.append(point)
            
            self.trajectory_pub.publish(traj)
            time.sleep(0.05)
        
        print("[SAFE] Robot at safe position")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT position messages"""
        try:
            data = json.loads(msg.payload.decode())
            
            if 'x' in data and 'y' in data and 'z' in data:
                x = data['x']
                y = data['y']
                z = data['z']
                
                # Deadzone (same as dual_controller)
                if abs(x - self.last_x) < DEADZONE:
                    x = self.last_x
                if abs(y - self.last_y) < DEADZONE:
                    y = self.last_y
                if abs(z - self.last_z) < DEADZONE:
                    z = self.last_z
                    
                self.last_x, self.last_y, self.last_z = x, y, z
                
                raw_position = (x, y, z)
                self.last_raw_position = raw_position
                
                # Process through ANN if enabled
                if self.ann_enabled and self.ann_predictor:
                    start_time = time.time()
                    
                    if self.smoothing_enabled and self.prediction_enabled:
                        smoothed, predicted = self.ann_predictor.smooth_and_predict(raw_position)
                        processed_position = predicted  # Use prediction
                    elif self.smoothing_enabled:
                        processed_position = self.ann_predictor.smooth(raw_position)
                    elif self.prediction_enabled:
                        processed_position = self.ann_predictor.predict_next(raw_position)
                    else:
                        processed_position = raw_position
                    
                    ann_latency = time.time() - start_time
                    
                    # Update metrics
                    self.metrics.add_sample(
                        raw_position, 
                        processed_position, 
                        raw_position,  # actual
                        latency_ann=ann_latency
                    )
                    
                    self.current_position = processed_position
                else:
                    self.current_position = raw_position
                
                # Record if active
                if self.is_recording:
                    self.recorded_positions.append({
                        'time': time.time(),
                        'raw': raw_position,
                        'processed': self.current_position
                    })
                
                # Move robot
                self._move_to_position(self.current_position)
                
                # Console output (every 20 messages)
                self.message_count += 1
                if self.message_count % 20 == 0:
                    ann_status = "ANN" if self.ann_enabled else "RAW"
                    print(f"[RX] X:{x:.1f} Y:{y:.1f} Z:{z:.1f} -> "
                          f"J1:{math.degrees(self.joint_angles[0]):.0f}° "
                          f"J2:{math.degrees(self.joint_angles[1]):.0f}° "
                          f"J3:{math.degrees(self.joint_angles[2]):.0f}° [{ann_status}]")
                
        except Exception as e:
            pass
    
    def _setup_keyboard(self):
        """Setup keyboard listener"""
        def on_press(key):
            try:
                char = key.char.lower() if hasattr(key, 'char') else None
                
                if char == 'a':
                    self.toggle_ann()
                elif char == 's':
                    self.toggle_smoothing()
                elif char == 'p':
                    self.toggle_prediction()
                elif char == 'r':
                    self.toggle_recording()
                elif char == 'y':
                    self.toggle_playback()
                elif char == 't':
                    self.train_ann()
                elif char == 'm':
                    self.show_metrics()
                elif char == 'l':
                    self.toggle_soldering()
                elif char == 'q':
                    self.shutdown()
                    
            except AttributeError:
                pass
        
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
    
    # =========================================================================
    # ANN Control
    # =========================================================================
    
    def toggle_ann(self):
        """Toggle ANN processing on/off"""
        self.ann_enabled = not self.ann_enabled
        status = "ON" if self.ann_enabled else "OFF"
        print(f"\n[ANN] Processing: {status}")
    
    def toggle_smoothing(self):
        """Toggle smoothing"""
        self.smoothing_enabled = not self.smoothing_enabled
        status = "ON" if self.smoothing_enabled else "OFF"
        print(f"[ANN] Smoothing: {status}")
    
    def toggle_prediction(self):
        """Toggle prediction"""
        self.prediction_enabled = not self.prediction_enabled
        status = "ON" if self.prediction_enabled else "OFF"
        print(f"[ANN] Prediction: {status}")
    
    def train_ann(self):
        """Train ANN from recorded data"""
        if not ANN_AVAILABLE or not ANN_TF_AVAILABLE:
            print("[WARN] TensorFlow not available for training")
            return
        
        # Check for training data
        export_path = os.path.join(RECORDING_DIR, 'exports', 'all_recordings_training.json')
        
        if not os.path.exists(export_path):
            # Use current recorded positions
            if len(self.recorded_positions) < 100:
                print("[WARN] Need at least 100 recorded positions for training")
                return
            
            positions = [p['raw'] for p in self.recorded_positions]
        else:
            # Load from file
            with open(export_path, 'r') as f:
                data = json.load(f)
            positions = [(p['x'], p['y'], p['z']) for p in data['positions']]
        
        print(f"\n[TRAIN] Training ANN on {len(positions)} positions...")
        
        # Train
        self.ann_predictor.train(positions, epochs=50, verbose=1)
        
        # Save model
        self.ann_predictor.save_model("motion_predictor")
        
        print("[TRAIN] Training complete!")
    
    def show_metrics(self):
        """Show comparison metrics"""
        stats = self.metrics.get_stats()
        
        print("\n" + "=" * 60)
        print("Performance Comparison: ANN vs Raw")
        print("=" * 60)
        
        if stats is None:
            print("Not enough data. Move the robot to collect samples.")
            print("=" * 60 + "\n")
            return
        
        print(f"Samples collected: {stats['samples']}")
        print("-" * 60)
        print(f"{'Metric':<25} {'Raw':<15} {'ANN':<15} {'Improvement'}")
        print("-" * 60)
        
        print(f"{'Mean Error':<25} {stats['raw']['mean_error']:.6f}     {stats['ann']['mean_error']:.6f}     " +
              f"{stats['error_improvement']:+.1f}%")
        
        print(f"{'Std Error':<25} {stats['raw']['std_error']:.6f}     {stats['ann']['std_error']:.6f}")
        
        print(f"{'Mean Jerk (Smoothness)':<25} {stats['raw']['mean_jerk']:.6f}     {stats['ann']['mean_jerk']:.6f}     " +
              f"{stats['smoothness_improvement']:+.1f}%")
        
        print(f"{'Mean Latency (ms)':<25} {stats['raw']['mean_latency']*1000:.2f}        {stats['ann']['mean_latency']*1000:.2f}")
        
        print("=" * 60)
        
        if stats['smoothness_improvement'] > 0:
            print(f"✓ ANN provides {stats['smoothness_improvement']:.1f}% smoother motion")
        else:
            print("○ ANN smoothing effect not significant with current data")
        
        print("=" * 60 + "\n")
    
    # =========================================================================
    # Recording
    # =========================================================================
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.is_playing:
            print("[WARN] Cannot record while playing")
            return
        
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start recording"""
        self.is_recording = True
        self.recorded_positions = []
        self.record_start_time = time.time()
        print("\n[REC] ● Recording started... Press 'R' to stop")
    
    def stop_recording(self):
        """Stop recording and save"""
        self.is_recording = False
        duration = time.time() - self.record_start_time
        
        print(f"[REC] ■ Recording stopped")
        print(f"[REC] Duration: {duration:.2f}s")
        print(f"[REC] Samples: {len(self.recorded_positions)}")
        
        # Save to file
        if self.recorded_positions:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(RECORDING_DIR, 'exports')
            os.makedirs(export_dir, exist_ok=True)
            
            filename = os.path.join(export_dir, f"recording_{timestamp}.json")
            
            export_data = {
                'metadata': {
                    'timestamp': timestamp,
                    'duration': duration,
                    'samples': len(self.recorded_positions),
                    'ann_enabled': self.ann_enabled
                },
                'positions': [
                    {
                        'time': p['time'],
                        'x': p['raw'][0],
                        'y': p['raw'][1],
                        'z': p['raw'][2]
                    }
                    for p in self.recorded_positions
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"[REC] Saved: {filename}")
    
    # =========================================================================
    # Playback
    # =========================================================================
    
    def toggle_playback(self):
        """Toggle playback on/off"""
        if self.is_recording:
            print("[WARN] Cannot playback while recording")
            return
        
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        """Start playback of last recording"""
        if not self.recorded_positions:
            # Try to load from file
            export_dir = os.path.join(RECORDING_DIR, 'exports')
            if os.path.exists(export_dir):
                files = sorted([f for f in os.listdir(export_dir) if f.startswith('recording_')])
                if files:
                    filepath = os.path.join(export_dir, files[-1])
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.playback_positions = [
                        (p['x'], p['y'], p['z'])
                        for p in data['positions']
                    ]
                    print(f"[PLAY] Loaded: {files[-1]}")
        else:
            self.playback_positions = [p['raw'] for p in self.recorded_positions]
        
        if not self.playback_positions:
            print("[WARN] No recording to play")
            return
        
        self.is_playing = True
        self.playback_index = 0
        
        print(f"\n[PLAY] ▶ Playing {len(self.playback_positions)} positions...")
        
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()
    
    def _playback_loop(self):
        """Playback thread"""
        rate = 30  # Hz
        interval = 1.0 / rate
        
        while self.is_playing and self.playback_index < len(self.playback_positions):
            position = self.playback_positions[self.playback_index]
            
            # Process through ANN if enabled
            if self.ann_enabled and self.ann_predictor:
                if self.smoothing_enabled:
                    position = self.ann_predictor.smooth(position)
                if self.prediction_enabled:
                    position = self.ann_predictor.predict_next()
            
            self._move_to_position(position)
            
            self.playback_index += 1
            time.sleep(interval)
            
            if self.playback_index % 100 == 0:
                progress = (self.playback_index / len(self.playback_positions)) * 100
                print(f"[PLAY] Progress: {progress:.0f}%")
        
        if self.is_playing:
            print("[PLAY] ■ Playback complete")
            self.is_playing = False
    
    def stop_playback(self):
        """Stop playback"""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        print("[PLAY] ■ Stopped")
    
    # =========================================================================
    # Robot Control
    # =========================================================================
    
    def _move_to_position(self, position):
        """Move robot to position using reference-based mapping (same as dual_controller)"""
        x, y, z = position
        
        # Reference based mapping (dual_controller ile aynı)
        offset_x = x - CENTER_X
        offset_y = y - CENTER_Y
        offset_z = z - CENTER_Z
        
        # Normalize to -1..1
        norm_x = max(-1.0, min(1.0, offset_x / RANGE_X))
        norm_y = max(-1.0, min(1.0, offset_y / RANGE_Y))
        norm_z = max(-1.0, min(1.0, offset_z / RANGE_Z))
        
        # Apply inversion
        norm_x *= INVERT_X
        norm_y *= INVERT_Y
        norm_z *= INVERT_Z
        
        # Map to joint angles
        joint1 = HOME_POSITION[0] + norm_x * JOINT1_MOVE_LIMIT
        joint2 = HOME_POSITION[1] + norm_z * JOINT2_MOVE_LIMIT
        joint3 = HOME_POSITION[2] + norm_y * JOINT3_MOVE_LIMIT
        
        # Apply joint limits
        joint1 = max(JOINT1_MIN, min(JOINT1_MAX, joint1))
        joint2 = max(JOINT2_MIN, min(JOINT2_MAX, joint2))
        joint3 = max(JOINT3_MIN, min(JOINT3_MAX, joint3))
        
        # Smoothing
        self.joint_angles[0] += (joint1 - self.joint_angles[0]) * SMOOTHING
        self.joint_angles[1] += (joint2 - self.joint_angles[1]) * SMOOTHING
        self.joint_angles[2] += (joint3 - self.joint_angles[2]) * SMOOTHING
        
        # Publish trajectory
        traj = JointTrajectory()
        traj.header.stamp = rospy.Time.now()
        traj.joint_names = ['joint1', 'joint2', 'joint3']
        
        point = JointTrajectoryPoint()
        point.positions = self.joint_angles
        point.time_from_start = rospy.Duration(TRAJECTORY_TIME)
        traj.points.append(point)
        
        self.trajectory_pub.publish(traj)
    
    # =========================================================================
    # Soldering Effect
    # =========================================================================
    
    def toggle_soldering(self):
        """Toggle soldering effect"""
        self.is_soldering = not self.is_soldering
        
        if self.is_soldering:
            print("[SOLDER] Soldering ON")
            # Spawn single spark
            self._spawn_spark()
        else:
            print("[SOLDER] Soldering OFF")
            self._delete_spark()
    
    def _spawn_spark(self):
        """Spawn spark effect at tool tip"""
        if not self.gazebo_available:
            return
        
        # Delete old spark if exists
        if self.spark_spawned:
            self._delete_spark()
        
        try:
            pose = Pose()
            pose.position.x = -0.05
            pose.position.y = 0.06
            pose.position.z = 0.25
            pose.orientation.w = 1.0
            
            self.spawn_model(
                model_name="solder_spark",
                model_xml=SPARK_SDF,
                robot_namespace="",
                initial_pose=pose,
                reference_frame="3dof_rrr_robot_arm::link3"
            )
            
            self.spark_spawned = True
            print("[SOLDER] Spark spawned")
            
        except Exception as e:
            rospy.logwarn(f"Spark spawn error: {e}")
    
    def _delete_spark(self):
        """Delete spark effect"""
        if not self.gazebo_available:
            return
        
        if not self.spark_spawned:
            return
        
        try:
            self.delete_model(model_name="solder_spark")
            self.spark_spawned = False
            print("[SOLDER] Spark removed")
        except Exception as e:
            # Spark may already be deleted
            self.spark_spawned = False
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n[EXIT] Shutting down...")
        self.running = False
        
        if self.is_recording:
            self.stop_recording()
        
        if self.is_playing:
            self.stop_playback()
        
        if self.is_soldering:
            self.is_soldering = False
            self._delete_spark()
        
        # Go to safe position before shutdown
        self._go_to_safe_position()
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        rospy.signal_shutdown("User shutdown")
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)
        
        try:
            while not rospy.is_shutdown() and self.running:
                rate.sleep()
        except KeyboardInterrupt:
            self.shutdown()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    try:
        controller = MQTTRobotANNController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
