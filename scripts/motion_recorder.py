#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
Motion Recorder - ROS Bag Based Recording/Playback System
==========================================================
TÜBİTAK 2209-B Projesi için Hareket Kayıt ve Oynatma Sistemi

Bu modül:
1. MQTT'den gelen pozisyonları ROS bag dosyasına kaydeder
2. Kaydedilmiş hareketleri geri oynatır
3. ANN eğitimi için veri export eder
4. Kayıt metadata yönetimi

Kullanım:
    rosrun 3dof_rrr_robot_arm motion_recorder.py

Klavye Kontrolleri:
    [R] - Kayıt başlat/durdur
    [P] - Oynatma başlat/durdur
    [L] - Kayıtları listele
    [E] - JSON export (ANN eğitimi için)
    [D] - Kayıt sil
    [Q] - Çıkış

Author: Emirtuğ Kacar
Date: 2024
"""

import rospy
import rosbag
import os
import json
import time
import threading
from datetime import datetime
from collections import deque

# ROS Messages
from std_msgs.msg import Float64MultiArray, String, Header
from geometry_msgs.msg import Point, PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# MQTT
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[WARN] paho-mqtt not installed: pip install paho-mqtt")

# Keyboard
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("[WARN] pynput not installed: pip install pynput")


# =============================================================================
# Configuration
# =============================================================================
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC_POSITION = "robot/position"
MQTT_TOPIC_COMMAND = "robot/command"

# Recording settings
RECORDING_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'recordings'
)
os.makedirs(RECORDING_DIR, exist_ok=True)


class MotionRecorder:
    """
    ROS Bag based motion recording and playback system.
    
    Records:
    - Raw MQTT positions
    - Joint angles
    - Timestamps
    - Metadata (duration, sample count, etc.)
    """
    
    def __init__(self):
        """Initialize the motion recorder"""
        rospy.init_node('motion_recorder', anonymous=True)
        
        # State
        self.is_recording = False
        self.is_playing = False
        self.current_bag = None
        self.current_bag_path = None
        
        # Position data
        self.position_buffer = deque(maxlen=10000)
        self.current_position = None
        self.joint_angles = [0.0, 0.0, 0.0]
        
        # Recording stats
        self.record_start_time = None
        self.sample_count = 0
        
        # Playback
        self.playback_thread = None
        self.playback_positions = []
        self.playback_index = 0
        self.playback_speed = 1.0
        
        # Publishers
        self.position_pub = rospy.Publisher(
            '/recorded_position', 
            PointStamped, 
            queue_size=10
        )
        self.trajectory_pub = rospy.Publisher(
            '/arm_controller/command',
            JointTrajectory,
            queue_size=1
        )
        self.status_pub = rospy.Publisher(
            '/recorder_status',
            String,
            queue_size=10
        )
        
        # MQTT Setup
        self.mqtt_client = None
        self.mqtt_connected = False
        if MQTT_AVAILABLE:
            self._setup_mqtt()
        
        # Keyboard listener
        self.keyboard_listener = None
        if KEYBOARD_AVAILABLE:
            self._setup_keyboard()
        
        # Index file for recordings
        self.index_file = os.path.join(RECORDING_DIR, 'recordings_index.json')
        self.recordings_index = self._load_index()
        
        print("\n" + "=" * 60)
        print("Motion Recorder - ROS Bag System")
        print("=" * 60)
        print(f"Recording directory: {RECORDING_DIR}")
        print(f"Total recordings: {len(self.recordings_index)}")
        print("-" * 60)
        print("Controls:")
        print("  [R] Record    [P] Playback    [L] List")
        print("  [E] Export    [D] Delete      [Q] Quit")
        print("=" * 60 + "\n")
    
    def _setup_mqtt(self):
        """Setup MQTT connection"""
        self.mqtt_client = mqtt.Client(
            client_id=f"motion_recorder_{int(time.time())}",
            protocol=mqtt.MQTTv311
        )
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        
        try:
            self.mqtt_client.connect_async(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self.mqtt_client.loop_start()
            print("[MQTT] Connecting...")
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.mqtt_connected = True
            self.mqtt_client.subscribe(MQTT_TOPIC_POSITION)
            print(f"[MQTT] Connected, subscribed to {MQTT_TOPIC_POSITION}")
        else:
            print(f"[MQTT] Connection failed: {rc}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            data = json.loads(msg.payload.decode())
            
            if 'x' in data and 'y' in data and 'z' in data:
                position = (data['x'], data['y'], data['z'])
                self.current_position = position
                
                # Record if active
                if self.is_recording and self.current_bag:
                    self._record_position(position)
                    
        except Exception as e:
            pass  # Ignore parse errors
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback"""
        self.mqtt_connected = False
        print("[MQTT] Disconnected")
    
    def _setup_keyboard(self):
        """Setup keyboard listener"""
        def on_press(key):
            try:
                char = key.char.lower() if hasattr(key, 'char') else None
                
                if char == 'r':
                    self.toggle_recording()
                elif char == 'p':
                    self.toggle_playback()
                elif char == 'l':
                    self.list_recordings()
                elif char == 'e':
                    self.export_for_training()
                elif char == 'd':
                    self.delete_recording()
                elif char == 'q':
                    self.shutdown()
                    
            except AttributeError:
                pass
        
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
    
    def _load_index(self):
        """Load recordings index"""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save recordings index"""
        with open(self.index_file, 'w') as f:
            json.dump(self.recordings_index, f, indent=2)
    
    def _generate_recording_name(self):
        """Generate unique recording name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"motion_{timestamp}"
    
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
        """Start recording to bag file"""
        if self.is_recording:
            return
        
        name = self._generate_recording_name()
        bag_path = os.path.join(RECORDING_DIR, f"{name}.bag")
        
        try:
            self.current_bag = rosbag.Bag(bag_path, 'w')
            self.current_bag_path = bag_path
            self.is_recording = True
            self.record_start_time = rospy.Time.now()
            self.sample_count = 0
            self.position_buffer.clear()
            
            print(f"\n[REC] ● Recording started: {name}")
            print("[REC] Press 'R' to stop...")
            
            self.status_pub.publish(String(data=f"recording:{name}"))
            
        except Exception as e:
            print(f"[ERR] Failed to start recording: {e}")
    
    def _record_position(self, position):
        """Record a position sample"""
        if not self.is_recording or not self.current_bag:
            return
        
        try:
            # Create PointStamped message
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "aruco_marker"
            msg.point.x = position[0]
            msg.point.y = position[1]
            msg.point.z = position[2]
            
            # Write to bag
            self.current_bag.write('/motion/position', msg)
            
            # Buffer for in-memory access
            self.position_buffer.append({
                'time': msg.header.stamp.to_sec(),
                'position': position
            })
            
            self.sample_count += 1
            
            # Progress indicator every 100 samples
            if self.sample_count % 100 == 0:
                duration = (rospy.Time.now() - self.record_start_time).to_sec()
                print(f"[REC] Samples: {self.sample_count} | Duration: {duration:.1f}s")
                
        except Exception as e:
            print(f"[ERR] Recording error: {e}")
    
    def stop_recording(self):
        """Stop recording and save"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        duration = (rospy.Time.now() - self.record_start_time).to_sec()
        
        # Close bag file
        if self.current_bag:
            self.current_bag.close()
        
        # Extract name from path
        name = os.path.basename(self.current_bag_path).replace('.bag', '')
        
        # Save to index
        self.recordings_index[name] = {
            'path': self.current_bag_path,
            'created': datetime.now().isoformat(),
            'duration': duration,
            'samples': self.sample_count,
            'sample_rate': self.sample_count / duration if duration > 0 else 0
        }
        self._save_index()
        
        print(f"\n[REC] ■ Recording stopped")
        print(f"[REC] Duration: {duration:.2f}s")
        print(f"[REC] Samples: {self.sample_count}")
        print(f"[REC] Saved: {self.current_bag_path}")
        
        self.status_pub.publish(String(data="recording:stopped"))
        
        self.current_bag = None
        self.current_bag_path = None
    
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
    
    def start_playback(self, recording_name=None):
        """Start playback of a recording"""
        if self.is_playing:
            return
        
        # Get recording to play
        if recording_name is None:
            if not self.recordings_index:
                print("[WARN] No recordings available")
                return
            # Use most recent recording
            recording_name = sorted(self.recordings_index.keys())[-1]
        
        if recording_name not in self.recordings_index:
            print(f"[WARN] Recording not found: {recording_name}")
            return
        
        recording = self.recordings_index[recording_name]
        bag_path = recording['path']
        
        print(f"\n[PLAY] ▶ Playing: {recording_name}")
        print(f"[PLAY] Duration: {recording['duration']:.2f}s")
        print(f"[PLAY] Press 'P' to stop...")
        
        # Load positions from bag
        self.playback_positions = self._load_positions_from_bag(bag_path)
        
        if not self.playback_positions:
            print("[WARN] No positions in recording")
            return
        
        self.is_playing = True
        self.playback_index = 0
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_loop)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
        self.status_pub.publish(String(data=f"playing:{recording_name}"))
    
    def _load_positions_from_bag(self, bag_path):
        """Load positions from bag file"""
        positions = []
        
        try:
            bag = rosbag.Bag(bag_path, 'r')
            
            for topic, msg, t in bag.read_messages(topics=['/motion/position']):
                positions.append({
                    'time': msg.header.stamp.to_sec(),
                    'position': (msg.point.x, msg.point.y, msg.point.z)
                })
            
            bag.close()
            print(f"[PLAY] Loaded {len(positions)} positions")
            
        except Exception as e:
            print(f"[ERR] Failed to load bag: {e}")
        
        return positions
    
    def _playback_loop(self):
        """Playback thread loop"""
        if not self.playback_positions:
            return
        
        start_time = time.time()
        first_sample_time = self.playback_positions[0]['time']
        
        while self.is_playing and self.playback_index < len(self.playback_positions):
            sample = self.playback_positions[self.playback_index]
            
            # Calculate when to play this sample
            sample_relative_time = (sample['time'] - first_sample_time) / self.playback_speed
            current_relative_time = time.time() - start_time
            
            # Wait if needed
            if sample_relative_time > current_relative_time:
                time.sleep(sample_relative_time - current_relative_time)
            
            if not self.is_playing:
                break
            
            # Publish position
            position = sample['position']
            
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "playback"
            msg.point.x = position[0]
            msg.point.y = position[1]
            msg.point.z = position[2]
            self.position_pub.publish(msg)
            
            # Also publish to MQTT for robot control
            if self.mqtt_connected:
                mqtt_msg = json.dumps({
                    'x': position[0],
                    'y': position[1],
                    'z': position[2],
                    'source': 'playback'
                })
                self.mqtt_client.publish(MQTT_TOPIC_POSITION, mqtt_msg)
            
            self.playback_index += 1
            
            # Progress
            if self.playback_index % 100 == 0:
                progress = (self.playback_index / len(self.playback_positions)) * 100
                print(f"[PLAY] Progress: {progress:.1f}%")
        
        # Playback complete
        if self.is_playing:
            print("[PLAY] ■ Playback complete")
            self.is_playing = False
            self.status_pub.publish(String(data="playing:stopped"))
    
    def stop_playback(self):
        """Stop playback"""
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)
        print("[PLAY] ■ Stopped")
        self.status_pub.publish(String(data="playing:stopped"))
    
    # =========================================================================
    # Management
    # =========================================================================
    
    def list_recordings(self):
        """List all recordings"""
        print("\n" + "=" * 60)
        print("Available Recordings")
        print("=" * 60)
        
        if not self.recordings_index:
            print("  No recordings found")
        else:
            for i, (name, info) in enumerate(sorted(self.recordings_index.items())):
                print(f"  [{i+1}] {name}")
                print(f"      Duration: {info['duration']:.2f}s")
                print(f"      Samples:  {info['samples']}")
                print(f"      Rate:     {info['sample_rate']:.1f} Hz")
                print()
        
        print("=" * 60 + "\n")
    
    def delete_recording(self, name=None):
        """Delete a recording"""
        if not self.recordings_index:
            print("[WARN] No recordings to delete")
            return
        
        if name is None:
            self.list_recordings()
            print("Enter recording number to delete (or 'c' to cancel):")
            # In actual use, would need input handling
            # For now, delete the oldest
            name = sorted(self.recordings_index.keys())[0]
        
        if name in self.recordings_index:
            info = self.recordings_index[name]
            
            # Delete bag file
            if os.path.exists(info['path']):
                os.remove(info['path'])
            
            # Remove from index
            del self.recordings_index[name]
            self._save_index()
            
            print(f"[DEL] Deleted: {name}")
    
    def export_for_training(self, name=None, output_format='json'):
        """
        Export recording for ANN training.
        
        Args:
            name: Recording name (None = most recent)
            output_format: 'json' or 'csv'
        """
        if not self.recordings_index:
            print("[WARN] No recordings to export")
            return None
        
        if name is None:
            name = sorted(self.recordings_index.keys())[-1]
        
        if name not in self.recordings_index:
            print(f"[WARN] Recording not found: {name}")
            return None
        
        recording = self.recordings_index[name]
        positions = self._load_positions_from_bag(recording['path'])
        
        if not positions:
            print("[WARN] No positions to export")
            return None
        
        # Export directory
        export_dir = os.path.join(RECORDING_DIR, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        if output_format == 'json':
            export_path = os.path.join(export_dir, f"{name}_training.json")
            
            export_data = {
                'metadata': {
                    'recording_name': name,
                    'duration': recording['duration'],
                    'samples': recording['samples'],
                    'sample_rate': recording['sample_rate'],
                    'exported': datetime.now().isoformat()
                },
                'positions': [
                    {
                        'time': p['time'],
                        'x': p['position'][0],
                        'y': p['position'][1],
                        'z': p['position'][2]
                    }
                    for p in positions
                ]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"[EXPORT] Saved: {export_path}")
            print(f"[EXPORT] Positions: {len(positions)}")
            
            return export_path
        
        elif output_format == 'csv':
            export_path = os.path.join(export_dir, f"{name}_training.csv")
            
            with open(export_path, 'w') as f:
                f.write("time,x,y,z\n")
                for p in positions:
                    f.write(f"{p['time']},{p['position'][0]},{p['position'][1]},{p['position'][2]}\n")
            
            print(f"[EXPORT] Saved: {export_path}")
            return export_path
        
        return None
    
    def export_all_for_training(self):
        """Export all recordings into a single training file"""
        if not self.recordings_index:
            print("[WARN] No recordings to export")
            return None
        
        all_positions = []
        
        for name in sorted(self.recordings_index.keys()):
            recording = self.recordings_index[name]
            positions = self._load_positions_from_bag(recording['path'])
            
            for p in positions:
                all_positions.append({
                    'time': p['time'],
                    'x': p['position'][0],
                    'y': p['position'][1],
                    'z': p['position'][2],
                    'recording': name
                })
        
        export_dir = os.path.join(RECORDING_DIR, 'exports')
        os.makedirs(export_dir, exist_ok=True)
        
        export_path = os.path.join(export_dir, 'all_recordings_training.json')
        
        export_data = {
            'metadata': {
                'recording_count': len(self.recordings_index),
                'total_samples': len(all_positions),
                'exported': datetime.now().isoformat()
            },
            'positions': all_positions
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[EXPORT] Saved combined training data: {export_path}")
        print(f"[EXPORT] Total positions: {len(all_positions)}")
        
        return export_path
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n[EXIT] Shutting down...")
        
        if self.is_recording:
            self.stop_recording()
        
        if self.is_playing:
            self.stop_playback()
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        rospy.signal_shutdown("User requested shutdown")
    
    def run(self):
        """Main loop"""
        rate = rospy.Rate(10)  # 10 Hz status update
        
        try:
            while not rospy.is_shutdown():
                # Status display
                status = []
                
                if self.is_recording:
                    duration = (rospy.Time.now() - self.record_start_time).to_sec()
                    status.append(f"[REC] {duration:.1f}s / {self.sample_count} samples")
                
                if self.is_playing:
                    progress = (self.playback_index / len(self.playback_positions)) * 100 if self.playback_positions else 0
                    status.append(f"[PLAY] {progress:.1f}%")
                
                if status:
                    # Could publish to topic for UI
                    pass
                
                rate.sleep()
                
        except KeyboardInterrupt:
            self.shutdown()


# =============================================================================
# Training Data Utilities
# =============================================================================

def load_training_data(json_path):
    """
    Load training data from exported JSON file.
    
    Args:
        json_path: Path to exported JSON file
        
    Returns:
        List of (x, y, z) tuples
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    positions = [
        (p['x'], p['y'], p['z'])
        for p in data['positions']
    ]
    
    return positions


def train_ann_from_recordings(recording_dir=RECORDING_DIR):
    """
    Train ANN from all recordings.
    
    Returns:
        Trained ANNMotionPredictor
    """
    # Import ANN module
    from ann_motion_predictor import ANNMotionPredictor
    
    # Load all exported data
    export_path = os.path.join(recording_dir, 'exports', 'all_recordings_training.json')
    
    if not os.path.exists(export_path):
        print("[WARN] No combined training data found. Export recordings first.")
        return None
    
    positions = load_training_data(export_path)
    
    if len(positions) < 100:
        print("[WARN] Not enough training data (need at least 100 positions)")
        return None
    
    # Create and train predictor
    predictor = ANNMotionPredictor(sequence_length=10, prediction_horizon=1)
    predictor.train(positions, epochs=50, verbose=1)
    
    # Save model
    predictor.save_model("trained_motion_predictor")
    
    return predictor


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    try:
        recorder = MotionRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass
