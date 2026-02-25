#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3DoF RRR Robot Arm - Real Robot Controller
Dynamixel servo motorları kontrol eden script
"""

import rospy
import yaml
import math
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Dynamixel SDK
try:
    from dynamixel_sdk import *
    DYNAMIXEL_AVAILABLE = True
except ImportError:
    rospy.logwarn("dynamixel_sdk bulunamadi! pip install dynamixel-sdk")
    DYNAMIXEL_AVAILABLE = False

# Dynamixel Control Table (Protocol 2.0 - XL430)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_MOVING_SPEED = 112
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_TEMPERATURE = 146

class RobotController:
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=True)
        
        # Load config
        config_path = rospy.get_param('~config', 
            '/home/emirtug/catkin_ws/src/3dof_rrr_robot_arm/config/dynamixel_config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dxl_config = self.config['dynamixel']
        
        # Motor IDs
        self.motor_ids = {
            'joint1': self.dxl_config['motors']['joint1']['id'],
            'joint2': self.dxl_config['motors']['joint2']['id'],
            'joint3': self.dxl_config['motors']['joint3']['id']
        }
        
        # Publishers
        self.joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        
        # Subscribers
        self.cmd_sub = rospy.Subscriber('/robot_command', Float64MultiArray, self.command_callback)
        
        # Initialize Dynamixel
        if DYNAMIXEL_AVAILABLE:
            self.init_dynamixel()
        else:
            rospy.logwarn("Simülasyon modunda çalışıyor (Dynamixel yok)")
            self.port_handler = None
        
        rospy.loginfo("Robot Controller başlatıldı!")
        rospy.loginfo("Komut topic: /robot_command (Float64MultiArray - [j1, j2, j3] radyan)")
        
    def init_dynamixel(self):
        """Dynamixel portunu başlat"""
        port = self.dxl_config['port']
        baudrate = self.dxl_config['baudrate']
        
        self.port_handler = PortHandler(port)
        self.packet_handler = PacketHandler(2.0)  # Protocol 2.0
        
        if not self.port_handler.openPort():
            rospy.logerr(f"Port açılamadı: {port}")
            rospy.logerr("Lütfen kontrol edin:")
            rospy.logerr("  1. USB bağlı mı?")
            rospy.logerr("  2. sudo chmod 666 /dev/ttyACM0")
            self.port_handler = None
            return
            
        if not self.port_handler.setBaudRate(baudrate):
            rospy.logerr(f"Baudrate ayarlanamadı: {baudrate}")
            return
            
        rospy.loginfo(f"Dynamixel port açıldı: {port} @ {baudrate}")
        
        # Torque enable
        self.enable_torque(True)
        
    def enable_torque(self, enable):
        """Tüm motorların torque'unu aç/kapat"""
        if not self.port_handler:
            return
            
        value = 1 if enable else 0
        for joint, motor_id in self.motor_ids.items():
            result, error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, ADDR_TORQUE_ENABLE, value)
            if result != COMM_SUCCESS:
                rospy.logwarn(f"{joint} (ID:{motor_id}) torque ayarlanamadı")
            else:
                rospy.loginfo(f"{joint} (ID:{motor_id}) torque: {'ON' if enable else 'OFF'}")
    
    def rad_to_position(self, joint_name, rad):
        """Radyan -> Dynamixel pozisyon değeri"""
        motor = self.dxl_config['motors'][joint_name]
        min_angle = motor['min_angle']
        max_angle = motor['max_angle']
        min_pos = motor['min_position']
        max_pos = motor['max_position']
        
        # Limit kontrolü
        rad = max(min_angle, min(max_angle, rad))
        
        # Linear interpolation
        position = int(min_pos + (rad - min_angle) * (max_pos - min_pos) / (max_angle - min_angle))
        return position
    
    def position_to_rad(self, joint_name, position):
        """Dynamixel pozisyon -> Radyan"""
        motor = self.dxl_config['motors'][joint_name]
        min_angle = motor['min_angle']
        max_angle = motor['max_angle']
        min_pos = motor['min_position']
        max_pos = motor['max_position']
        
        # Linear interpolation
        rad = min_angle + (position - min_pos) * (max_angle - min_angle) / (max_pos - min_pos)
        return rad
    
    def set_joint_position(self, joint_name, rad):
        """Tek joint pozisyon ayarla"""
        if not self.port_handler:
            return
            
        motor_id = self.motor_ids[joint_name]
        position = self.rad_to_position(joint_name, rad)
        
        result, error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_POSITION, position)
        
        if result != COMM_SUCCESS:
            rospy.logwarn(f"{joint_name} pozisyon yazılamadı: {self.packet_handler.getTxRxResult(result)}")
    
    def get_joint_position(self, joint_name):
        """Tek joint pozisyon oku"""
        if not self.port_handler:
            return 0.0
            
        motor_id = self.motor_ids[joint_name]
        
        position, result, error = self.packet_handler.read4ByteTxRx(
            self.port_handler, motor_id, ADDR_PRESENT_POSITION)
        
        if result != COMM_SUCCESS:
            rospy.logwarn(f"{joint_name} pozisyon okunamadı")
            return 0.0
            
        return self.position_to_rad(joint_name, position)
    
    def command_callback(self, msg):
        """Hareket komutu callback"""
        if len(msg.data) < 3:
            rospy.logwarn("Komut 3 değer içermeli: [j1, j2, j3]")
            return
            
        j1, j2, j3 = msg.data[0], msg.data[1], msg.data[2]
        rospy.loginfo(f"Komut alındı: j1={j1:.2f}, j2={j2:.2f}, j3={j3:.2f}")
        
        self.set_joint_position('joint1', j1)
        self.set_joint_position('joint2', j2)
        self.set_joint_position('joint3', j3)
    
    def publish_joint_states(self):
        """Joint state publish et"""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['joint1', 'joint2', 'joint3']
        
        if self.port_handler:
            msg.position = [
                self.get_joint_position('joint1'),
                self.get_joint_position('joint2'),
                self.get_joint_position('joint3')
            ]
        else:
            msg.position = [0.0, 0.0, 0.0]
            
        msg.velocity = [0.0, 0.0, 0.0]
        msg.effort = [0.0, 0.0, 0.0]
        
        self.joint_state_pub.publish(msg)
    
    def home(self):
        """Home pozisyonuna git"""
        rospy.loginfo("Home pozisyonuna gidiliyor...")
        for joint_name in ['joint1', 'joint2', 'joint3']:
            home_pos = self.dxl_config['motors'][joint_name]['home_position']
            rad = self.position_to_rad(joint_name, home_pos)
            self.set_joint_position(joint_name, rad)
    
    def run(self):
        """Ana döngü"""
        rate = rospy.Rate(50)  # 50 Hz
        
        while not rospy.is_shutdown():
            self.publish_joint_states()
            rate.sleep()
    
    def shutdown(self):
        """Kapatma"""
        rospy.loginfo("Kapatılıyor...")
        if self.port_handler:
            self.enable_torque(False)
            self.port_handler.closePort()


def main():
    controller = None
    try:
        controller = RobotController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if controller:
            controller.shutdown()

if __name__ == '__main__':
    main()
