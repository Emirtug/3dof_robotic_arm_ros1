#!/usr/bin/env python3
"""
MoveIt Direct Angle Sync (ROS1 Version)
========================================
- Reads angles (radians) from MoveIt in real-time.
- Performs calibration in degrees for mechanical differences.
- Hides calibration offset in terminal output (shows clean angle).
"""

import rospy
from sensor_msgs.msg import JointState
from dynamixel_sdk import *
import numpy as np
import sys

# ==================== DYNAMIXEL SETTINGS ====================
ADDR_GOAL_POSITION = 116
ADDR_TORQUE_ENABLE = 64
DXL_IDS = [3, 2, 1]  # ID3=joint1, ID2=joint2, ID1=joint3
OFFSETS = [512, 3053, 2038]
DIRECTIONS = [1, -1, 1]

# ==================== CALIBRATION (FINE TUNING) ====================
# [ID3_Calibration, ID2_Calibration, ID1_Calibration] (in degrees)
CALIBRATION_DEGREES = [0.0, 9.0, 0.0] 

# ==================== SMOOTHING SETTINGS ====================
ALPHA = 0.04  # Approximately 2 seconds settling time


class MoveItDirectSync:
    def __init__(self):
        rospy.init_node('moveit_direct_sync')
        
        # Dynamixel Setup
        self.port_handler = PortHandler('/dev/ttyACM0')
        self.packet_handler = PacketHandler(2.0)
        self.group_sync_write = GroupSyncWrite(self.port_handler, self.packet_handler, ADDR_GOAL_POSITION, 4)
        
        if self.port_handler.openPort() and self.port_handler.setBaudRate(57600):
            for dxl_id in DXL_IDS:
                self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1)
            self.connected = True
            rospy.loginfo("✓ Motors connected.")
        else:
            self.connected = False
            rospy.logerr("✗ Motor connection failed!")

        # State Variables
        self.target_angles = [0.0, 0.0, 0.0]
        self.current_angles = [0.0, 0.0, 0.0]
        
        if self.connected:
            _, start_rads = self.read_actual_angles()
            self.current_angles = list(start_rads)
            self.target_angles = list(start_rads)
        
        self.subscription = rospy.Subscriber('/joint_states', JointState, self.listener_callback)
        self.timer = rospy.Timer(rospy.Duration(0.02), self.control_loop)
        
        print("\n" + "="*60)
        print("   SMOOTH CHASER + CLEAN OUTPUT ACTIVE")
        print(f"   J2 Calibration (+{CALIBRATION_DEGREES[1]}°) added to motor but hidden on screen.")
        print("="*60)

    def read_actual_angles(self):
        rads = []
        raws = []
        for i, dxl_id in enumerate(DXL_IDS):
            pos, _, _ = self.packet_handler.read4ByteTxRx(self.port_handler, dxl_id, 132)
            if pos is None: pos = OFFSETS[i]
            raws.append(pos)
            rads.append((pos - OFFSETS[i]) * (np.pi / 2048.0) * DIRECTIONS[i])
        return raws, rads

    def listener_callback(self, msg):
        try:
            j1_idx = msg.name.index('joint1')
            j2_idx = msg.name.index('joint2')
            j3_idx = msg.name.index('joint3')
            
            # Add calibration to target that will be sent to motor
            self.target_angles = [
                msg.position[j1_idx] + np.radians(CALIBRATION_DEGREES[0]),
                msg.position[j2_idx] + np.radians(CALIBRATION_DEGREES[1]),
                msg.position[j3_idx] + np.radians(CALIBRATION_DEGREES[2])
            ]
        except (ValueError, IndexError):
            pass

    def control_loop(self, event):
        for i in range(3):
            self.current_angles[i] += (self.target_angles[i] - self.current_angles[i]) * ALPHA
        
        if self.connected:
            self.send_to_motors(self.current_angles)
            
        # FOR DISPLAY OUTPUT: Subtract calibration from current angles
        display_degs = [
            np.degrees(self.current_angles[0]) - CALIBRATION_DEGREES[0],
            np.degrees(self.current_angles[1]) - CALIBRATION_DEGREES[1],
            np.degrees(self.current_angles[2]) - CALIBRATION_DEGREES[2]
        ]
        
        sys.stdout.write(f"\r📡 MOVEIT ANGLE -> J1:{display_degs[0]:6.1f}° | J2:{display_degs[1]:6.1f}° | J3:{display_degs[2]:6.1f}°  ")
        sys.stdout.flush()

    def send_to_motors(self, rads):
        self.group_sync_write.clearParam()
        for i in range(3):
            val = int(OFFSETS[i] + (rads[i] * DIRECTIONS[i] * (2048.0 / np.pi)))
            val = max(0, min(4095, val))
            param = [DXL_LOBYTE(DXL_LOWORD(val)), DXL_HIBYTE(DXL_LOWORD(val)),
                     DXL_LOBYTE(DXL_HIWORD(val)), DXL_HIBYTE(DXL_HIWORD(val))]
            self.group_sync_write.addParam(DXL_IDS[i], param)
        self.group_sync_write.txPacket()

    def stop(self):
        if self.connected: 
            self.port_handler.closePort()


def main():
    try:
        node = MoveItDirectSync()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'node' in locals():
            node.stop()


if __name__ == '__main__':
    main()
