#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3DoF RRR Robot Arm - Gazebo Basic Move Script
Gazebo simülasyonunda robotu hareket ettiren basit script
"""

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math

class BasicMove:
    def __init__(self):
        rospy.init_node('basic_move', anonymous=True)
        
        # Action client for arm_controller
        self.client = actionlib.SimpleActionClient(
            '/arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        
        rospy.loginfo("Arm controller'a baglaniliyor...")
        self.client.wait_for_server(timeout=rospy.Duration(10.0))
        rospy.loginfo("Arm controller'a baglandi!")
        
        # Joint isimleri
        self.joint_names = ['joint1', 'joint2', 'joint3']
        
        rospy.loginfo("Basic Move Node baslatildi!")
        rospy.loginfo("Komutlar:")
        rospy.loginfo("  1: Home pozisyonu (0, 0, 0)")
        rospy.loginfo("  2: Pozisyon 1")
        rospy.loginfo("  3: Pozisyon 2")
        rospy.loginfo("  4: Pozisyon 3")
        rospy.loginfo("  5: Wave (el sallama)")
        rospy.loginfo("  6: Sinusoidal hareket")
        rospy.loginfo("  q: Cikis")
    
    def move_joints(self, positions, duration=2.0):
        """Jointleri belirtilen pozisyona hareket ettir"""
        rospy.loginfo("Hareket: joint1={:.2f}, joint2={:.2f}, joint3={:.2f}".format(
            positions[0], positions[1], positions[2]))
        
        # Trajectory goal oluştur
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        
        # Trajectory point ekle
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0, 0.0, 0.0]
        point.time_from_start = rospy.Duration(duration)
        
        goal.trajectory.points.append(point)
        
        # Goal'u gönder ve bekle
        self.client.send_goal(goal)
        self.client.wait_for_result(timeout=rospy.Duration(duration + 2.0))
        
        return self.client.get_result()
    
    def home_position(self):
        """Home pozisyonuna git"""
        rospy.loginfo("Home pozisyonuna gidiliyor...")
        self.move_joints([0.0, 0.0, 0.0])
    
    def position_1(self):
        """Pozisyon 1"""
        rospy.loginfo("Pozisyon 1'e gidiliyor...")
        self.move_joints([0.5, 0.3, 0.2])
    
    def position_2(self):
        """Pozisyon 2"""
        rospy.loginfo("Pozisyon 2'ye gidiliyor...")
        self.move_joints([-0.5, 0.5, -0.3])
    
    def position_3(self):
        """Pozisyon 3"""
        rospy.loginfo("Pozisyon 3'e gidiliyor...")
        self.move_joints([1.0, -0.5, 0.5])
    
    def wave(self):
        """El sallama hareketi"""
        rospy.loginfo("El sallama hareketi...")
        for i in range(3):
            self.move_joints([0.0, 0.3, 0.5], 0.5)
            self.move_joints([0.0, 0.3, -0.5], 0.5)
        self.home_position()
    
    def sinusoidal_motion(self):
        """Sinusoidal hareket - trajectory olarak"""
        rospy.loginfo("Sinusoidal hareket basliyor (5 saniye)...")
        
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        
        # 5 saniyelik trajectory oluştur
        num_points = 50
        for i in range(num_points):
            t = i * 0.1  # 0.1 saniye aralıklarla
            
            point = JointTrajectoryPoint()
            point.positions = [
                0.5 * math.sin(2 * math.pi * 0.5 * t),
                0.3 * math.sin(2 * math.pi * 0.3 * t),
                0.4 * math.sin(2 * math.pi * 0.7 * t)
            ]
            point.velocities = [0.0, 0.0, 0.0]
            point.time_from_start = rospy.Duration(t + 0.1)
            goal.trajectory.points.append(point)
        
        self.client.send_goal(goal)
        self.client.wait_for_result(timeout=rospy.Duration(7.0))
        
        rospy.loginfo("Sinusoidal hareket bitti")
        self.home_position()
    
    def run(self):
        """Ana döngü"""
        while not rospy.is_shutdown():
            cmd = input("\nKomut girin (1-6, q): ")
            
            if cmd == '1':
                self.home_position()
            elif cmd == '2':
                self.position_1()
            elif cmd == '3':
                self.position_2()
            elif cmd == '4':
                self.position_3()
            elif cmd == '5':
                self.wave()
            elif cmd == '6':
                self.sinusoidal_motion()
            elif cmd == 'q' or cmd == 'Q':
                rospy.loginfo("Cikis yapiliyor...")
                break
            else:
                rospy.logwarn("Gecersiz komut!")


def main():
    try:
        mover = BasicMove()
        mover.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
