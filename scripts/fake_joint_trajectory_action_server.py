#!/usr/bin/env python3
"""
Fake Joint Trajectory Action Server for MoveIt (ROS1)
======================================================
This provides a fake FollowJointTrajectory action server that allows
MoveIt to execute planned trajectories. It publishes joint states
based on the trajectory commands.
"""

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryResult, FollowJointTrajectoryFeedback
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
import time


class FakeJointTrajectoryActionServer:
    def __init__(self):
        rospy.init_node('fake_joint_trajectory_action_server')
        
        # Current joint positions
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.current_positions = [0.0, 0.0, 0.0]
        
        # Joint state publisher
        self.joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        
        # Action server for arm controller
        self._action_server = actionlib.SimpleActionServer(
            '/arm_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction,
            execute_cb=self.execute_callback,
            auto_start=False
        )
        self._action_server.start()
        
        # Timer to publish joint states
        self.timer = rospy.Timer(rospy.Duration(0.02), self.publish_joint_states)
        
        rospy.loginfo('Fake Joint Trajectory Action Server started')
        rospy.loginfo('Listening on: /arm_controller/follow_joint_trajectory')

    def publish_joint_states(self, event):
        """Publish current joint states."""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.joint_names
        msg.position = self.current_positions
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)
        self.joint_state_pub.publish(msg)

    def execute_callback(self, goal):
        """Execute the trajectory."""
        rospy.loginfo('Executing trajectory...')
        
        trajectory = goal.trajectory
        
        if not trajectory.points:
            rospy.logwarn('Empty trajectory received')
            result = FollowJointTrajectoryResult()
            result.error_code = FollowJointTrajectoryResult.SUCCESSFUL
            self._action_server.set_succeeded(result)
            return
        
        # Map joint names to indices
        joint_indices = {}
        for i, name in enumerate(trajectory.joint_names):
            if name in self.joint_names:
                joint_indices[name] = self.joint_names.index(name)
        
        # Execute trajectory points
        start_time = time.time()
        
        for i, point in enumerate(trajectory.points):
            # Check for preempt
            if self._action_server.is_preempt_requested():
                rospy.loginfo('Trajectory execution preempted')
                self._action_server.set_preempted()
                return
            
            # Calculate time to wait
            target_time = point.time_from_start.to_sec()
            current_time = time.time() - start_time
            
            if target_time > current_time:
                rospy.sleep(target_time - current_time)
            
            # Update positions
            for j, name in enumerate(trajectory.joint_names):
                if name in joint_indices:
                    idx = joint_indices[name]
                    self.current_positions[idx] = point.positions[j]
            
            # Publish feedback
            feedback = FollowJointTrajectoryFeedback()
            feedback.joint_names = self.joint_names
            feedback.actual.positions = self.current_positions
            feedback.desired.positions = list(point.positions)
            self._action_server.publish_feedback(feedback)
        
        rospy.loginfo('Trajectory execution completed')
        
        result = FollowJointTrajectoryResult()
        result.error_code = FollowJointTrajectoryResult.SUCCESSFUL
        self._action_server.set_succeeded(result)


def main():
    try:
        server = FakeJointTrajectoryActionServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
