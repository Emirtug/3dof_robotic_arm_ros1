#!/usr/bin/env python3
"""
Fake Joint Trajectory Action Server
Provides a fake action server for MoveIt's Plan & Execute functionality
Publishes joint states to update robot visualization
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
import threading
import time


class FakeJointTrajectoryActionServer(Node):
    def __init__(self):
        super().__init__('fake_joint_trajectory_action_server')
        
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory',
            self.execute_callback
        )
        
        # Publisher for joint states
        self._joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # Current joint positions
        self._current_positions = {'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0}
        
        # Lock for thread safety
        self._position_lock = threading.Lock()
        
        # Timer to continuously publish joint states (100 Hz for smooth visualization)
        self._publish_timer = self.create_timer(0.01, self.publish_joint_states)
        
        self.get_logger().info('Fake Joint Trajectory Action Server started on arm_controller/follow_joint_trajectory')
    
    def publish_joint_states(self):
        """Continuously publish current joint states"""
        with self._position_lock:
            joint_state = JointState()
            joint_state.header.stamp = self.get_clock().now().to_msg()
            # Ensure joint names are in correct order: joint1, joint2, joint3
            joint_state.name = ['joint1', 'joint2', 'joint3']
            joint_state.position = [
                self._current_positions.get('joint1', 0.0),
                self._current_positions.get('joint2', 0.0),
                self._current_positions.get('joint3', 0.0)
            ]
            joint_state.velocity = [0.0] * len(joint_state.name)
            joint_state.effort = [0.0] * len(joint_state.name)
            
            self._joint_state_pub.publish(joint_state)

    def execute_callback(self, goal_handle):
        goal = goal_handle.request
        self.get_logger().info(f'Received trajectory goal with {len(goal.trajectory.points)} points')
        
        # Accept the goal
        goal_handle.succeed()
        
        # Execute trajectory in a separate thread to avoid blocking
        def execute_trajectory():
            if goal.trajectory.points:
                # Get joint names from trajectory
                joint_names = goal.trajectory.joint_names
                
                # Execute trajectory points
                start_time = time.time()
                for i, point in enumerate(goal.trajectory.points):
                    # Calculate time from start
                    if i == 0:
                        point_time = 0.0
                    else:
                        point_time = (point.time_from_start.sec + point.time_from_start.nanosec * 1e-9)
                    
                    # Wait until the right time
                    elapsed = time.time() - start_time
                    if point_time > elapsed:
                        time.sleep(point_time - elapsed)
                    
                    # Update current positions (thread-safe)
                    with self._position_lock:
                        for j, joint_name in enumerate(joint_names):
                            if j < len(point.positions):
                                self._current_positions[joint_name] = point.positions[j]
                    
                    # Small delay for smooth visualization
                    if i < len(goal.trajectory.points) - 1:
                        time.sleep(0.01)
                
                with self._position_lock:
                    self.get_logger().info(f'Trajectory executed, final positions: {self._current_positions}')
        
        # Execute in background thread
        thread = threading.Thread(target=execute_trajectory)
        thread.daemon = True
        thread.start()
        
        # Create result
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        
        return result


def main(args=None):
    rclpy.init(args=args)
    node = FakeJointTrajectoryActionServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
