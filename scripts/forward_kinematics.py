"""
3DoF RRR Robot Arm Forward Kinematics
URDF'den alınan link uzunlukları:
L1 = 0.12 m (link1)
L2 = 0.15 m (link2)
L3 = 0.10 m (link3)
"""
import numpy as np

def forward_kinematics(joint1, joint2, joint3, L1=0.12, L2=0.15, L3=0.10):
    """
    joint1, joint2, joint3: radians
    Returns: (x, y, z) in meters
    """
    # Taban dönüşü
    theta1 = joint1
    theta2 = joint2
    theta3 = joint3
    # Planar RRR kinematik
    x = (L2 * np.cos(theta2) + L3 * np.cos(theta2 + theta3)) * np.cos(theta1)
    y = (L2 * np.cos(theta2) + L3 * np.cos(theta2 + theta3)) * np.sin(theta1)
    z = L1 + L2 * np.sin(theta2) + L3 * np.sin(theta2 + theta3)
    return x, y, z

if __name__ == "__main__":
    # Test
    j1, j2, j3 = 0.0, 0.0, 0.0
    print("FK:", forward_kinematics(j1, j2, j3))
