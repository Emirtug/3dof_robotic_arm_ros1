"""
3DoF RRR Robot Arm DH Parametreleri ve DH tablosu
Klasik DH parametreleri ile forward kinematics
"""
import numpy as np

# DH parametreleri (örnek değerler)
# [a, alpha, d, theta]
DH_TABLE = [
    [0,      0,   0.12,  None],  # Joint1: taban
    [0.15,   0,     0,   None],  # Joint2: kol
    [0.10,   0,     0,   None],  # Joint3: uç
]

def dh_transform(a, alpha, d, theta):
    """DH parametrelerinden dönüşüm matrisi"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,      0,      0,    1]
    ])

def forward_kinematics_dh(joint_angles):
    """
    joint_angles: [theta1, theta2, theta3] (radyan)
    Returns: end-effector pozisyonu (x, y, z)
    """
    T = np.eye(4)
    for i, (a, alpha, d, _) in enumerate(DH_TABLE):
        theta = joint_angles[i]
        T = T @ dh_transform(a, alpha, d, theta)
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    return x, y, z

if __name__ == "__main__":
    # Test
    j1, j2, j3 = 0.0, 0.0, 0.0
    print("DH FK:", forward_kinematics_dh([j1, j2, j3]))
