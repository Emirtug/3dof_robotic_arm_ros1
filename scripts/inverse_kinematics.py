"""
3DoF RRR Robot Arm Inverse Kinematics
URDF'den alınan link uzunlukları:
L1 = 0.12 m (link1)
L2 = 0.15 m (link2)
L3 = 0.10 m (link3)
"""
import numpy as np

def inverse_kinematics(x, y, z, L1=0.12, L2=0.15, L3=0.10):
    """
    x, y, z: hedef pozisyon (metre)
    Returns: (joint1, joint2, joint3) in radians
    """
    # Taban açısı
    joint1 = np.arctan2(y, x)
    # Planar çözüm
    r = np.sqrt(x**2 + y**2)
    z_offset = z - L1
    D = (r**2 + z_offset**2 - L2**2 - L3**2) / (2 * L2 * L3)
    if abs(D) > 1:
        raise ValueError("Hedef pozisyon erişilemez!")
    joint3 = np.arccos(D)
    phi = np.arctan2(z_offset, r)
    psi = np.arctan2(L3 * np.sin(joint3), L2 + L3 * np.cos(joint3))
    joint2 = phi - psi
    return joint1, joint2, joint3

if __name__ == "__main__":
    # Test
    x, y, z = 0.2, 0.0, 0.15
    print("IK:", inverse_kinematics(x, y, z))
