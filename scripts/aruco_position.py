#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
ArUco Position Tracker
Kamera lensi = 0,0,0 referans noktası
Pozisyon verisini ekran ve terminalde gösterir
"""

import cv2
import cv2.aruco as aruco
import numpy as np
from datetime import datetime

# ============== AYARLAR ==============
CAMERA_ID = 0                    # USB 2.0 Camera (harici)
MARKER_SIZE = 0.015              # 15mm (metre cinsinden)
TARGET_ID = 102                  # Hedef marker ID

# Kamera parametreleri (tahmini - kalibrasyon sonra yapılacak)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FOCAL_LENGTH = 640               # Tahmini focal length (piksel)

# =====================================

def create_camera_matrix():
    """Kamera intrinsic matrisini oluştur"""
    cx = FRAME_WIDTH / 2         # Principal point X
    cy = FRAME_HEIGHT / 2        # Principal point Y
    
    camera_matrix = np.array([
        [FOCAL_LENGTH, 0, cx],
        [0, FOCAL_LENGTH, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def main():
    # ArUco setup (OpenCV 4.x API)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Kamera matrisi
    camera_matrix, dist_coeffs = create_camera_matrix()
    
    # Kamerayı aç
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"HATA: Kamera acilamadi! /dev/video{CAMERA_ID}")
        return
    
    print("=" * 60)
    print("       ArUco Position Tracker")
    print("=" * 60)
    print(f"  Referans: Kamera lensi = (0, 0, 0)")
    print(f"  Kamera: /dev/video{CAMERA_ID}")
    print(f"  Hedef Marker ID: {TARGET_ID}")
    print(f"  Marker Boyutu: {MARKER_SIZE*1000:.0f}mm")
    print("=" * 60)
    print("  Koordinat Sistemi:")
    print("    X+ = Saga")
    print("    Y+ = Asagi")
    print("    Z+ = Kameradan uzaklasma (derinlik)")
    print("=" * 60)
    print("  Kontroller: [Q] Cikis  [S] Screenshot  [R] Reset")
    print("=" * 60)
    print()
    
    # Son pozisyon (filtreleme için)
    last_position = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame alinamadi!")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Marker tespiti (OpenCV 4.x API)
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        
        # Siyah bilgi paneli (üstte)
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 100), (0, 0, 0), -1)
        
        # Koordinat referans göstergesi (sol üst köşe)
        cv2.putText(frame, "KAMERA (0,0,0)", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id == TARGET_ID:
                    # Pose tahmin et
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        [corners[i]], MARKER_SIZE, camera_matrix, dist_coeffs
                    )
                    
                    # Pozisyon verisi (metre -> cm)
                    tvec = tvecs[0][0]
                    x_cm = tvec[0] * 100
                    y_cm = tvec[1] * 100
                    z_cm = tvec[2] * 100
                    
                    # Mesafe hesapla
                    distance = np.sqrt(x_cm**2 + y_cm**2 + z_cm**2)
                    
                    last_position = (x_cm, y_cm, z_cm)
                    
                    # Terminal çıktısı
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{timestamp}] X: {x_cm:+7.2f} cm | Y: {y_cm:+7.2f} cm | Z: {z_cm:+7.2f} cm | Mesafe: {distance:.1f} cm")
                    
                    # Ekran çıktısı - Ana pozisyon
                    cv2.putText(frame, f"MARKER ID: {TARGET_ID} TESPIT EDILDI", 
                                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Pozisyon değerleri (büyük font)
                    cv2.putText(frame, f"X: {x_cm:+.1f} cm", (10, 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)
                    cv2.putText(frame, f"Y: {y_cm:+.1f} cm", (180, 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
                    cv2.putText(frame, f"Z: {z_cm:+.1f} cm", (350, 75), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
                    
                    # Mesafe (sağ üst)
                    cv2.putText(frame, f"Mesafe: {distance:.1f} cm", 
                                (FRAME_WIDTH - 180, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Axis çiz (OpenCV 4.x)
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                      rvecs[0], tvecs[0], MARKER_SIZE * 2)
                    
                    # Marker merkezi işaretle
                    center = corners[i][0].mean(axis=0).astype(int)
                    cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
                    
        else:
            # Marker bulunamadi
            cv2.putText(frame, "MARKER BULUNAMADI", (10, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if last_position:
                cv2.putText(frame, f"Son: X:{last_position[0]:+.1f} Y:{last_position[1]:+.1f} Z:{last_position[2]:+.1f}", 
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Crosshair (ekran merkezi)
        cv2.line(frame, (FRAME_WIDTH//2 - 20, FRAME_HEIGHT//2), 
                 (FRAME_WIDTH//2 + 20, FRAME_HEIGHT//2), (50, 50, 50), 1)
        cv2.line(frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2 - 20), 
                 (FRAME_WIDTH//2, FRAME_HEIGHT//2 + 20), (50, 50, 50), 1)
        
        # Pencere göster
        cv2.imshow('ArUco Position Tracker - [Q] Quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"position_{datetime.now().strftime('%H%M%S')}.png"
            cv2.imwrite(filename, frame)
            print(f">>> Screenshot kaydedildi: {filename}")
        elif key == ord('r'):
            last_position = None
            print(">>> Reset yapildi")
    
    print("\nKamera kapatiliyor...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
