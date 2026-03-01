#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUco Test - ROS olmadan direkt kamera testi
Everest SC-HD03 kamera ile ArUco marker tespiti
"""

import cv2
import cv2.aruco as aruco
import numpy as np

def main():
    # Kamera ID (video2 = 2)
    CAMERA_ID = 2
    MARKER_SIZE = 0.015  # 15mm
    TARGET_ID = 102
    
    # ArUco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = aruco.DetectorParameters_create()
    
    # Default camera matrix (640x480)
    camera_matrix = np.array([
        [640, 0, 320],
        [0, 640, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    # Kamerayı aç
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"HATA: Kamera acilamadi! /dev/video{CAMERA_ID}")
        print("Diger kameralari dene: 0, 1, 2...")
        return
    
    print("=" * 50)
    print("ArUco Marker Test")
    print("=" * 50)
    print(f"Kamera: /dev/video{CAMERA_ID}")
    print(f"Dictionary: DICT_ARUCO_ORIGINAL")
    print(f"Hedef ID: {TARGET_ID}")
    print(f"Marker boyutu: {MARKER_SIZE*1000:.0f}mm")
    print("=" * 50)
    print("'q' = Cikis")
    print("'s' = Screenshot kaydet")
    print("=" * 50)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame alinamadi!")
            break
        
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Marker tespiti
        corners, ids, rejected = aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )
        
        # Sonuçları çiz
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i, marker_id in enumerate(ids.flatten()):
                # Pose tahmin et
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    [corners[i]], MARKER_SIZE, camera_matrix, dist_coeffs
                )
                
                # Axis çiz
                aruco.drawAxis(frame, camera_matrix, dist_coeffs,
                               rvecs[0], tvecs[0], MARKER_SIZE)
                
                tvec = tvecs[0][0]
                distance = np.linalg.norm(tvec) * 100  # cm
                
                # Bilgileri göster
                color = (0, 255, 0) if marker_id == TARGET_ID else (255, 165, 0)
                y_offset = 30 + i * 90
                
                cv2.putText(frame, f"ID: {marker_id}", 
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Mesafe: {distance:.1f} cm",
                            (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"X:{tvec[0]*100:.1f} Y:{tvec[1]*100:.1f} Z:{tvec[2]*100:.1f}",
                            (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if marker_id == TARGET_ID:
                    cv2.putText(frame, ">> HEDEF MARKER! <<", 
                                (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print(f"\r[Frame {frame_count}] ID:{marker_id} | Mesafe:{distance:.1f}cm | "
                          f"X:{tvec[0]*100:.1f} Y:{tvec[1]*100:.1f} Z:{tvec[2]*100:.1f}", end="")
        else:
            cv2.putText(frame, "Marker bulunamadi...", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Rejected: {len(rejected) if rejected else 0}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # FPS göster
        cv2.putText(frame, f"Frame: {frame_count}", 
                    (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Pencereyi göster
        cv2.imshow('ArUco Test - Press Q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"aruco_capture_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"\nKaydedildi: {filename}")
    
    print("\n\nKamera kapatiliyor...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
