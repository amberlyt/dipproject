#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:13:57 2025

@author: amberlyteoh
"""

import cv2
import numpy as np
import os

# === File Paths ===
video_path = 'videos/street.mp4'
face_detector_path = 'face_detector.xml'
output_path = 'videos/blurred_output.mp4'
talking_path = 'videos/talking.mp4'
endscreen_path = 'videos/endscreen.mp4'
watermark1_path = 'watermarks/watermark1.png'
watermark2_path = 'watermarks/watermark2.png'
processed_path = 'TaskA_Output/final_processed_video.avi'
final_output_path = 'TaskA_Output/final_with_endscreen.avi'

# === Setup Video Properties ===
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))


# === Step 1: Brightness Estimation ===
brightness_values = []
for i in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    brightness_values.append(avg_brightness)

overall_avg_brightness = np.mean(brightness_values)
brightness_increase_needed = overall_avg_brightness < 100

print(f"Average brightness of first 30 frames: {overall_avg_brightness:.2f}")
print("Nighttime detected:", brightness_increase_needed)

# === Brightness Function ===
def brighten(frame, value=40):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    bright_frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return bright_frame


# === Step 2: Blur Faces ===

# === Load Face Detector ===
face_cascade = cv2.CascadeClassifier(face_detector_path)

# === Process The Video Frame By Frame ===
while True:
    ret, frame = cap.read()
    if not ret:
        break  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect Faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
  
    # Blur each face
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(face, (99, 99), 30)

    out.write(frame)

# === Preview Video ===
    cv2.imshow("Blurring Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

# === Release ===
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()
