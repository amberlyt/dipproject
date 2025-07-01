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
talking_path = 'videos/talking.mp4'
endscreen_path = 'videos/endscreen.mp4'
watermark1_path = 'watermarks/watermark1.png'
watermark2_path = 'watermarks/watermark2.png'
processed_path = 'outputs/final_processed_video.avi'
final_output_path = 'outputs/final_with_endscreen.avi'

# === Setup Video Properties ===
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

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
