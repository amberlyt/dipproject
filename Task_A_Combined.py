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
video_path ='/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/street.mp4'
face_detector_path = '/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/face_detector.xml'
output_path ='/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/videos/blurred_output.mp4'
talking_path ='/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/talking.mp4'
endscreen_path ='/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/endscreen.mp4'
watermark1_path ='/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/watermark1.png'
watermark2_path ='/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/watermark2.png'
processed_path = '/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/outputs/final_processed_video.avi'
final_output_path = '/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/outputs/final_with_endscreen.avi'

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
    
# Save blurred video
out.release()
cap.release()
    
# === Step 3: Add overlay video ===

# === Process The Video Frame By Frame ===
def resize_overlay_video(talking_path, video_path_size, output_path):
    cap = cv2.VideoCapture(talking_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open overlay video: {talking_path}")
        
# Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Overlay video is empty")
        
# Resize/scale video
    h, w = frame.shape[:2]
    aspect_ratio = h / w
    target_width = int(video_path_size[0] * 0.24)
    target_height = int(target_width * aspect_ratio)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Write the updated frames
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (target_width, target_height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (target_width, target_height))
        out.write(resized)

    cap.release()
    out.release()
    return output_path

# Load all frames of the overlay video
def load_overlay_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot read overlay video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Apply the overlay video over the main video
def apply_overlay(frame, overlay_frame, position=(20, 20)):
    x, y = position
    h, w = overlay_frame.shape[:2]
    if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
        frame[y:y+h, x:x+w] = overlay_frame
    return frame



# === Step 4: Add watermarks ===

def insert_watermark(frame, count, watermark1, watermark2):
    frame = cv2.convertScaleAbs(frame, alpha=1.08, beta=8)
    wm = watermark1 if (count % 200) < 100 else watermark2
    if wm.shape[2] == 4:
        overlay = frame.copy().astype(np.float32)
        wm_rgb = wm[:, :, :3].astype(np.float32)
        wm_alpha = np.clip((wm[:, :, 3].astype(np.float32) / 255.0) * 1.8, 0, 1)
        for c in range(3):
            overlay[:, :, c] = wm_alpha * wm_rgb[:, :, c] + (1.0 - wm_alpha) * overlay[:, :, c]
        return np.clip(overlay, 0, 255).astype(np.uint8)
    else:
        return cv2.addWeighted(frame, 0.85, wm, 1, 0)

# === Step 5: Add endscreen ===

def attach_endscreen(writer, end_path, resolution):
    cap = cv2.VideoCapture(end_path)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resolution)
        writer.write(frame)
    cap.release()

# === Step 6: Final Processing Pipeline ===

# Load processed video and overlay frames
cap = cv2.VideoCapture(output_path)
overlay_frames = load_overlay_frames(talking_path)
wm1 = cv2.imread(watermark1_path, cv2.IMREAD_UNCHANGED)
wm2 = cv2.imread(watermark2_path, cv2.IMREAD_UNCHANGED)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(os.path.dirname(processed_path), exist_ok=True)
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

# Create final output writer
final_writer = cv2.VideoWriter(processed_path, fourcc, 30.0, (frame_width, frame_height))

frame_count = 0
overlay_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Brighten if needed
    if brightness_increase_needed:
        frame = brighten(frame)

    # Overlay talking head
    if overlay_index < len(overlay_frames):
        frame = apply_overlay(frame, overlay_frames[overlay_index])
        overlay_index += 1

    # Insert watermark
    frame = insert_watermark(frame, frame_count, wm1, wm2)

    # Write to final video
    final_writer.write(frame)
    frame_count += 1

cap.release()

# Release all
final_writer.release()
cv2.destroyAllWindows()
