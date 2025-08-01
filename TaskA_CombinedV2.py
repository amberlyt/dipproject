#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:13:57 2025

@author: iantan
"""

import cv2
import numpy as np
import os
import time
from typing import Tuple

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
    
# Save blurred video
out.release()
cap.release()
time.sleep(1)  # Wait for file to finish writing

# Ensure output was written
if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
    raise FileNotFoundError(f"Blurred video not found or is empty: {output_path}")

# === Utilities ===
def get_video_dimensions(path):
    cap = cv2.VideoCapture(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return width, height, fps

# === Step 3: Add overlay video ===

def resize_overlay_video(overlay_path: str, main_video_size: Tuple[int, int], output_path: str) -> str:
    cap = cv2.VideoCapture(overlay_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open overlay video: {overlay_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret or frame is None:
        raise ValueError(f"Overlay video is empty or unreadable: {overlay_path}")

    h, w = frame.shape[:2]
    aspect_ratio = h / w
    target_width = int(main_video_size[0] * 0.24)
    target_height = int(target_width * aspect_ratio)

    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid target size: {target_width}x{target_height}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (target_width, target_height))
        out.write(resized)

    cap.release()
    out.release()
    return output_path

def load_overlay_frames(path: str) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Unable to read overlay video: {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

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

def attach_end_screen(writer, end_path, resolution):
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

def process_single_video(video_path, overlay_path, wm1_path, wm2_path, end_path, output_path):
    watermark1 = cv2.imread(wm1_path, cv2.IMREAD_UNCHANGED)
    watermark2 = cv2.imread(wm2_path, cv2.IMREAD_UNCHANGED)
    width, height, fps = get_video_dimensions(video_path)

    resized_overlay_path = "resized_overlay_temp.mp4"
    resize_overlay_video(overlay_path, (width, height), resized_overlay_path)
    talking_frames = load_overlay_frames(resized_overlay_path)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if brightness_increase_needed:
            frame = brighten(frame)
        if frame_num < len(talking_frames):
            frame = apply_overlay(frame, talking_frames[frame_num])
        frame = insert_watermark(frame, frame_num, watermark1, watermark2)
        writer.write(frame)
        frame_num += 1

    cap.release()
    attach_end_screen(writer, end_path, (width, height))
    writer.release()

    if os.path.exists(resized_overlay_path):
        os.remove(resized_overlay_path)

# === Run Final Processing ===

os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(os.path.dirname(processed_path), exist_ok=True)
os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

process_single_video(
    video_path=output_path,
    overlay_path=talking_path,
    wm1_path=watermark1_path,
    wm2_path=watermark2_path,
    end_path=endscreen_path,
    output_path=final_output_path
)

cv2.destroyAllWindows()
