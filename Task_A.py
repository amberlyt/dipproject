import cv2
import numpy as np
import os

# === File Paths ===
video_path = 'videos/street.mp4'
face_detector_path = 'face_detector.xml'
talking_path = 'videos/talking.mp4'
endscreen_path = 'videos/endscreen.mp4'
watermark1_path = 'watermarks/watermark1.png'
watermark2_path = 'watermarks/watermark2.png'
final_output_path = 'outputs/final_with_endscreen.avi'

# === Load Face Detector ===
face_cascade = cv2.CascadeClassifier(face_detector_path)

# === Load Video Properties ===
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(final_output_path, fourcc, fps, (frame_width, frame_height))

# === STEP 1: Brightness Detection ===
brightness_values = []
for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness_values.append(np.mean(gray))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
brightness_increase_needed = np.mean(brightness_values) < 100  # Night scene check

def brighten(frame, value=40):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + value, 0, 255)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

# === Watermark Function ===
def insert_watermark(frame, count, wm1, wm2):
    wm = wm1 if (count % 200) < 100 else wm2
    if wm.shape[2] == 4:
        overlay = frame.copy().astype(np.float32)
        wm_rgb = wm[:, :, :3].astype(np.float32)
        wm_alpha = (wm[:, :, 3].astype(np.float32) / 255.0) * 1.5
        for c in range(3):
            overlay[:, :, c] = wm_alpha * wm_rgb[:, :, c] + (1 - wm_alpha) * overlay[:, :, c]
        return np.clip(overlay, 0, 255).astype(np.uint8)
    else:
        return cv2.addWeighted(frame, 0.85, wm, 1, 0)

# === Overlay Function ===
def apply_overlay(frame, overlay_frame, position=(20, 20)):
    x, y = position
    h, w = overlay_frame.shape[:2]
    if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
        frame[y:y+h, x:x+w] = overlay_frame
    return frame

# === STEP 2: Load Overlay Video Frames ===
def load_overlay_frames(path, target_size):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, target_size)
        frames.append(resized)
    cap.release()
    return frames

# === Load Watermarks ===
wm1 = cv2.imread(watermark1_path, cv2.IMREAD_UNCHANGED)
wm2 = cv2.imread(watermark2_path, cv2.IMREAD_UNCHANGED)

# === Prepare Overlay Frames ===
target_overlay_width = int(frame_width * 0.24)
talking_cap = cv2.VideoCapture(talking_path)
ret, sample = talking_cap.read()
talking_cap.release()
target_overlay_height = int(target_overlay_width * (sample.shape[0] / sample.shape[1])) if ret else 0
overlay_frames = load_overlay_frames(talking_path, (target_overlay_width, target_overlay_height))

# === STEP 3: Process Main Video ===
frame_count = 0
overlay_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Brighten if necessary
    if brightness_increase_needed:
        frame = brighten(frame)

    # Face Detection & Blurring
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(face, (99, 99), 30)

    # Overlay Talking Video
    if overlay_index < len(overlay_frames):
        frame = apply_overlay(frame, overlay_frames[overlay_index])
        overlay_index += 1

    # Add Watermark
    frame = insert_watermark(frame, frame_count, wm1, wm2)

    # Write Frame
    out.write(frame)
    frame_count += 1

cap.release()

# === STEP 4: Append Endscreen Video ===
cap_end = cv2.VideoCapture(endscreen_path)
while True:
    ret, frame = cap_end.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)
cap_end.release()

# === STEP 5: Finalize Output ===
out.release()
cv2.destroyAllWindows()
print("Final video created at:", final_output_path)