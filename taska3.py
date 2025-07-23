import cv2
import numpy as np
import os
from typing import Tuple

class YouTubeVideoProcessor:
    def __init__(self):
        self.watermark1 = None
        self.watermark2 = None

    def get_video_dimensions(self, path: str) -> Tuple[int, int, float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, fps

    def resize_overlay_video(self, overlay_path: str, main_video_size: Tuple[int, int], output_path: str) -> str:
        cap = cv2.VideoCapture(overlay_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open overlay video: {overlay_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Overlay video is empty: {overlay_path}")

        h, w = frame.shape[:2]
        aspect_ratio = h / w
        target_width = int(main_video_size[0] * 0.24)
        target_height = int(target_width * aspect_ratio)
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

    def load_talking_frames(self, path: str) -> list:
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

    def apply_overlay(self, frame, overlay_frame, position=(20, 20)):
        x, y = position
        h, w = overlay_frame.shape[:2]
        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            frame[y:y+h, x:x+w] = overlay_frame
        return frame

    def insert_watermark(self, frame, count):
        frame = cv2.convertScaleAbs(frame, alpha=1.08, beta=8)
        wm = self.watermark1 if (count % 200) < 100 else self.watermark2
        if wm.shape[2] == 4:
            overlay = frame.copy().astype(np.float32)
            wm_rgb = wm[:, :, :3].astype(np.float32)
            wm_alpha = np.clip((wm[:, :, 3].astype(np.float32) / 255.0) * 1.8, 0, 1)  # Increased clarity
            for c in range(3):
                overlay[:, :, c] = wm_alpha * wm_rgb[:, :, c] + (1.0 - wm_alpha) * overlay[:, :, c]
            return np.clip(overlay, 0, 255).astype(np.uint8)
        else:
            return cv2.addWeighted(frame, 0.85, wm, 1, 0)  # Increased opacity

    def attach_end_video(self, writer, end_path, resolution):
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

    def process_single_video(self, video_path, overlay_path, wm1_path, wm2_path, end_path, output_path):
        self.watermark1 = cv2.imread(wm1_path, cv2.IMREAD_UNCHANGED)
        self.watermark2 = cv2.imread(wm2_path, cv2.IMREAD_UNCHANGED)
        width, height, fps = self.get_video_dimensions(video_path)

        resized_overlay_path = "resized_overlay_temp.mp4"
        self.resize_overlay_video(overlay_path, (width, height), resized_overlay_path)
        talking_frames = self.load_talking_frames(resized_overlay_path)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num < len(talking_frames):
                frame = self.apply_overlay(frame, talking_frames[frame_num])
            frame = self.insert_watermark(frame, frame_num)
            writer.write(frame)
            frame_num += 1

        cap.release()
        self.attach_end_video(writer, end_path, (width, height))
        writer.release()

        if os.path.exists(resized_overlay_path):
            os.remove(resized_overlay_path)

    def process_all_videos(self, main_videos, overlay_path, wm1, wm2, endscreen, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for video_path in main_videos:
            fname = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, fname + "_processed.mp4")
            print(f"Processing: {video_path}")
            try:
                self.process_single_video(video_path, overlay_path, wm1, wm2, endscreen, output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Failed {video_path}: {e}")


if __name__ == "__main__":
    processor = YouTubeVideoProcessor()

    main_videos = [
        "/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/Recorded Videos (4)/alley.mp4",
        "/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/Recorded Videos (4)/office.mp4", 
        "/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/Recorded Videos (4)/singapore.mp4",
        "/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/Recorded Videos (4)/traffic.mp4"
    ]
    
    talking_video = "/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/talking.mp4"
    watermark1 = "/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/watermark1.png"
    watermark2 = "/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/watermark2.png"
    endscreen = "/Users/iantan/Documents/CSC2014-DIP Lab Exercises/CSC2014- Group Assignment_Aug-2025/endscreen.mp4"

    processor.process_all_videos(main_videos, talking_video, watermark1, watermark2, endscreen, output_dir="processed_videos")
