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