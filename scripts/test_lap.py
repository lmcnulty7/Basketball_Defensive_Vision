"""Test whether lap (linear assignment) crashes on a real frame with detections."""
import sys
sys.path.insert(0, '.')
import cv2
import numpy as np

print('Reading real frame...')
cap = cv2.VideoCapture('data/raw/curry_classic_clip.mp4')
ret, frame = cap.read()
cap.release()

from ultralytics import YOLO
m = YOLO('yolov8m.pt')

# First: detect without tracking (no lap involved)
print('predict() on real frame (no lap)...')
r = m.predict(frame, classes=[0], conf=0.4, verbose=False, device='cpu')
print(f'  detections: {len(r[0].boxes)}')

# Second: track with botsort instead of bytetrack (different matching)
print('track() with botsort (no lap dependency)...')
r2 = m.track(frame, classes=[0], persist=True,
             tracker='botsort.yaml', conf=0.4, verbose=False, device='cpu')
print(f'  tracks: {len(r2[0].boxes)}')

print('ALL DONE')
