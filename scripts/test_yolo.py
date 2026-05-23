from ultralytics import YOLO
import cv2
import numpy as np

print('1. Loading model...')
m = YOLO('yolov8m.pt')

print('2. Making test frame...')
img = np.zeros((480, 854, 3), dtype='uint8')

print('3. Running predict...')
r = m.predict(img, verbose=False, device='cpu')
print('predict OK:', len(r[0].boxes), 'detections')

print('4. Running track...')
r2 = m.track(img, verbose=False, device='cpu', persist=True, tracker='configs/bytetrack.yaml')
print('track OK')
