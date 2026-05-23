"""
Minimal reproduction of the pipeline's first frame to isolate the segfault.
Loads all modules in the same order as PipelineRunner, then processes one
real video frame.
"""
import sys
sys.path.insert(0, '.')

print('Importing all modules...')
import cv2, numpy as np

# Import everything the pipeline imports
from src.ingestion.video_reader import VideoReader
from src.detection.ball_detector import BallDetector
from src.tracking.multi_tracker import MultiTracker
from src.tracking.ball_tracker import BallTracker
from src.tracking.track_manager import TrackManager
from src.classification.team_classifier import TeamClassifier
from src.court.keypoint_detector import NeuralKeyDetector
from src.court.homography import CourtHomography
from src.court.court_model import CourtModel
from src.events.steal_detector import StealDetector
from src.events.contest_detector import ContestDetector
from src.events.rebound_detector import ReboundDetector
from src.events.block_detector import BlockDetector
from src.events.deflection_detector import DeflectionDetector
from src.events.event_detector import EventOrchestrator
from src.stats.matchup_tracker import MatchupTracker
from src.stats.speed_calculator import SpeedCalculator
from src.stats.aggregator import StatsAggregator
print('All imports OK')

print('Reading one real video frame...')
cap = cv2.VideoCapture('data/raw/curry_classic_clip.mp4')
ret, frame = cap.read()
cap.release()
print(f'Frame shape: {frame.shape}')

print('Creating MultiTracker...')
tracker = MultiTracker('models/checkpoints/yolov8m.pt', device='cpu',
                       tracker_config='configs/bytetrack.yaml')

print('Creating BallDetector...')
ball_det = BallDetector('models/checkpoints/ball_yolo.pt',
                        fallback_weights='models/checkpoints/yolov8m.pt',
                        device='cpu')

print('Running tracker.update() on real frame...')
tracks = tracker.update(frame, frame_idx=0, timestamp_sec=0.0)
print(f'Tracks: {len(tracks)}  IDs: {[t.track_id for t in tracks]}')

print('Running ball_det.detect() on real frame...')
ball, is_pred = ball_det.detect(frame)
print(f'Ball: {ball}  predicted: {is_pred}')

print('ALL DONE — no segfault')
