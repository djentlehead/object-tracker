import cv2
import numpy as np
import time
import json
import os
from collections import deque

class SimplifiedTracker:
    def __init__(self, config_file=None):
        self.load_config(config_file)
        
        self.tracker_types = ['CSRT', 'MOSSE', 'KCF', 'BOOSTING', 'MIL', 'TLD']
        self.current_tracker_type = 'CSRT'
        
        self.fps_counter = deque(maxlen=30)
        self.lost_count = 0
        self.total_frames = 0
        
        self.paused = False
        self.show_trail = True
        self.show_info = True
        self.trail_points = deque(maxlen=50)
        
        self.csrt_tracker = None
        self.mosse_tracker = None
        self.bbox = None
        self.tracking_history = deque(maxlen=100)
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            "display_width": 1280,
            "display_height": 720,
            "frame_skip": 3,
            "confidence_threshold": 0.7,
            "save_tracking_data": True
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.config = {**default_config, **config}
        else:
            self.config = default_config
    
    def create_tracker(self, tracker_type='CSRT'):
        """Create tracker based on type"""
        tracker_type = tracker_type.upper()
        
        if tracker_type == 'CSRT':
            return cv2.legacy.TrackerCSRT_create()
        elif tracker_type == 'MOSSE':
            return cv2.legacy.TrackerMOSSE_create()
        elif tracker_type == 'KCF':
            return cv2.legacy.TrackerKCF_create()
        elif tracker_type == 'BOOSTING':
            return cv2.legacy.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            return cv2.legacy.TrackerMIL_create()
        elif tracker_type == 'TLD':
            return cv2.legacy.TrackerTLD_create()
        else:
            return cv2.legacy.TrackerCSRT_create()
    
    def adaptive_threshold(self, frame, bbox):
        """Calculate adaptive confidence threshold based on image quality"""
        x, y, w, h = map(int, bbox)
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0.5
            
        roi = frame[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0.5
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()  
        
        
        if blur > 500:
            return 0.8 
        elif blur > 100:
            return 0.6 
        else:
            return 0.4  
    
    def track_object(self, frame, frame_count):
        """Track single object using hybrid approach"""
        frame_skip = self.config['frame_skip']
        
        if frame_count % frame_skip == 0:
            # Run CSRT for accuracy
            success, bbox = self.csrt_tracker.update(frame)
            if success:
                self.bbox = bbox
                # Reinitialize MOSSE with updated position
                self.mosse_tracker = cv2.legacy.TrackerMOSSE_create()
                self.mosse_tracker.init(frame, bbox)
        else:
            # Run MOSSE for speed between CSRT updates
            success, bbox = self.mosse_tracker.update(frame)
            if success:
                self.bbox = bbox
        
        if success:
            # Calculate confidence
            confidence = self.adaptive_threshold(frame, bbox)
            
            # Add to history
            center_x = int(bbox[0] + bbox[2] / 2)
            center_y = int(bbox[1] + bbox[3] / 2)
            self.tracking_history.append((center_x, center_y, bbox))
            
            return {
                'bbox': bbox,
                'success': True,
                'confidence': confidence
            }
        else:
            self.lost_count += 1
            return {'success': False}
    
    def draw_enhanced_ui(self, frame, tracking_result, fps):
        """Draw enhanced UI with tracking info"""
        height, width = frame.shape[:2]
        
        # Draw tracking box and trail
        if tracking_result['success']:
            x, y, w, h = map(int, tracking_result['bbox'])
            
            # Draw main bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw center point
            center_x, center_y = x + w // 2, y + h // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # Draw confidence and tracker type
            label = f"Tracking ({self.current_tracker_type}) - Conf: {tracking_result.get('confidence', 0):.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)
            
            # Draw trail if enabled
            if self.show_trail and len(self.tracking_history) > 1:
                for i in range(1, len(self.tracking_history)):
                    pt1 = (self.tracking_history[i-1][0], self.tracking_history[i-1][1])
                    pt2 = (self.tracking_history[i][0], self.tracking_history[i][1])
                    
                    # Fade trail
                    alpha = i / len(self.tracking_history)
                    trail_color = (0, int(255 * alpha), 0)
                    cv2.line(frame, pt1, pt2, trail_color, 2)
        else:
            # Draw "LOST" indicator
            cv2.putText(frame, "LOST - Move object back into view", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Draw info panel if enabled
        if self.show_info:
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Info text
            info_y = 35
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 25
            cv2.putText(frame, f"Tracker: {self.current_tracker_type}", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 25
            cv2.putText(frame, f"Frame: {self.total_frames}", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 25
            cv2.putText(frame, f"Lost Count: {self.lost_count}", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 25
            
            # Controls
            cv2.putText(frame, "Controls:", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 20
            cv2.putText(frame, "SPACE: Pause | T: Trail | I: Info", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 15
            cv2.putText(frame, "C: Change tracker | Q: Quit", (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Paused indicator
        if self.paused:
            cv2.putText(frame, "PAUSED", (width // 2 - 50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    
    def save_tracking_data(self, filename="tracking_data.json"):
        """Save tracking data to JSON file"""
        data = {
            'total_frames': self.total_frames,
            'lost_count': self.lost_count,
            'tracking_history': list(self.tracking_history),
            'config': self.config
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tracking data saved to {filename}")
    
    def run(self, video_path):
        """Main tracking loop"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        
        # Read first frame
        success, frame = cap.read()
        if not success:
            print("Error: Could not read the video file.")
            return
        
        # Resize frame
        display_width = self.config['display_width']
        display_height = self.config['display_height']
        frame = cv2.resize(frame, (display_width, display_height))
        
        print("Simplified Object Tracker")
        print("Controls:")
        print("- SPACE: Pause/Resume")
        print("- 't': Toggle trail")
        print("- 'i': Toggle info panel")
        print("- 'c': Change tracker type")
        print("- 'q': Quit")
        
        # Select ROI for tracking
        bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object to Track")
        
        if bbox == (0, 0, 0, 0):
            print("No object selected. Exiting.")
            return
        
        # Initialize trackers
        self.csrt_tracker = cv2.legacy.TrackerCSRT_create()
        self.csrt_tracker.init(frame, bbox)
        
        self.mosse_tracker = cv2.legacy.TrackerMOSSE_create()
        self.mosse_tracker.init(frame, bbox)
        
        self.bbox = bbox
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            if not self.paused:
                success, frame = cap.read()
                if not success:
                    break
                
                frame = cv2.resize(frame, (display_width, display_height))
                frame_count += 1
                self.total_frames += 1
                
                # Track object
                tracking_result = self.track_object(frame, frame_count)
                
                # Calculate FPS
                current_time = time.time()
                if frame_count > 0:
                    fps = frame_count / (current_time - start_time)
                    self.fps_counter.append(fps)
                    avg_fps = sum(self.fps_counter) / len(self.fps_counter)
                else:
                    avg_fps = 0
                
                # Draw UI
                self.draw_enhanced_ui(frame, tracking_result, avg_fps)
            
            # Display frame
            cv2.namedWindow("Simplified Tracker", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Simplified Tracker", display_width, display_height)
            cv2.imshow("Simplified Tracker", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")
            elif key == ord('t'):
                self.show_trail = not self.show_trail
                print(f"Trail: {'ON' if self.show_trail else 'OFF'}")
            elif key == ord('i'):
                self.show_info = not self.show_info
                print(f"Info panel: {'ON' if self.show_info else 'OFF'}")
            elif key == ord('c'):
                # Change tracker type
                current_index = self.tracker_types.index(self.current_tracker_type)
                next_index = (current_index + 1) % len(self.tracker_types)
                self.current_tracker_type = self.tracker_types[next_index]
                
                # Reinitialize trackers with current bbox if available
                if self.bbox is not None:
                    self.csrt_tracker = self.create_tracker(self.current_tracker_type)
                    self.csrt_tracker.init(frame, self.bbox)
                    
                    self.mosse_tracker = cv2.legacy.TrackerMOSSE_create()
                    self.mosse_tracker.init(frame, self.bbox)
                
                print(f"Switched to {self.current_tracker_type} tracker")
        
        # Save tracking data if enabled
        if self.config.get('save_tracking_data', False):
            self.save_tracking_data()
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    video_path = input("Enter the path to the video file: ")
    
    tracker = SimplifiedTracker()
    tracker.run(video_path)

if __name__ == "__main__":
    main()
