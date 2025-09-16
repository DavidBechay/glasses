#!/usr/bin/env python3
"""
Smart Glasses System - Camera/Vision Processing Module
Handles camera capture, image processing, and computer vision
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from pathlib import Path
import config

class CameraProcessor:
    def __init__(self):
        self.camera = None
        self.is_recording = False
        self.recording_thread = None
        self.current_frame = None
        self.frame_width = config.settings.VIDEO_WIDTH
        self.frame_height = config.settings.VIDEO_HEIGHT
        self.fps = config.settings.VIDEO_FPS
        
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self.camera.set(cv2.CAP_PROP_FPS, self.fps)
                return True
            else:
                print("Failed to open camera")
                return False
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def start_recording(self):
        """Start video recording"""
        if self.is_recording:
            return False
            
        if not self.camera or not self.camera.isOpened():
            if not self.initialize_camera():
                return False
        
        try:
            self.is_recording = True
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.start()
            config.system_status.VIDEO_RECORDING = True
            return True
        except Exception as e:
            print(f"Error starting video recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """Stop video recording and save to file"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        config.system_status.VIDEO_RECORDING = False
        
        # Save video data to file
        if hasattr(self, 'video_writer') and self.video_writer:
            self.video_writer.release()
            return self.current_video_file
        return None
    
    def _recording_loop(self):
        """Main recording loop"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_video_file = config.VIDEO_DIR / f"recording_{timestamp}.mp4"
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                str(self.current_video_file),
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height)
            )
            
            while self.is_recording and self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    # Process frame (add timestamp, etc.)
                    processed_frame = self._process_frame(frame)
                    self.current_frame = processed_frame.copy()
                    
                    # Write frame to video file
                    self.video_writer.write(processed_frame)
                else:
                    break
                    
        except Exception as e:
            print(f"Error in recording loop: {e}")
            self.is_recording = False
    
    def _process_frame(self, frame):
        """Process frame with computer vision techniques"""
        try:
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add frame counter
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 1
            cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Apply some computer vision processing
            # Edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            # Blend original with edges
            processed_frame = cv2.addWeighted(frame, 0.8, edges_colored, 0.2, 0)
            
            return processed_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
    
    def get_current_frame(self):
        """Get current frame for display"""
        if self.current_frame is not None:
            return self.current_frame
        elif self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return self._process_frame(frame)
        return None
    
    def detect_objects(self, frame):
        """Simple object detection simulation"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_objects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    detected_objects.append({
                        'type': 'object',
                        'confidence': min(0.9, area / 10000),
                        'bbox': (x, y, w, h)
                    })
            
            return detected_objects
        except Exception as e:
            print(f"Error in object detection: {e}")
            return []
    
    def cleanup(self):
        """Clean up camera resources"""
        self.stop_recording()
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()
