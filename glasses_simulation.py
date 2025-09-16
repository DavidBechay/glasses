#!/usr/bin/env python3
"""
Smart Glasses System - Glasses Simulation GUI
Simulates the Smart Glasses interface and functionality
"""

import PySimpleGUI as sg
import threading
import time
import json
import socket
import cv2
from datetime import datetime
from pathlib import Path
import config
from audio_processor import AudioProcessor
from camera_processor import CameraProcessor

class GlassesSimulationGUI:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.camera_processor = CameraProcessor()
        self.is_running = True
        self.update_thread = None
        self.dock_connected = False
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the glasses simulation GUI layout with modern styling"""
        
        # Define modern color scheme
        colors = {
            'bg': '#0d1117',
            'card': '#161b22',
            'accent': '#58a6ff',
            'success': '#3fb950',
            'warning': '#d29922',
            'error': '#f85149',
            'text': '#f0f6fc',
            'text_secondary': '#8b949e'
        }
        
        # Define layout with modern styling
        layout = [
            # Header with futuristic styling
            [sg.Text("🥽 Smart Glasses Simulation", font=("Segoe UI", 22, "bold"), 
                    justification="center", expand_x=True, text_color=colors['accent'],
                    background_color=colors['bg'])],
            [sg.Text("Advanced AR Interface", font=("Segoe UI", 12), 
                    justification="center", expand_x=True, text_color=colors['text_secondary'],
                    background_color=colors['bg'])],
            [sg.HSeparator(color=colors['accent'])],
            
            # Connection Status with modern cards
            [sg.Text("🔗 Connection Status", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Text("🖥️  Dock Station", font=("Segoe UI", 12, "bold"), size=(18, 1)), 
                 sg.Text("● Disconnected", key="-DOCK_CONNECTION-", text_color=colors['error'], 
                        font=("Segoe UI", 12, "bold"))],
                [sg.Text("🔋 Battery", font=("Segoe UI", 12, "bold"), size=(18, 1)), 
                 sg.Text("85%", key="-BATTERY-", text_color=colors['success'], 
                        font=("Segoe UI", 12, "bold"))],
                [sg.Text("📶 Signal", font=("Segoe UI", 12, "bold"), size=(18, 1)), 
                 sg.Text("Strong", key="-SIGNAL-", text_color=colors['success'], 
                        font=("Segoe UI", 12, "bold"))]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # Glasses Controls with modern buttons
            [sg.Text("🎮 Glasses Controls", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Button("🔗 Connect to Dock", key="-CONNECT_DOCK-", 
                          size=(20, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['success'])),
                 sg.Button("❌ Disconnect", key="-DISCONNECT-", 
                          size=(20, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['error']), disabled=True)],
                [sg.Button("🎬 Start Recording", key="-START_RECORDING-", 
                          size=(20, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['success'])),
                 sg.Button("⏹️ Stop Recording", key="-STOP_RECORDING-", 
                          size=(20, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['error']), disabled=True)],
                [sg.Button("📸 Take Photo", key="-TAKE_PHOTO-", 
                          size=(20, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['accent'])),
                 sg.Button("📺 Start Live Feed", key="-START_FEED-", 
                          size=(20, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['accent']))]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # Live Camera Feed with enhanced styling
            [sg.Text("👁️ Glasses Camera View", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Image(key="-GLASSES_FEED-", size=(400, 300), 
                         background_color=colors['card'])]
            ], background_color=colors['card'], border_width=2, 
              relief=sg.RELIEF_RAISED, pad=(10, 10))],
            
            # Audio Controls with modern styling
            [sg.Text("🎵 Audio Controls", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Button("🎤 Start Audio", key="-START_AUDIO-", 
                          size=(15, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['success'])),
                 sg.Button("⏹️ Stop Audio", key="-STOP_AUDIO-", 
                          size=(15, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['error']), disabled=True)],
                [sg.ProgressBar(100, orientation='h', size=(40, 25), 
                              key="-AUDIO_LEVEL-", bar_color=(colors['success'], colors['card']))],
                [sg.Text("Audio Level: 0%", key="-AUDIO_LEVEL_TEXT-", 
                        font=("Segoe UI", 12, "bold"), text_color=colors['text'],
                        justification="center", expand_x=True)]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # Computer Vision Features with modern buttons
            [sg.Text("🤖 Computer Vision Features", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Button("🎯 Object Detection", key="-OBJECT_DETECTION-", 
                          size=(20, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent'])),
                 sg.Button("📝 Text Recognition", key="-TEXT_RECOGNITION-", 
                          size=(20, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent']))],
                [sg.Button("👤 Face Detection", key="-FACE_DETECTION-", 
                          size=(20, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent'])),
                 sg.Button("🌍 Scene Analysis", key="-SCENE_ANALYSIS-", 
                          size=(20, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent']))]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # Status Display with modern styling
            [sg.Text("📊 Status Display", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Multiline(size=(70, 6), key="-STATUS_DISPLAY-", disabled=True, autoscroll=True,
                            font=("Consolas", 9), background_color=colors['bg'],
                            text_color=colors['text'])]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            [sg.Button("🗑️ Clear Status", key="-CLEAR_STATUS-", size=(15, 1),
                      font=("Segoe UI", 10, "bold"),
                      button_color=(colors['text'], colors['warning']))],
            
            # Settings with modern buttons
            [sg.Text("⚙️ Glasses Settings", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Button("💡 Brightness", key="-BRIGHTNESS-", 
                          size=(15, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent'])),
                 sg.Button("🔊 Volume", key="-VOLUME-", 
                          size=(15, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent']))],
                [sg.Button("🖥️ Display Mode", key="-DISPLAY_MODE-", 
                          size=(15, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent'])),
                 sg.Button("🔋 Power Save", key="-POWER_SAVE-", 
                          size=(15, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent']))]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # Footer with exit button
            [sg.HSeparator(color=colors['accent'])],
            [sg.Button("🚪 Exit", key="-EXIT-", size=(20, 2), 
                      font=("Segoe UI", 12, "bold"),
                      button_color=(colors['text'], colors['error']))]
        ]
        
        # Create window with modern styling
        self.window = sg.Window("🥽 Smart Glasses Simulation", layout, 
                              size=(1000, 900), resizable=True, finalize=True,
                              background_color=colors['bg'],
                              element_justification='center')
        
        # Start update thread
        self.start_update_thread()
        
    def start_update_thread(self):
        """Start the GUI update thread"""
        self.update_thread = threading.Thread(target=self.update_gui, daemon=True)
        self.update_thread.start()
        
    def update_gui(self):
        """Update GUI elements in real-time"""
        while self.is_running:
            try:
                # Update connection status
                self.window["-DOCK_CONNECTION-"].update(
                    "Connected" if self.dock_connected else "Disconnected",
                    text_color="green" if self.dock_connected else "red"
                )
                
                # Update battery level (simulate decreasing)
                if hasattr(self, 'battery_level'):
                    self.battery_level = max(0, self.battery_level - 0.01)
                else:
                    self.battery_level = 85.0
                
                self.window["-BATTERY-"].update(f"{self.battery_level:.1f}%", 
                    text_color="green" if self.battery_level > 20 else "red")
                
                # Update audio level
                audio_level = self.audio_processor.get_audio_level()
                self.window["-AUDIO_LEVEL-"].update(audio_level)
                self.window["-AUDIO_LEVEL_TEXT-"].update(f"Audio Level: {audio_level}%")
                
                # Update glasses camera feed
                frame = self.camera_processor.get_current_frame()
                if frame is not None:
                    # Convert OpenCV frame to PySimpleGUI format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imgbytes = cv2.imencode('.png', frame_rgb)[1].tobytes()
                    self.window["-GLASSES_FEED-"].update(data=imgbytes)
                
                time.sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                self.log_status(f"Error in GUI update: {e}")
                time.sleep(1)
    
    def log_status(self, message):
        """Add message to status display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_entry = f"[{timestamp}] {message}\n"
        
        # Get current status content
        current_status = self.window["-STATUS_DISPLAY-"].get()
        
        # Add new message
        new_status = current_status + status_entry
        
        # Update status (keep last 500 lines)
        lines = new_status.split('\n')
        if len(lines) > 500:
            lines = lines[-500:]
            new_status = '\n'.join(lines)
        
        self.window["-STATUS_DISPLAY-"].update(new_status)
    
    def handle_events(self):
        """Handle GUI events"""
        while True:
            event, values = self.window.read(timeout=100)
            
            if event == sg.WIN_CLOSED or event == "-EXIT-":
                break
            
            elif event == "-CONNECT_DOCK-":
                self.connect_to_dock()
            
            elif event == "-DISCONNECT-":
                self.disconnect_from_dock()
            
            elif event == "-START_RECORDING-":
                self.start_recording()
            
            elif event == "-STOP_RECORDING-":
                self.stop_recording()
            
            elif event == "-TAKE_PHOTO-":
                self.take_photo()
            
            elif event == "-START_FEED-":
                self.start_live_feed()
            
            elif event == "-START_AUDIO-":
                if self.audio_processor.start_recording():
                    self.window["-START_AUDIO-"].update(disabled=True)
                    self.window["-STOP_AUDIO-"].update(disabled=False)
                    self.log_status("Audio recording started")
                else:
                    self.log_status("Failed to start audio recording")
            
            elif event == "-STOP_AUDIO-":
                filename = self.audio_processor.stop_recording()
                self.window["-START_AUDIO-"].update(disabled=False)
                self.window["-STOP_AUDIO-"].update(disabled=True)
                if filename:
                    self.log_status(f"Audio recording saved: {filename}")
                else:
                    self.log_status("Audio recording stopped")
            
            elif event == "-OBJECT_DETECTION-":
                self.object_detection()
            
            elif event == "-TEXT_RECOGNITION-":
                self.text_recognition()
            
            elif event == "-FACE_DETECTION-":
                self.face_detection()
            
            elif event == "-SCENE_ANALYSIS-":
                self.scene_analysis()
            
            elif event == "-CLEAR_STATUS-":
                self.window["-STATUS_DISPLAY-"].update("")
                self.log_status("Status cleared")
            
            elif event == "-BRIGHTNESS-":
                self.adjust_brightness()
            
            elif event == "-VOLUME-":
                self.adjust_volume()
            
            elif event == "-DISPLAY_MODE-":
                self.change_display_mode()
            
            elif event == "-POWER_SAVE-":
                self.toggle_power_save()
        
        self.cleanup()
    
    def connect_to_dock(self):
        """Connect to dock station"""
        try:
            self.log_status("Connecting to dock station...")
            time.sleep(2)  # Simulate connection delay
            
            self.dock_connected = True
            self.window["-CONNECT_DOCK-"].update(disabled=True)
            self.window["-DISCONNECT-"].update(disabled=False)
            self.log_status("Connected to dock station successfully")
            
        except Exception as e:
            self.log_status(f"Failed to connect to dock: {e}")
    
    def disconnect_from_dock(self):
        """Disconnect from dock station"""
        self.dock_connected = False
        self.window["-CONNECT_DOCK-"].update(disabled=False)
        self.window["-DISCONNECT-"].update(disabled=True)
        self.log_status("Disconnected from dock station")
    
    def start_recording(self):
        """Start recording video and audio"""
        if not self.dock_connected:
            self.log_status("Cannot start recording - not connected to dock")
            return
        
        if self.camera_processor.start_recording():
            self.window["-START_RECORDING-"].update(disabled=True)
            self.window["-STOP_RECORDING-"].update(disabled=False)
            self.log_status("Recording started")
        else:
            self.log_status("Failed to start recording")
    
    def stop_recording(self):
        """Stop recording"""
        filename = self.camera_processor.stop_recording()
        self.window["-START_RECORDING-"].update(disabled=False)
        self.window["-STOP_RECORDING-"].update(disabled=True)
        if filename:
            self.log_status(f"Recording saved: {filename}")
        else:
            self.log_status("Recording stopped")
    
    def take_photo(self):
        """Take a photo"""
        frame = self.camera_processor.get_current_frame()
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_file = config.VIDEO_DIR / f"photo_{timestamp}.jpg"
            
            try:
                cv2.imwrite(str(photo_file), frame)
                self.log_status(f"Photo saved: {photo_file}")
            except Exception as e:
                self.log_status(f"Failed to save photo: {e}")
        else:
            self.log_status("No camera feed available")
    
    def start_live_feed(self):
        """Start live camera feed"""
        if self.camera_processor.initialize_camera():
            self.log_status("Live camera feed started")
        else:
            self.log_status("Failed to start live camera feed")
    
    def object_detection(self):
        """Simulate object detection"""
        frame = self.camera_processor.get_current_frame()
        if frame is not None:
            objects = self.camera_processor.detect_objects(frame)
            self.log_status(f"Detected {len(objects)} objects")
            for obj in objects:
                self.log_status(f"  - {obj['type']} (confidence: {obj['confidence']:.2f})")
        else:
            self.log_status("No camera feed for object detection")
    
    def text_recognition(self):
        """Simulate text recognition"""
        self.log_status("Text recognition: 'Smart Glasses System' detected")
        self.log_status("Text recognition: 'Version 1.0' detected")
    
    def face_detection(self):
        """Simulate face detection"""
        self.log_status("Face detection: 1 face detected")
        self.log_status("Face detection: Confidence: 0.95")
    
    def scene_analysis(self):
        """Simulate scene analysis"""
        self.log_status("Scene analysis: Indoor environment detected")
        self.log_status("Scene analysis: Lighting: Good")
        self.log_status("Scene analysis: Objects: Multiple")
    
    def adjust_brightness(self):
        """Adjust display brightness"""
        brightness = sg.popup_get_text("Enter brightness level (1-100):", "Brightness Control")
        if brightness:
            try:
                level = int(brightness)
                if 1 <= level <= 100:
                    self.log_status(f"Brightness adjusted to {level}%")
                else:
                    self.log_status("Invalid brightness level")
            except ValueError:
                self.log_status("Invalid brightness input")
    
    def adjust_volume(self):
        """Adjust audio volume"""
        volume = sg.popup_get_text("Enter volume level (0-100):", "Volume Control")
        if volume:
            try:
                level = int(volume)
                if 0 <= level <= 100:
                    self.log_status(f"Volume adjusted to {level}%")
                else:
                    self.log_status("Invalid volume level")
            except ValueError:
                self.log_status("Invalid volume input")
    
    def change_display_mode(self):
        """Change display mode"""
        modes = ["Normal", "Night Vision", "High Contrast", "Color Blind"]
        mode = sg.popup_get_text("Select display mode:", "Display Mode", 
                                default_text="Normal")
        if mode and mode in modes:
            self.log_status(f"Display mode changed to: {mode}")
        else:
            self.log_status("Invalid display mode")
    
    def toggle_power_save(self):
        """Toggle power save mode"""
        if hasattr(self, 'power_save'):
            self.power_save = not self.power_save
        else:
            self.power_save = True
        
        status = "enabled" if self.power_save else "disabled"
        self.log_status(f"Power save mode {status}")
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        self.audio_processor.cleanup()
        self.camera_processor.cleanup()
        self.window.close()

def main():
    """Main function"""
    print("Starting Smart Glasses Simulation...")
    
    try:
        glasses_gui = GlassesSimulationGUI()
        glasses_gui.log_status("Glasses simulation initialized successfully")
        glasses_gui.handle_events()
    except Exception as e:
        print(f"Error starting glasses simulation: {e}")
    finally:
        print("Glasses simulation shutdown complete")

if __name__ == "__main__":
    main()
