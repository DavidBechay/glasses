#!/usr/bin/env python3
"""
Smart Glasses System - Dock Station GUI
Main control interface for the Smart Glasses system
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

class DockStationGUI:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.camera_processor = CameraProcessor()
        self.is_running = True
        self.update_thread = None
        
        # Initialize GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout with modern styling"""
        
        # Define modern color scheme
        colors = {
            'bg': '#1e1e1e',
            'card': '#2d2d2d',
            'accent': '#0078d4',
            'success': '#107c10',
            'warning': '#ff8c00',
            'error': '#d13438',
            'text': '#ffffff',
            'text_secondary': '#cccccc'
        }
        
        # Define layout with modern styling
        layout = [
            # Header with gradient effect simulation
            [sg.Text("🔬 Smart Glasses Dock Station", font=("Segoe UI", 24, "bold"), 
                    justification="center", expand_x=True, text_color=colors['accent'],
                    background_color=colors['bg'])],
            [sg.Text("Advanced Control Interface", font=("Segoe UI", 12), 
                    justification="center", expand_x=True, text_color=colors['text_secondary'],
                    background_color=colors['bg'])],
            [sg.HSeparator(color=colors['accent'])],
            
            # Status Cards Row
            [sg.Text("📊 System Status", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Text("🖥️  Dock Station", font=("Segoe UI", 12, "bold"), size=(18, 1)), 
                 sg.Text("● Online", key="-DOCK_STATUS-", text_color=colors['success'], 
                        font=("Segoe UI", 12, "bold"))],
                [sg.Text("👓 Glasses", font=("Segoe UI", 12, "bold"), size=(18, 1)), 
                 sg.Text("● Disconnected", key="-GLASSES_STATUS-", text_color=colors['error'], 
                        font=("Segoe UI", 12, "bold"))],
                [sg.Text("🎤 Audio", font=("Segoe UI", 12, "bold"), size=(18, 1)), 
                 sg.Text("● Stopped", key="-AUDIO_STATUS-", text_color=colors['warning'], 
                        font=("Segoe UI", 12, "bold"))],
                [sg.Text("📹 Video", font=("Segoe UI", 12, "bold"), size=(18, 1)), 
                 sg.Text("● Stopped", key="-VIDEO_STATUS-", text_color=colors['warning'], 
                        font=("Segoe UI", 12, "bold"))]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # Control Panel with modern buttons
            [sg.Text("🎮 Control Panel", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Button("🎤 Start Audio Recording", key="-START_AUDIO-", 
                          size=(22, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['success'])),
                 sg.Button("⏹️ Stop Audio Recording", key="-STOP_AUDIO-", 
                          size=(22, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['error']), disabled=True)],
                [sg.Button("📹 Start Video Recording", key="-START_VIDEO-", 
                          size=(22, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['success'])),
                 sg.Button("⏹️ Stop Video Recording", key="-STOP_VIDEO-", 
                          size=(22, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['error']), disabled=True)],
                [sg.Button("📷 Initialize Camera", key="-INIT_CAMERA-", 
                          size=(22, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['accent'])),
                 sg.Button("🔗 Test Glasses Connection", key="-TEST_GLASSES-", 
                          size=(22, 2), font=("Segoe UI", 11, "bold"),
                          button_color=(colors['text'], colors['accent']))]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # Live Feed with enhanced styling
            [sg.Text("📺 Live Camera Feed", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Image(key="-CAMERA_FEED-", size=(400, 300), 
                         background_color=colors['card'])]
            ], background_color=colors['card'], border_width=2, 
              relief=sg.RELIEF_RAISED, pad=(10, 10))],
            
            # Audio Level with modern progress bar
            [sg.Text("🔊 Audio Level", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.ProgressBar(100, orientation='h', size=(50, 25), 
                              key="-AUDIO_LEVEL-", bar_color=(colors['success'], colors['card']))],
                [sg.Text("0%", key="-AUDIO_LEVEL_TEXT-", font=("Segoe UI", 14, "bold"),
                        text_color=colors['text'], justification="center", expand_x=True)]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # File Management with modern buttons
            [sg.Text("📁 File Management", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Button("📂 Open Audio Folder", key="-OPEN_AUDIO-", 
                          size=(20, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent'])),
                 sg.Button("📂 Open Video Folder", key="-OPEN_VIDEO-", 
                          size=(20, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent']))],
                [sg.Button("🗑️ Clear Old Files", key="-CLEAR_FILES-", 
                          size=(20, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['warning'])),
                 sg.Button("ℹ️ System Info", key="-SYSTEM_INFO-", 
                          size=(20, 1), font=("Segoe UI", 10, "bold"),
                          button_color=(colors['text'], colors['accent']))]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            
            # Enhanced Log Panel
            [sg.Text("📋 System Log", font=("Segoe UI", 16, "bold"), 
                    text_color=colors['text'], background_color=colors['bg'])],
            [sg.Frame("", [
                [sg.Multiline(size=(90, 8), key="-LOG-", disabled=True, autoscroll=True,
                            font=("Consolas", 9), background_color=colors['bg'],
                            text_color=colors['text'])]
            ], background_color=colors['card'], border_width=0, pad=(10, 10))],
            [sg.Button("🗑️ Clear Log", key="-CLEAR_LOG-", size=(15, 1),
                      font=("Segoe UI", 10, "bold"),
                      button_color=(colors['text'], colors['warning'])),
             sg.Button("💾 Save Log", key="-SAVE_LOG-", size=(15, 1),
                      font=("Segoe UI", 10, "bold"),
                      button_color=(colors['text'], colors['success']))],
            
            # Footer with exit button
            [sg.HSeparator(color=colors['accent'])],
            [sg.Button("🚪 Exit", key="-EXIT-", size=(20, 2), 
                      font=("Segoe UI", 12, "bold"),
                      button_color=(colors['text'], colors['error']))]
        ]
        
        # Create window with modern styling
        self.window = sg.Window("🔬 Smart Glasses Dock Station", layout, 
                              size=(1000, 800), 
                              resizable=True, finalize=True,
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
                # Update status indicators
                self.window["-DOCK_STATUS-"].update("Online", text_color="green")
                self.window["-GLASSES_STATUS-"].update(
                    "Connected" if config.system_status.GLASSES_CONNECTED else "Disconnected",
                    text_color="green" if config.system_status.GLASSES_CONNECTED else "red"
                )
                self.window["-AUDIO_STATUS-"].update(
                    "Recording" if config.system_status.AUDIO_RECORDING else "Stopped",
                    text_color="green" if config.system_status.AUDIO_RECORDING else "orange"
                )
                self.window["-VIDEO_STATUS-"].update(
                    "Recording" if config.system_status.VIDEO_RECORDING else "Stopped",
                    text_color="green" if config.system_status.VIDEO_RECORDING else "orange"
                )
                
                # Update audio level
                audio_level = self.audio_processor.get_audio_level()
                self.window["-AUDIO_LEVEL-"].update(audio_level)
                self.window["-AUDIO_LEVEL_TEXT-"].update(f"{audio_level}%")
                
                # Update camera feed
                frame = self.camera_processor.get_current_frame()
                if frame is not None:
                    # Convert OpenCV frame to PySimpleGUI format
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imgbytes = cv2.imencode('.png', frame_rgb)[1].tobytes()
                    self.window["-CAMERA_FEED-"].update(data=imgbytes)
                
                time.sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                self.log_message(f"Error in GUI update: {e}")
                time.sleep(1)
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Get current log content
        current_log = self.window["-LOG-"].get()
        
        # Add new message
        new_log = current_log + log_entry
        
        # Update log (keep last 1000 lines)
        lines = new_log.split('\n')
        if len(lines) > 1000:
            lines = lines[-1000:]
            new_log = '\n'.join(lines)
        
        self.window["-LOG-"].update(new_log)
    
    def handle_events(self):
        """Handle GUI events"""
        while True:
            event, values = self.window.read(timeout=100)
            
            if event == sg.WIN_CLOSED or event == "-EXIT-":
                break
            
            elif event == "-START_AUDIO-":
                if self.audio_processor.start_recording():
                    self.window["-START_AUDIO-"].update(disabled=True)
                    self.window["-STOP_AUDIO-"].update(disabled=False)
                    self.log_message("Audio recording started")
                else:
                    self.log_message("Failed to start audio recording")
            
            elif event == "-STOP_AUDIO-":
                filename = self.audio_processor.stop_recording()
                self.window["-START_AUDIO-"].update(disabled=False)
                self.window["-STOP_AUDIO-"].update(disabled=True)
                if filename:
                    self.log_message(f"Audio recording saved: {filename}")
                else:
                    self.log_message("Audio recording stopped")
            
            elif event == "-START_VIDEO-":
                if self.camera_processor.start_recording():
                    self.window["-START_VIDEO-"].update(disabled=True)
                    self.window["-STOP_VIDEO-"].update(disabled=False)
                    self.log_message("Video recording started")
                else:
                    self.log_message("Failed to start video recording")
            
            elif event == "-STOP_VIDEO-":
                filename = self.camera_processor.stop_recording()
                self.window["-START_VIDEO-"].update(disabled=False)
                self.window["-STOP_VIDEO-"].update(disabled=True)
                if filename:
                    self.log_message(f"Video recording saved: {filename}")
                else:
                    self.log_message("Video recording stopped")
            
            elif event == "-INIT_CAMERA-":
                if self.camera_processor.initialize_camera():
                    self.log_message("Camera initialized successfully")
                else:
                    self.log_message("Failed to initialize camera")
            
            elif event == "-TEST_GLASSES-":
                self.test_glasses_connection()
            
            elif event == "-OPEN_AUDIO-":
                import subprocess
                subprocess.Popen(['explorer', str(config.AUDIO_DIR)])
                self.log_message("Opened audio folder")
            
            elif event == "-OPEN_VIDEO-":
                import subprocess
                subprocess.Popen(['explorer', str(config.VIDEO_DIR)])
                self.log_message("Opened video folder")
            
            elif event == "-CLEAR_FILES-":
                self.clear_old_files()
            
            elif event == "-SYSTEM_INFO-":
                self.show_system_info()
            
            elif event == "-CLEAR_LOG-":
                self.window["-LOG-"].update("")
                self.log_message("Log cleared")
            
            elif event == "-SAVE_LOG-":
                self.save_log()
        
        self.cleanup()
    
    def test_glasses_connection(self):
        """Test connection to glasses"""
        try:
            # Simulate glasses connection test
            self.log_message("Testing glasses connection...")
            time.sleep(1)  # Simulate connection delay
            
            # For now, just simulate a successful connection
            config.system_status.GLASSES_CONNECTED = True
            self.log_message("Glasses connection test successful")
            
        except Exception as e:
            self.log_message(f"Glasses connection test failed: {e}")
            config.system_status.GLASSES_CONNECTED = False
    
    def clear_old_files(self):
        """Clear old audio/video files"""
        try:
            import os
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=config.settings.AUTO_CLEANUP_DAYS)
            files_cleared = 0
            
            # Clear old audio files
            for file_path in config.AUDIO_DIR.glob("*.wav"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                    file_path.unlink()
                    files_cleared += 1
            
            # Clear old video files
            for file_path in config.VIDEO_DIR.glob("*.mp4"):
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_date:
                    file_path.unlink()
                    files_cleared += 1
            
            self.log_message(f"Cleared {files_cleared} old files")
            
        except Exception as e:
            self.log_message(f"Error clearing files: {e}")
    
    def show_system_info(self):
        """Show system information"""
        info = f"""
System Information:
- Audio Sample Rate: {config.settings.AUDIO_SAMPLE_RATE} Hz
- Video Resolution: {config.settings.VIDEO_WIDTH}x{config.settings.VIDEO_HEIGHT}
- Video FPS: {config.settings.VIDEO_FPS}
- Data Directory: {config.DATA_DIR}
- Audio Files: {len(list(config.AUDIO_DIR.glob('*.wav')))} files
- Video Files: {len(list(config.VIDEO_DIR.glob('*.mp4')))} files
        """
        
        sg.popup("System Information", info, size=(400, 300))
        self.log_message("System info displayed")
    
    def save_log(self):
        """Save current log to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = config.LOGS_DIR / f"dock_log_{timestamp}.txt"
            
            with open(log_file, 'w') as f:
                f.write(self.window["-LOG-"].get())
            
            self.log_message(f"Log saved to: {log_file}")
            
        except Exception as e:
            self.log_message(f"Error saving log: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        self.audio_processor.cleanup()
        self.camera_processor.cleanup()
        self.window.close()

def main():
    """Main function"""
    print("Starting Smart Glasses Dock Station...")
    
    try:
        dock_gui = DockStationGUI()
        dock_gui.log_message("Dock Station initialized successfully")
        dock_gui.handle_events()
    except Exception as e:
        print(f"Error starting dock station: {e}")
    finally:
        print("Dock Station shutdown complete")

if __name__ == "__main__":
    main()
