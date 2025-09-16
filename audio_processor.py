#!/usr/bin/env python3
"""
Smart Glasses System - Audio Processing Module
Handles audio recording, processing, and analysis
"""

import pyaudio
import numpy as np
import wave
import threading
import time
from datetime import datetime
from pathlib import Path
import config

class AudioProcessor:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.recording_thread = None
        self.audio_data = []
        self.sample_rate = config.settings.AUDIO_SAMPLE_RATE
        self.chunk_size = config.settings.AUDIO_CHUNK_SIZE
        self.channels = config.settings.AUDIO_CHANNELS
        
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return False
            
        try:
            self.is_recording = True
            self.audio_data = []
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.start()
            config.system_status.AUDIO_RECORDING = True
            return True
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """Stop audio recording and save to file"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        config.system_status.AUDIO_RECORDING = False
        
        # Save audio data to file
        if self.audio_data:
            filename = self._save_audio_file()
            return filename
        return None
    
    def _recording_loop(self):
        """Main recording loop"""
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            while self.is_recording:
                data = stream.read(self.chunk_size)
                self.audio_data.append(data)
                
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Error in recording loop: {e}")
            self.is_recording = False
    
    def _save_audio_file(self):
        """Save recorded audio to WAV file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = config.AUDIO_DIR / f"recording_{timestamp}.wav"
        
        try:
            with wave.open(str(filename), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_data))
            
            print(f"Audio saved to: {filename}")
            return str(filename)
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return None
    
    def get_audio_level(self):
        """Get current audio level for visualization"""
        if not self.audio_data:
            return 0
        
        try:
            # Get the last chunk of audio data
            last_chunk = self.audio_data[-1] if self.audio_data else b'\x00' * self.chunk_size
            audio_array = np.frombuffer(last_chunk, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array**2))
            # Normalize to 0-100 range
            level = min(100, (rms / 32767) * 100)
            return int(level)
        except:
            return 0
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_recording()
        if hasattr(self, 'audio'):
            self.audio.terminate()
