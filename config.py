#!/usr/bin/env python3
"""
Smart Glasses System - Core Configuration
Centralized settings and constants for the entire system
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
VIDEO_DIR = DATA_DIR / "video"
LOGS_DIR = DATA_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, AUDIO_DIR, VIDEO_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# System settings
class Settings:
    # Audio settings
    AUDIO_SAMPLE_RATE = 44100
    AUDIO_CHUNK_SIZE = 1024
    AUDIO_CHANNELS = 1
    AUDIO_FORMAT = "int16"
    
    # Video settings
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480
    VIDEO_FPS = 30
    
    # Communication settings
    DOCK_PORT = 8000
    GLASSES_PORT = 8001
    COMMUNICATION_TIMEOUT = 5.0
    
    # GUI settings
    WINDOW_SIZE = (800, 600)
    THEME = "DarkBlue3"
    
    # File settings
    MAX_AUDIO_FILES = 100
    MAX_VIDEO_FILES = 50
    AUTO_CLEANUP_DAYS = 7

# System status
class SystemStatus:
    DOCK_CONNECTED = False
    GLASSES_CONNECTED = False
    AUDIO_RECORDING = False
    VIDEO_RECORDING = False
    PROCESSING_ACTIVE = False

# Global system state
system_status = SystemStatus()
settings = Settings()
