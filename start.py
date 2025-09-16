#!/usr/bin/env python3
"""
Smart Glasses System - Startup Script
Starts the best document detection system
"""

import subprocess
import sys
import os

def main():
    print("🔬 Starting Smart Glasses - Best Document Detection System")
    print("=" * 60)
    
    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        import cv2
        import numpy
        import pyaudio
        print("✅ All dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed")
    
    print("\n🚀 Starting the system...")
    print("📱 The web interface will open at: http://localhost:8006")
    print("📄 Advanced OpenCV document detection - No people detection")
    print("🎤 Voice detection enabled")
    print("📸 Auto-capture enabled")
    print("\n" + "=" * 60)
    
    # Start the system
    try:
        subprocess.run([sys.executable, "smart_glasses_best.py"])
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting system: {e}")

if __name__ == "__main__":
    main()
