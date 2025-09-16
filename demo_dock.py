#!/usr/bin/env python3
"""
Smart Glasses System - Quick Demo
Demonstrates the working Dock Station functionality
"""

import subprocess
import sys
import time

def main():
    print("=" * 60)
    print("Smart Glasses System - Dock Station Demo")
    print("=" * 60)
    print()
    print("🚀 Starting Dock Station...")
    print("📱 This will open the main control interface")
    print("🎮 Features available:")
    print("   - Audio recording with level visualization")
    print("   - Video recording with live camera feed")
    print("   - System status monitoring")
    print("   - File management")
    print("   - Real-time logging")
    print()
    print("💡 Instructions:")
    print("   1. Click 'Initialize Camera' to set up camera")
    print("   2. Use 'Start Audio Recording' to record audio")
    print("   3. Use 'Start Video Recording' to record video")
    print("   4. Monitor status indicators and logs")
    print("   5. Use 'Open Audio/Video Folder' to see recordings")
    print()
    
    try:
        # Start dock station
        dock_process = subprocess.Popen([sys.executable, "dock_station.py"])
        
        print("✅ Dock Station started successfully!")
        print("⏹️  Press Ctrl+C to stop the application")
        
        # Wait for process
        while True:
            time.sleep(1)
            if dock_process.poll() is not None:
                print("❌ Dock Station stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Dock Station...")
        if dock_process:
            dock_process.terminate()
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()
