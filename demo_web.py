#!/usr/bin/env python3
"""
Smart Glasses Web System - Demo
Clean web-only interface with intelligent features
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("=" * 60)
    print("🔬 Smart Glasses Web System Demo")
    print("=" * 60)
    print()
    print("🚀 Starting Smart Glasses Web System...")
    print("📱 This will open a clean web interface")
    print()
    print("🧠 Intelligent Features:")
    print("   😴 Standby Mode: Audio and camera start in low-power standby")
    print("   🎤 Voice Detection: Automatically increases quality when speech detected")
    print("   📄 Document Detection: Like CamScanner - detects papers and boards")
    print("   📸 Auto Capture: Automatically takes photos when documents detected")
    print()
    print("🎯 How It Works:")
    print("   1. System starts in standby mode (low power, low quality)")
    print("   2. When you speak, voice detection increases audio quality")
    print("   3. Camera continuously scans for documents (papers, boards)")
    print("   4. When documents detected, automatically captures photos")
    print("   5. All features work intelligently in the background")
    print()
    print("🌐 Web Interface Features:")
    print("   - Modern gradient design")
    print("   - Real-time status indicators")
    print("   - Live camera feed")
    print("   - Audio level visualization")
    print("   - Document detection status")
    print("   - Auto-capture counter")
    print("   - System logging")
    print("   - File management")
    print()
    print("📋 Available Controls:")
    print("   - Toggle standby mode")
    print("   - Force audio/video recording")
    print("   - Initialize camera")
    print("   - Toggle auto-capture")
    print("   - Adjust detection sensitivity")
    print("   - Open file folders")
    print("   - Clear old files")
    print()
    
    try:
        # Start web system
        gui_process = subprocess.Popen([sys.executable, "smart_glasses_web.py"])
        
        # Wait for server to start
        time.sleep(3)
        
        # Open browser
        print("🌐 Opening web interface...")
        webbrowser.open('http://localhost:8004')
        
        print("✅ Smart Glasses Web System started successfully!")
        print("📱 Interface: http://localhost:8004")
        print("🧠 Intelligent features are now active")
        print()
        print("💡 Try These Features:")
        print("   1. Speak near the microphone - watch quality increase")
        print("   2. Show papers or boards to camera - auto-capture photos")
        print("   3. Adjust detection sensitivity with the slider")
        print("   4. Toggle standby mode and auto-capture")
        print("   5. Monitor real-time status indicators")
        print("   6. Check the system log for activity")
        print()
        print("⏹️  Press Ctrl+C to stop the application")
        
        # Wait for process
        while True:
            time.sleep(1)
            if gui_process.poll() is not None:
                print("❌ Smart Glasses Web System stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Smart Glasses Web System...")
        if gui_process:
            gui_process.terminate()
        print("✅ Shutdown complete")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
