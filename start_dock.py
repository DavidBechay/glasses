#!/usr/bin/env python3
"""
Simple Dock App Starter
Starts the dock app and shows the localhost link clearly.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("🚀 Starting Smart Glasses Dock Application...")
    print("=" * 60)
    
    # Check if dock_app.py exists
    dock_script = Path("dock_app.py")
    if not dock_script.exists():
        print("❌ dock_app.py not found!")
        return
    
    try:
        # Start the dock app
        process = subprocess.Popen([sys.executable, "dock_app.py"])
        
        # Wait a moment for it to start
        time.sleep(3)
        
        # Show the localhost link
        print("\n🎉 Dock Application Started!")
        print("=" * 60)
        print("🌐 Web Interface: http://localhost:8007")
        print("📱 API Endpoint:  http://localhost:8007/api")
        print("🔌 WebSocket:     ws://localhost:8007/ws")
        print("=" * 60)
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8007")
            print("🌐 Opened in browser!")
        except:
            print("❌ Could not open browser automatically")
        
        # Wait for user to stop
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping dock application...")
            process.terminate()
            process.wait()
            print("✅ Dock application stopped")
            
    except Exception as e:
        print(f"❌ Error starting dock app: {e}")

if __name__ == "__main__":
    main()
