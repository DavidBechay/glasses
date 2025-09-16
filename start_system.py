#!/usr/bin/env python3
"""
Smart Glasses System - Main Startup Script
Launches both Dock Station and Glasses Simulation
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def start_dock_station():
    """Start the dock station"""
    print("🚀 Starting Dock Station...")
    try:
        dock_process = subprocess.Popen([sys.executable, "dock_station.py"])
        return dock_process
    except Exception as e:
        print(f"❌ Failed to start dock station: {e}")
        return None

def start_glasses_simulation():
    """Start the glasses simulation"""
    print("👓 Starting Glasses Simulation...")
    try:
        glasses_process = subprocess.Popen([sys.executable, "glasses_simulation.py"])
        return glasses_process
    except Exception as e:
        print(f"❌ Failed to start glasses simulation: {e}")
        return None

def main():
    """Main startup function"""
    print("=" * 60)
    print("Smart Glasses System - Complete Simulation")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("dock_station.py").exists():
        print("❌ Error: dock_station.py not found. Please run from the project root directory.")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Cannot continue without required packages")
        return
    
    # Create data directories
    print("📁 Creating data directories...")
    os.makedirs("data/audio", exist_ok=True)
    os.makedirs("data/video", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)
    print("✅ Data directories created")
    
    # Start both applications
    print("\n🚀 Starting Smart Glasses System...")
    
    dock_process = start_dock_station()
    time.sleep(2)  # Give dock station time to start
    
    glasses_process = start_glasses_simulation()
    
    if dock_process and glasses_process:
        print("\n✅ Both applications started successfully!")
        print("\n📱 Dock Station: Main control interface")
        print("👓 Glasses Simulation: Smart glasses interface")
        print("\n💡 Tips:")
        print("   - Use Dock Station to control recording and system status")
        print("   - Use Glasses Simulation to simulate glasses functionality")
        print("   - Connect glasses to dock for full functionality")
        print("\n⏹️  Press Ctrl+C to stop both applications")
        
        try:
            # Wait for processes
            while True:
                time.sleep(1)
                if dock_process.poll() is not None:
                    print("❌ Dock Station stopped unexpectedly")
                    break
                if glasses_process.poll() is not None:
                    print("❌ Glasses Simulation stopped unexpectedly")
                    break
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Smart Glasses System...")
            
            # Terminate processes
            if dock_process:
                dock_process.terminate()
            if glasses_process:
                glasses_process.terminate()
            
            print("✅ Shutdown complete")
    else:
        print("❌ Failed to start one or both applications")

if __name__ == "__main__":
    main()
