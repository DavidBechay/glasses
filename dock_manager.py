#!/usr/bin/env python3
"""
Dock System Manager
A Python application that manages both the dock backend and frontend.
"""

import subprocess
import sys
import os
import time
import signal
import threading
import webbrowser
from pathlib import Path
import psutil
import requests
from typing import Optional, List

class DockSystemManager:
    def __init__(self):
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.running = False
        self.backend_port = 8007
        self.frontend_port = 3001
        
        # Paths
        self.project_root = Path(__file__).parent
        self.backend_script = self.project_root / "dock_app.py"
        self.frontend_dir = self.project_root / "dock-frontend"
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\n🛑 Shutting down dock system...")
        self.stop_all()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        print("🔍 Checking dependencies...")
        
        # Check Python
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ is required")
            return False
        
        # Check if dock_app.py exists
        if not self.backend_script.exists():
            print(f"❌ Backend script not found: {self.backend_script}")
            return False
        
        # Check if frontend directory exists
        if not self.frontend_dir.exists():
            print(f"❌ Frontend directory not found: {self.frontend_dir}")
            return False
        
        # Check if Node.js is installed
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Node.js is not installed")
                return False
            print(f"✅ Node.js version: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Node.js is not installed")
            return False
        
        # Check if npm is installed
        try:
            result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ npm is not installed")
                return False
            print(f"✅ npm version: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ npm is not installed")
            return False
        
        # Check if frontend dependencies are installed
        node_modules = self.frontend_dir / "node_modules"
        if not node_modules.exists():
            print("📦 Installing frontend dependencies...")
            try:
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=self.frontend_dir,
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"❌ Failed to install frontend dependencies: {result.stderr}")
                    return False
                print("✅ Frontend dependencies installed")
            except Exception as e:
                print(f"❌ Error installing frontend dependencies: {e}")
                return False
        
        print("✅ All dependencies are available")
        return True
    
    def check_ports(self) -> bool:
        """Check if required ports are available."""
        print("🔍 Checking port availability...")
        
        # Check backend port
        if self._is_port_in_use(self.backend_port):
            print(f"❌ Port {self.backend_port} (backend) is already in use")
            return False
        
        # Check frontend port
        if self._is_port_in_use(self.frontend_port):
            print(f"❌ Port {self.frontend_port} (frontend) is already in use")
            return False
        
        print("✅ All ports are available")
        return True
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False
    
    def start_backend(self) -> bool:
        """Start the dock backend."""
        print(f"🚀 Starting dock backend on port {self.backend_port}...")
        
        try:
            self.backend_process = subprocess.Popen(
                [sys.executable, str(self.backend_script)],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for the backend to start
            time.sleep(3)
            
            # Check if the process is still running
            if self.backend_process.poll() is not None:
                stdout, stderr = self.backend_process.communicate()
                print(f"❌ Backend failed to start: {stderr}")
                return False
            
            # Check if the backend is responding
            if self._wait_for_backend():
                print("✅ Dock backend started successfully")
                return True
            else:
                print("❌ Backend is not responding")
                return False
                
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def start_frontend(self) -> bool:
        """Start the dock frontend."""
        print(f"🚀 Starting dock frontend on port {self.frontend_port}...")
        
        try:
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev", "--", "--port", str(self.frontend_port)],
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for the frontend to start
            time.sleep(5)
            
            # Check if the process is still running
            if self.frontend_process.poll() is not None:
                stdout, stderr = self.frontend_process.communicate()
                print(f"❌ Frontend failed to start: {stderr}")
                return False
            
            # Check if the frontend is responding
            if self._wait_for_frontend():
                print("✅ Dock frontend started successfully")
                return True
            else:
                print("❌ Frontend is not responding")
                return False
                
        except Exception as e:
            print(f"❌ Error starting frontend: {e}")
            return False
    
    def _wait_for_backend(self, timeout: int = 30) -> bool:
        """Wait for the backend to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.backend_port}/api/status", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False
    
    def _wait_for_frontend(self, timeout: int = 30) -> bool:
        """Wait for the frontend to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.frontend_port}", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False
    
    def stop_backend(self):
        """Stop the dock backend."""
        if self.backend_process:
            print("🛑 Stopping dock backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            self.backend_process = None
            print("✅ Dock backend stopped")
    
    def stop_frontend(self):
        """Stop the dock frontend."""
        if self.frontend_process:
            print("🛑 Stopping dock frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            self.frontend_process = None
            print("✅ Dock frontend stopped")
    
    def stop_all(self):
        """Stop all services."""
        self.running = False
        self.stop_frontend()
        self.stop_backend()
    
    def open_browser(self):
        """Open the dock frontend in the browser."""
        try:
            webbrowser.open(f"http://localhost:{self.frontend_port}")
            print(f"🌐 Opened dock frontend in browser: http://localhost:{self.frontend_port}")
        except Exception as e:
            print(f"❌ Failed to open browser: {e}")
    
    def monitor_processes(self):
        """Monitor the running processes."""
        while self.running:
            # Check backend
            if self.backend_process and self.backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                self.running = False
                break
            
            # Check frontend
            if self.frontend_process and self.frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                self.running = False
                break
            
            time.sleep(1)
    
    def run(self):
        """Run the complete dock system."""
        print("🎯 Starting Dock System Manager...")
        print("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed")
            return False
        
        # Check ports
        if not self.check_ports():
            print("❌ Port check failed")
            return False
        
        print("\n🚀 Starting services...")
        
        # Start backend
        if not self.start_backend():
            print("❌ Failed to start backend")
            return False
        
        # Start frontend
        if not self.start_frontend():
            print("❌ Failed to start frontend")
            self.stop_backend()
            return False
        
        # Mark as running
        self.running = True
        
        print("\n🎉 Dock system is running!")
        print("=" * 60)
        print(f"🌐 Dock Web Interface: http://localhost:{self.backend_port}")
        print(f"📱 Dock Frontend:      http://localhost:{self.frontend_port}")
        print(f"🔧 Dock API:           http://localhost:{self.backend_port}/api")
        print(f"🔌 WebSocket:          ws://localhost:{self.backend_port}/ws")
        print("=" * 60)
        print("Press Ctrl+C to stop all services")
        
        # Open browser
        self.open_browser()
        
        # Monitor processes
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            pass
        
        return True

def main():
    """Main entry point."""
    print("🖥️  Dock System Manager")
    print("A Python application to manage the dock backend and frontend")
    print()
    
    manager = DockSystemManager()
    
    try:
        success = manager.run()
        if success:
            print("\n✅ Dock system completed successfully")
        else:
            print("\n❌ Dock system failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        manager.stop_all()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        manager.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
