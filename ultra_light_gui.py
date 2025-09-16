#!/usr/bin/env python3
"""
Smart Glasses System - Ultra Lightweight Web GUI
Minimal web interface for maximum performance
"""

import asyncio
import json
import base64
import time
from datetime import datetime
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import config
from audio_processor import AudioProcessor
from camera_processor import CameraProcessor

app = FastAPI(title="Smart Glasses Ultra Light GUI")

# Initialize processors
audio_processor = AudioProcessor()
camera_processor = CameraProcessor()

# WebSocket connections
active_connections = []

@app.get("/")
async def get_main_page():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Glasses Ultra Light</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            
            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .card {
                background: rgba(255, 255, 255, 0.15);
                border-radius: 15px;
                padding: 20px;
                backdrop-filter: blur(5px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
            }
            
            .card h3 {
                font-size: 1.3rem;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .status-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                margin-bottom: 20px;
            }
            
            .status-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #ff4757;
                animation: pulse 2s infinite;
            }
            
            .status-indicator.online {
                background: #2ed573;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                border: none;
                color: white;
                padding: 12px 24px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1rem;
                font-weight: 600;
                transition: all 0.3s ease;
                margin: 5px;
                min-width: 120px;
            }
            
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            }
            
            .btn:active {
                transform: translateY(0);
            }
            
            .btn-danger {
                background: linear-gradient(45deg, #ff4757, #ff3742);
            }
            
            .btn-success {
                background: linear-gradient(45deg, #2ed573, #1e90ff);
            }
            
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }
            
            .progress-bar {
                width: 100%;
                height: 20px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #2ed573, #1e90ff);
                border-radius: 10px;
                transition: width 0.3s ease;
                width: 0%;
            }
            
            .camera-feed {
                width: 100%;
                max-width: 400px;
                height: 300px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 15px;
                margin: 20px auto;
                display: block;
                border: 2px solid rgba(255, 255, 255, 0.2);
            }
            
            .log-container {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 15px;
                height: 200px;
                overflow-y: auto;
                font-family: 'Consolas', monospace;
                font-size: 0.9rem;
                margin-top: 20px;
            }
            
            .log-entry {
                margin-bottom: 5px;
                padding: 2px 0;
            }
            
            .log-timestamp {
                color: #ffa502;
                font-weight: bold;
            }
            
            .log-message {
                color: #ffffff;
            }
            
            .log-error {
                color: #ff4757;
            }
            
            .log-success {
                color: #2ed573;
            }
            
            .controls-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin: 15px 0;
            }
            
            .file-info {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                margin: 5px 0;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 15px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
                
                .grid {
                    grid-template-columns: 1fr;
                }
                
                .status-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 Smart Glasses Ultra Light</h1>
                <p>Minimal Interface • Maximum Performance</p>
            </div>
            
            <div class="grid">
                <!-- System Status Card -->
                <div class="card">
                    <h3>📊 System Status</h3>
                    <div class="status-grid">
                        <div class="status-item">
                            <span>🖥️ Dock Station</span>
                            <div class="status-indicator online" id="dock-status"></div>
                        </div>
                        <div class="status-item">
                            <span>👓 Glasses</span>
                            <div class="status-indicator" id="glasses-status"></div>
                        </div>
                        <div class="status-item">
                            <span>🎤 Audio</span>
                            <div class="status-indicator" id="audio-status"></div>
                        </div>
                        <div class="status-item">
                            <span>📹 Video</span>
                            <div class="status-indicator" id="video-status"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Audio Controls Card -->
                <div class="card">
                    <h3>🎵 Audio Controls</h3>
                    <div class="controls-grid">
                        <button class="btn btn-success" id="start-audio">🎤 Start Recording</button>
                        <button class="btn btn-danger" id="stop-audio" disabled>⏹️ Stop Recording</button>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="audio-level"></div>
                    </div>
                    <div style="text-align: center; margin-top: 10px;">
                        <span id="audio-level-text">Audio Level: 0%</span>
                    </div>
                </div>
                
                <!-- Video Controls Card -->
                <div class="card">
                    <h3>📹 Video Controls</h3>
                    <div class="controls-grid">
                        <button class="btn btn-success" id="start-video">📹 Start Recording</button>
                        <button class="btn btn-danger" id="stop-video" disabled>⏹️ Stop Recording</button>
                        <button class="btn" id="init-camera">📷 Initialize Camera</button>
                    </div>
                    <img id="camera-feed" class="camera-feed" style="display: none;">
                </div>
                
                <!-- File Management Card -->
                <div class="card">
                    <h3>📁 File Management</h3>
                    <div class="controls-grid">
                        <button class="btn" id="open-audio">📂 Audio Folder</button>
                        <button class="btn" id="open-video">📂 Video Folder</button>
                        <button class="btn" id="clear-files">🗑️ Clear Old Files</button>
                        <button class="btn" id="system-info">ℹ️ System Info</button>
                    </div>
                    <div id="file-info"></div>
                </div>
            </div>
            
            <!-- System Log -->
            <div class="card">
                <h3>📋 System Log</h3>
                <div class="log-container" id="log-container"></div>
                <div style="text-align: center; margin-top: 10px;">
                    <button class="btn" id="clear-log">🗑️ Clear Log</button>
                    <button class="btn" id="save-log">💾 Save Log</button>
                </div>
            </div>
        </div>
        
        <script>
            let ws;
            let isRecording = false;
            
            // Initialize WebSocket connection
            function initWebSocket() {
                ws = new WebSocket('ws://localhost:8002/ws');
                
                ws.onopen = function() {
                    addLog('Connected to Smart Glasses system', 'success');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function() {
                    addLog('Connection lost. Reconnecting...', 'error');
                    setTimeout(initWebSocket, 3000);
                };
                
                ws.onerror = function(error) {
                    addLog('WebSocket error: ' + error, 'error');
                };
            }
            
            // Handle incoming messages
            function handleMessage(data) {
                switch(data.type) {
                    case 'status_update':
                        updateStatus(data.status);
                        break;
                    case 'audio_level':
                        updateAudioLevel(data.level);
                        break;
                    case 'camera_frame':
                        updateCameraFeed(data.frame);
                        break;
                    case 'log_message':
                        addLog(data.message, data.level || 'info');
                        break;
                    case 'file_info':
                        updateFileInfo(data.info);
                        break;
                }
            }
            
            // Update system status
            function updateStatus(status) {
                document.getElementById('dock-status').className = 
                    'status-indicator' + (status.dock ? ' online' : '');
                document.getElementById('glasses-status').className = 
                    'status-indicator' + (status.glasses ? ' online' : '');
                document.getElementById('audio-status').className = 
                    'status-indicator' + (status.audio ? ' online' : '');
                document.getElementById('video-status').className = 
                    'status-indicator' + (status.video ? ' online' : '');
            }
            
            // Update audio level
            function updateAudioLevel(level) {
                document.getElementById('audio-level').style.width = level + '%';
                document.getElementById('audio-level-text').textContent = 
                    'Audio Level: ' + level + '%';
            }
            
            // Update camera feed
            function updateCameraFeed(frameData) {
                const img = document.getElementById('camera-feed');
                img.src = 'data:image/jpeg;base64,' + frameData;
                img.style.display = 'block';
            }
            
            // Add log message
            function addLog(message, level = 'info') {
                const container = document.getElementById('log-container');
                const timestamp = new Date().toLocaleTimeString();
                
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.innerHTML = `
                    <span class="log-timestamp">[${timestamp}]</span>
                    <span class="log-${level}">${message}</span>
                `;
                
                container.appendChild(entry);
                container.scrollTop = container.scrollHeight;
                
                // Keep only last 100 entries
                while (container.children.length > 100) {
                    container.removeChild(container.firstChild);
                }
            }
            
            // Update file info
            function updateFileInfo(info) {
                const container = document.getElementById('file-info');
                container.innerHTML = `
                    <div class="file-info">
                        <span>Audio Files: ${info.audio_files}</span>
                        <span>Video Files: ${info.video_files}</span>
                    </div>
                `;
            }
            
            // Send command to server
            function sendCommand(command, data = {}) {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: command,
                        ...data
                    }));
                }
            }
            
            // Event listeners
            document.getElementById('start-audio').onclick = function() {
                sendCommand('start_audio');
                this.disabled = true;
                document.getElementById('stop-audio').disabled = false;
                addLog('Audio recording started', 'success');
            };
            
            document.getElementById('stop-audio').onclick = function() {
                sendCommand('stop_audio');
                this.disabled = true;
                document.getElementById('start-audio').disabled = false;
                addLog('Audio recording stopped', 'info');
            };
            
            document.getElementById('start-video').onclick = function() {
                sendCommand('start_video');
                this.disabled = true;
                document.getElementById('stop-video').disabled = false;
                addLog('Video recording started', 'success');
            };
            
            document.getElementById('stop-video').onclick = function() {
                sendCommand('stop_video');
                this.disabled = true;
                document.getElementById('start-video').disabled = false;
                addLog('Video recording stopped', 'info');
            };
            
            document.getElementById('init-camera').onclick = function() {
                sendCommand('init_camera');
                addLog('Initializing camera...', 'info');
            };
            
            document.getElementById('open-audio').onclick = function() {
                sendCommand('open_audio_folder');
            };
            
            document.getElementById('open-video').onclick = function() {
                sendCommand('open_video_folder');
            };
            
            document.getElementById('clear-files').onclick = function() {
                sendCommand('clear_old_files');
            };
            
            document.getElementById('system-info').onclick = function() {
                sendCommand('get_system_info');
            };
            
            document.getElementById('clear-log').onclick = function() {
                document.getElementById('log-container').innerHTML = '';
                addLog('Log cleared', 'info');
            };
            
            document.getElementById('save-log').onclick = function() {
                sendCommand('save_log');
            };
            
            // Initialize on page load
            window.onload = function() {
                initWebSocket();
                addLog('Smart Glasses Ultra Light interface loaded', 'success');
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(0.1)
            
            # Send status update
            status = {
                'dock': True,
                'glasses': False,
                'audio': config.system_status.AUDIO_RECORDING,
                'video': config.system_status.VIDEO_RECORDING
            }
            
            await websocket.send_json({
                'type': 'status_update',
                'status': status
            })
            
            # Send audio level
            audio_level = audio_processor.get_audio_level()
            await websocket.send_json({
                'type': 'audio_level',
                'level': audio_level
            })
            
            # Send camera frame
            frame = camera_processor.get_current_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode()
                await websocket.send_json({
                    'type': 'camera_frame',
                    'frame': frame_data
                })
            
            # Check for incoming messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                message = json.loads(data)
                await handle_command(websocket, message)
            except asyncio.TimeoutError:
                continue
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def handle_command(websocket: WebSocket, message: dict):
    """Handle incoming commands"""
    command = message.get('type')
    
    if command == 'start_audio':
        if audio_processor.start_recording():
            await websocket.send_json({
                'type': 'log_message',
                'message': 'Audio recording started successfully',
                'level': 'success'
            })
        else:
            await websocket.send_json({
                'type': 'log_message',
                'message': 'Failed to start audio recording',
                'level': 'error'
            })
    
    elif command == 'stop_audio':
        filename = audio_processor.stop_recording()
        if filename:
            await websocket.send_json({
                'type': 'log_message',
                'message': f'Audio saved: {filename}',
                'level': 'success'
            })
        else:
            await websocket.send_json({
                'type': 'log_message',
                'message': 'Audio recording stopped',
                'level': 'info'
            })
    
    elif command == 'start_video':
        if camera_processor.start_recording():
            await websocket.send_json({
                'type': 'log_message',
                'message': 'Video recording started successfully',
                'level': 'success'
            })
        else:
            await websocket.send_json({
                'type': 'log_message',
                'message': 'Failed to start video recording',
                'level': 'error'
            })
    
    elif command == 'stop_video':
        filename = camera_processor.stop_recording()
        if filename:
            await websocket.send_json({
                'type': 'log_message',
                'message': f'Video saved: {filename}',
                'level': 'success'
            })
        else:
            await websocket.send_json({
                'type': 'log_message',
                'message': 'Video recording stopped',
                'level': 'info'
            })
    
    elif command == 'init_camera':
        if camera_processor.initialize_camera():
            await websocket.send_json({
                'type': 'log_message',
                'message': 'Camera initialized successfully',
                'level': 'success'
            })
        else:
            await websocket.send_json({
                'type': 'log_message',
                'message': 'Failed to initialize camera',
                'level': 'error'
            })
    
    elif command == 'get_system_info':
        audio_files = len(list(config.AUDIO_DIR.glob('*.wav')))
        video_files = len(list(config.VIDEO_DIR.glob('*.mp4')))
        
        await websocket.send_json({
            'type': 'file_info',
            'info': {
                'audio_files': audio_files,
                'video_files': video_files
            }
        })

if __name__ == "__main__":
    print("🚀 Starting Smart Glasses Ultra Light GUI...")
    print("📱 Open: http://localhost:8002")
    print("⚡ Ultra lightweight web interface")
    
    uvicorn.run(app, host="127.0.0.1", port=8002)
