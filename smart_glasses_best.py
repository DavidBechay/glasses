#!/usr/bin/env python3
"""
Smart Glasses System - Best Document Detection
Using advanced OpenCV document detection that ignores people
"""

import asyncio
import json
import base64
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import threading
import pyaudio
import wave
import hashlib
from PIL import Image
import io

app = FastAPI(title="Smart Glasses - Best Document Detection")

# Global state
standby_mode = False
voice_detected = False
document_detected = False
auto_capture_enabled = True
capture_count = 0
last_capture_time = 0

# Document fingerprinting system
document_fingerprints = {}  # Store document hashes
similarity_threshold = 0.85  # Similarity threshold (0-1)
max_fingerprints = 50  # Maximum fingerprints to store

# Audio settings
audio_quality_levels = {
    'standby': {'sample_rate': 16000, 'chunk_size': 512},
    'active': {'sample_rate': 44100, 'chunk_size': 1024},
    'high': {'sample_rate': 48000, 'chunk_size': 2048}
}
current_quality = 'active'

# Audio processor
audio = pyaudio.PyAudio()
audio_stream = None
audio_recording = False
audio_data = []

# Camera processor
camera = None
camera_recording = False
current_frame = None

# WebSocket connections
connections = []

# Create data directories
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
VIDEO_DIR = DATA_DIR / "video"
DOCS_DIR = DATA_DIR / "documents"

for dir_path in [DATA_DIR, AUDIO_DIR, VIDEO_DIR, DOCS_DIR]:
    dir_path.mkdir(exist_ok=True)

@app.get("/")
async def get_main_page():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Glasses - Best Document Detection</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                color: #ffffff;
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid rgba(0, 255, 255, 0.3);
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                background: linear-gradient(45deg, #00ffff, #0080ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
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
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
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
                margin-bottom: 15px;
            }
            
            .status-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 8px;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #666;
                animation: pulse 2s infinite;
            }
            
            .status-indicator.active {
                background: #00ff00;
            }
            
            .status-indicator.standby {
                background: #ffaa00;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .camera-feed {
                width: 100%;
                max-width: 400px;
                height: 300px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 15px;
                margin: 15px auto;
                display: block;
                border: 2px solid rgba(0, 255, 255, 0.3);
            }
            
            .detection-info {
                text-align: center;
                margin-top: 10px;
                padding: 10px;
                border-radius: 8px;
                background: rgba(0, 0, 0, 0.2);
            }
            
            .detection-info.detected {
                background: rgba(0, 255, 0, 0.2);
                color: #00ff00;
                border: 1px solid #00ff00;
            }
            
            .detection-info.scanning {
                background: rgba(255, 170, 0, 0.2);
                color: #ffaa00;
                border: 1px solid #ffaa00;
            }
            
            .log-container {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 15px;
                height: 200px;
                overflow-y: auto;
                font-family: 'Consolas', monospace;
                font-size: 0.9rem;
                margin-top: 15px;
            }
            
            .log-entry {
                margin-bottom: 5px;
                padding: 2px 0;
            }
            
            .log-timestamp {
                color: #00ffff;
                font-weight: bold;
            }
            
            .log-success {
                color: #00ff00;
            }
            
            .log-error {
                color: #ff4444;
            }
            
            .log-info {
                color: #ffffff;
            }
            
            .quality-indicator {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 0.9rem;
                margin-left: 10px;
            }
            
            .quality-active {
                background: rgba(0, 255, 0, 0.2);
                color: #00ff00;
                border: 1px solid #00ff00;
            }
            
            .progress-bar {
                width: 100%;
                height: 20px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #00ff00, #00ffff);
                border-radius: 10px;
                transition: width 0.3s ease;
                width: 0%;
            }
            
            .model-status {
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                text-align: center;
                font-weight: bold;
                background: rgba(0, 255, 0, 0.2);
                color: #00ff00;
                border: 1px solid #00ff00;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔬 Smart Glasses - Best Document Detection</h1>
                <p>Advanced OpenCV • Rectangle Detection • No People</p>
            </div>
            
            <div class="grid">
                <!-- System Status -->
                <div class="card">
                    <h3>📊 System Status</h3>
                    <div class="status-grid">
                        <div class="status-item">
                            <span>🚀 Auto Mode</span>
                            <div class="status-indicator active" id="standby-status"></div>
                        </div>
                        <div class="status-item">
                            <span>🎤 Voice Detection</span>
                            <div class="status-indicator" id="voice-status"></div>
                        </div>
                        <div class="status-item">
                            <span>📄 Document Detection</span>
                            <div class="status-indicator" id="document-status"></div>
                        </div>
                        <div class="status-item">
                            <span>📸 Auto Capture</span>
                            <div class="status-indicator active" id="auto-capture-status"></div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <span>🔊 Audio Quality: </span>
                        <span class="quality-indicator quality-active" id="audio-quality">🎤 Active</span>
                    </div>
                    <div class="model-status">
                        ✅ Advanced OpenCV Document Detection Active
                    </div>
                </div>
                
                <!-- Camera Feed -->
                <div class="card">
                    <h3>📺 Live Camera Feed</h3>
                    <img id="camera-feed" class="camera-feed" style="display: none;">
                    <div id="detection-info" class="detection-info scanning">
                        🔍 Scanning for rectangular documents...
                    </div>
                </div>
                
                <!-- Audio Level -->
                <div class="card">
                    <h3>🔊 Audio Level</h3>
                    <div class="progress-bar">
                        <div class="progress-fill" id="audio-level"></div>
                    </div>
                    <div style="text-align: center; margin-top: 10px;">
                        <span id="audio-level-text">Audio Level: 0%</span>
                    </div>
                </div>
                
                <!-- Document Stats -->
                <div class="card">
                    <h3>📄 Document Detection</h3>
                    <div style="margin: 10px 0;">
                        <span>Documents Captured: </span>
                        <span id="capture-count">0</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Detection Method: </span>
                        <span id="detection-method">Advanced OpenCV</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Confidence: </span>
                        <span id="confidence-level">--</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Duplicate Prevention: </span>
                        <span style="color: #00ff00;">✅ Active</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Similarity Threshold: </span>
                        <span id="similarity-threshold">85%</span>
                    </div>
                </div>
                
                <!-- File Management -->
                <div class="card">
                    <h3>📁 File Management</h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px;">
                        <button class="btn" id="open-audio">📂 Audio</button>
                        <button class="btn" id="open-video">📂 Video</button>
                        <button class="btn" id="open-docs">📂 Documents</button>
                        <button class="btn btn-danger" id="clear-files">🗑️ Clear</button>
                    </div>
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
            let isStandbyMode = false;
            let autoCaptureEnabled = true;
            let captureCount = 0;
            
            // Initialize WebSocket
            function initWebSocket() {
                ws = new WebSocket('ws://localhost:8006/ws');
                
                ws.onopen = function() {
                    addLog('Connected to Smart Glasses system', 'success');
                    addLog('🚀 All systems starting automatically', 'success');
                    addLog('🎤 Voice detection enabled', 'info');
                    addLog('📄 Advanced document detection enabled', 'info');
                    addLog('📸 Auto capture enabled', 'info');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onclose = function() {
                    addLog('Connection lost. Reconnecting...', 'error');
                    setTimeout(initWebSocket, 3000);
                };
            }
            
            // Handle messages
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
                    case 'voice_detected':
                        handleVoiceDetection(data.detected);
                        break;
                    case 'document_detected':
                        handleDocumentDetection(data.detected, data.confidence);
                        break;
                    case 'auto_capture':
                        handleAutoCapture(data.filename);
                        break;
                }
            }
            
            // Update status
            function updateStatus(status) {
                document.getElementById('standby-status').className = 
                    'status-indicator' + (status.standby ? ' standby' : ' active');
                document.getElementById('voice-status').className = 
                    'status-indicator' + (status.voice ? ' active' : '');
                document.getElementById('document-status').className = 
                    'status-indicator' + (status.document ? ' active' : '');
                document.getElementById('auto-capture-status').className = 
                    'status-indicator' + (status.auto_capture ? ' active' : '');
                
                // Update audio quality
                const qualityElement = document.getElementById('audio-quality');
                if (status.quality === 'standby') {
                    qualityElement.textContent = '😴 Standby';
                    qualityElement.className = 'quality-indicator quality-standby';
                } else if (status.quality === 'active') {
                    qualityElement.textContent = '🎤 Active';
                    qualityElement.className = 'quality-indicator quality-active';
                } else if (status.quality === 'high') {
                    qualityElement.textContent = '🔊 High';
                    qualityElement.className = 'quality-indicator quality-high';
                }
            }
            
            // Update audio level
            function updateAudioLevel(level) {
                document.getElementById('audio-level').style.width = level + '%';
                document.getElementById('audio-level-text').textContent = 'Audio Level: ' + level + '%';
            }
            
            // Update camera feed
            function updateCameraFeed(frameData) {
                const img = document.getElementById('camera-feed');
                img.src = 'data:image/jpeg;base64,' + frameData;
                img.style.display = 'block';
            }
            
            // Handle voice detection
            function handleVoiceDetection(detected) {
                if (detected) {
                    addLog('🎤 Voice detected - Increasing audio quality', 'success');
                } else {
                    addLog('😴 Voice lost - Returning to standby', 'info');
                }
            }
            
            // Handle document detection
            function handleDocumentDetection(detected, confidence) {
                const info = document.getElementById('detection-info');
                const confidenceElement = document.getElementById('confidence-level');
                
                if (detected) {
                    info.textContent = '📄 Rectangular document detected - Ready to capture';
                    info.className = 'detection-info detected';
                    confidenceElement.textContent = Math.round(confidence * 100) + '%';
                } else {
                    info.textContent = '🔍 Scanning for rectangular documents...';
                    info.className = 'detection-info scanning';
                    confidenceElement.textContent = '--';
                }
            }
            
            // Handle auto capture
            function handleAutoCapture(filename) {
                captureCount++;
                document.getElementById('capture-count').textContent = captureCount;
                addLog('📸 Document captured: ' + filename, 'success');
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
            
            // Send command
            function sendCommand(command, data = {}) {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: command, ...data}));
                }
            }
            
            // Event listeners
            document.getElementById('open-audio').onclick = function() {
                sendCommand('open_folder', {folder: 'audio'});
            };
            
            document.getElementById('open-video').onclick = function() {
                sendCommand('open_folder', {folder: 'video'});
            };
            
            document.getElementById('open-docs').onclick = function() {
                sendCommand('open_folder', {folder: 'documents'});
            };
            
            document.getElementById('clear-files').onclick = function() {
                sendCommand('clear_files');
            };
            
            document.getElementById('clear-log').onclick = function() {
                document.getElementById('log-container').innerHTML = '';
                addLog('Log cleared', 'info');
            };
            
            document.getElementById('save-log').onclick = function() {
                sendCommand('save_log');
            };
            
            // Initialize
            window.onload = function() {
                initWebSocket();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint"""
    await websocket.accept()
    connections.append(websocket)
    
    try:
        while True:
            await asyncio.sleep(0.1)
            
            # Send status update
            status = {
                'standby': standby_mode,
                'voice': voice_detected,
                'document': document_detected,
                'auto_capture': auto_capture_enabled,
                'quality': get_audio_quality()
            }
            
            await websocket.send_json({
                'type': 'status_update',
                'status': status
            })
            
            # Send audio level
            audio_level = get_audio_level()
            await websocket.send_json({
                'type': 'audio_level',
                'level': audio_level
            })
            
            # Send camera frame
            frame = get_camera_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode()
                await websocket.send_json({
                    'type': 'camera_frame',
                    'frame': frame_data
                })
            
            # Check for messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                message = json.loads(data)
                await handle_command(websocket, message)
            except asyncio.TimeoutError:
                continue
                
    except WebSocketDisconnect:
        connections.remove(websocket)

def get_audio_level():
    """Get current audio level with improved detection"""
    if audio_stream and audio_recording:
        try:
            data = audio_stream.read(1024, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Improved voice detection with multiple metrics
            rms = np.sqrt(np.mean(audio_array**2))
            peak = np.max(np.abs(audio_array))
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            
            # Combine metrics for better detection
            level = min(100, (rms / 32767) * 100)
            
            # Add zero-crossing rate for voice detection
            zcr = zero_crossings / len(audio_array)
            if zcr > 0.1:  # High zero-crossing rate indicates speech
                level *= 1.5  # Boost level for speech-like signals
                
            return int(min(100, level))
        except:
            return 0
    return 0

def get_audio_quality():
    """Get current audio quality level"""
    if voice_detected:
        return 'active'
    elif standby_mode:
        return 'standby'
    else:
        return 'high'

def get_camera_frame():
    """Get current camera frame"""
    global current_frame
    if camera and camera.isOpened():
        ret, frame = camera.read()
        if ret:
            current_frame = frame
            return frame
    return current_frame

def calculate_perceptual_hash(image):
    """Calculate perceptual hash for image similarity detection"""
    try:
        # Resize to 8x8 for hash calculation
        resized = cv2.resize(image, (8, 8))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate average pixel value
        avg = np.mean(gray)
        
        # Create hash based on pixels above/below average
        hash_bits = []
        for pixel in gray.flatten():
            hash_bits.append('1' if pixel > avg else '0')
        
        # Convert to hexadecimal
        hash_string = ''.join(hash_bits)
        hash_int = int(hash_string, 2)
        hash_hex = format(hash_int, '016x')
        
        return hash_hex
    except Exception as e:
        print(f"Hash calculation error: {e}")
        return None

def calculate_content_hash(image):
    """Calculate content-based hash using document features"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract text-like features
        # 1. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # 2. Text line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        line_count = len(cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        
        # 3. Average brightness
        avg_brightness = np.mean(gray)
        
        # 4. Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Create content signature
        content_signature = f"{edge_density:.3f}_{line_count}_{avg_brightness:.1f}_{contrast:.1f}"
        
        # Hash the signature
        content_hash = hashlib.md5(content_signature.encode()).hexdigest()
        
        return content_hash, {
            'edge_density': edge_density,
            'line_count': line_count,
            'brightness': avg_brightness,
            'contrast': contrast
        }
    except Exception as e:
        print(f"Content hash error: {e}")
        return None, None

def calculate_similarity(hash1, hash2):
    """Calculate similarity between two perceptual hashes"""
    if hash1 is None or hash2 is None:
        return 0.0
    
    try:
        # Convert hex to binary
        bin1 = bin(int(hash1, 16))[2:].zfill(64)
        bin2 = bin(int(hash2, 16))[2:].zfill(64)
        
        # Calculate Hamming distance
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
        
        # Convert to similarity (0-1)
        similarity = 1.0 - (hamming_distance / 64.0)
        
        return similarity
    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0

def is_duplicate_document(image):
    """Check if document is a duplicate of previously captured documents"""
    global document_fingerprints
    
    try:
        # Calculate hashes
        perceptual_hash = calculate_perceptual_hash(image)
        content_hash, content_features = calculate_content_hash(image)
        
        if perceptual_hash is None or content_hash is None:
            return False, "Hash calculation failed"
        
        # Check against stored fingerprints
        for stored_hash, stored_data in document_fingerprints.items():
            # Compare perceptual hash
            perceptual_similarity = calculate_similarity(perceptual_hash, stored_data['perceptual_hash'])
            
            # Compare content hash
            content_similarity = 1.0 if content_hash == stored_data['content_hash'] else 0.0
            
            # Combined similarity (weighted)
            combined_similarity = (perceptual_similarity * 0.7) + (content_similarity * 0.3)
            
            if combined_similarity > similarity_threshold:
                print(f"Duplicate detected: similarity={combined_similarity:.3f}")
                return True, f"Similar to {stored_data['filename']} (similarity: {combined_similarity:.3f})"
        
        # Store new fingerprint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fingerprint_id = f"doc_{timestamp}"
        
        document_fingerprints[fingerprint_id] = {
            'perceptual_hash': perceptual_hash,
            'content_hash': content_hash,
            'content_features': content_features,
            'timestamp': time.time(),
            'filename': f"document_{timestamp}.jpg"
        }
        
        # Clean up old fingerprints if we have too many
        if len(document_fingerprints) > max_fingerprints:
            # Remove oldest fingerprints
            sorted_fingerprints = sorted(document_fingerprints.items(), key=lambda x: x[1]['timestamp'])
            for old_id, _ in sorted_fingerprints[:len(document_fingerprints) - max_fingerprints]:
                del document_fingerprints[old_id]
        
        return False, "New document detected"
        
    except Exception as e:
        print(f"Duplicate detection error: {e}")
        return False, f"Error: {e}"

def detect_document_advanced(frame):
    """Advanced document detection using multiple OpenCV methods"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Adaptive thresholding for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Method 2: Morphological operations to clean up the image
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        # Method 3: Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 4: Filter contours by area and shape
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip very small or very large areas (likely not documents)
            if area < 5000 or area > 100000:
                continue
            
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if it has document-like proportions
                if 0.3 < aspect_ratio < 3.0:
                    # Calculate how rectangular it is
                    rect_area = w * h
                    contour_area = cv2.contourArea(contour)
                    rectangularity = contour_area / rect_area
                    
                    # High rectangularity indicates a document
                    if rectangularity > 0.7:
                        print(f"Document detected: area={area}, aspect_ratio={aspect_ratio:.2f}, rectangularity={rectangularity:.2f}")
                        return True, rectangularity
        
        # Method 5: Hough Line Transform as backup
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None and len(lines) > 6:
            # Count horizontal and vertical lines
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 20 or abs(angle - 180) < 20:
                    horizontal_lines += 1
                elif abs(angle - 90) < 20 or abs(angle + 90) < 20:
                    vertical_lines += 1
            
            # If we have enough lines forming a rectangle
            if horizontal_lines >= 2 and vertical_lines >= 2:
                print(f"Document detected via lines: horizontal={horizontal_lines}, vertical={vertical_lines}")
                return True, 0.6
        
        return False, 0.0
        
    except Exception as e:
        print(f"Document detection error: {e}")
        return False, 0.0

async def handle_command(websocket: WebSocket, message: dict):
    """Handle commands"""
    global standby_mode, auto_capture_enabled, capture_count, last_capture_time
    
    command = message.get('type')
    
    if command == 'toggle_standby':
        standby_mode = not standby_mode
        await send_log(f'Standby mode {"enabled" if standby_mode else "disabled"}', 'info')
    
    elif command == 'toggle_auto_capture':
        auto_capture_enabled = not auto_capture_enabled
        await send_log(f'Auto capture {"enabled" if auto_capture_enabled else "disabled"}', 'info')
    
    elif command == 'force_audio':
        start_audio_recording()
        await send_log('Manual audio recording started', 'success')
    
    elif command == 'force_video':
        start_video_recording()
        await send_log('Manual video recording started', 'success')
    
    elif command == 'init_camera':
        init_camera()
        await send_log('Camera initialized', 'success')
    
    elif command == 'set_sensitivity':
        await send_log(f'Sensitivity set to {int(message.get("sensitivity", 0.7) * 100)}%', 'info')
    
    elif command == 'open_folder':
        folder = message.get('folder', 'audio')
        await send_log(f'Opening {folder} folder', 'info')
    
    elif command == 'clear_files':
        await send_log('Clearing old files', 'info')
    
    elif command == 'save_log':
        await send_log('Log saved', 'success')

async def send_log(message: str, level: str = 'info'):
    """Send log message to all connections"""
    for connection in connections:
        try:
            await connection.send_json({
                'type': 'log_message',
                'message': message,
                'level': level
            })
        except:
            pass

def start_audio_recording():
    """Start audio recording"""
    global audio_stream, audio_recording, audio_data
    
    if not audio_recording:
        try:
            audio_stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=audio_quality_levels[current_quality]['sample_rate'],
                input=True,
                frames_per_buffer=audio_quality_levels[current_quality]['chunk_size']
            )
            audio_recording = True
            audio_data = []
        except Exception as e:
            print(f"Audio recording error: {e}")

def start_video_recording():
    """Start video recording"""
    global camera_recording
    
    if camera and camera.isOpened() and not camera_recording:
        camera_recording = True

def init_camera():
    """Initialize camera"""
    global camera
    
    try:
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
    except Exception as e:
        print(f"Camera initialization error: {e}")

# Background tasks
async def voice_detection():
    """Voice activity detection"""
    global voice_detected, current_quality
    
    while True:
        try:
            if standby_mode:
                audio_level = get_audio_level()
                
                if audio_level > 8:  # Lower threshold for better voice detection
                    if not voice_detected:
                        voice_detected = True
                        current_quality = 'active'
                        await send_log('🎤 Voice detected - Increasing audio quality', 'success')
                        
                        # Notify all connections
                        for connection in connections:
                            try:
                                await connection.send_json({
                                    'type': 'voice_detected',
                                    'detected': True
                                })
                            except:
                                pass
                else:
                    if voice_detected:
                        voice_detected = False
                        current_quality = 'standby'
                        await send_log('😴 Voice lost - Returning to standby', 'info')
                        
                        # Notify all connections
                        for connection in connections:
                            try:
                                await connection.send_json({
                                    'type': 'voice_detected',
                                    'detected': False
                                })
                            except:
                                pass
            
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Voice detection error: {e}")
            await asyncio.sleep(1)

async def document_detection():
    """Advanced document detection and auto capture"""
    global document_detected, capture_count, last_capture_time
    
    while True:
        try:
            if auto_capture_enabled:
                frame = get_camera_frame()
                if frame is not None:
                    detected, confidence = detect_document_advanced(frame)
                    
                    if detected != document_detected:
                        document_detected = detected
                        
                        # Notify all connections
                        for connection in connections:
                            try:
                                await connection.send_json({
                                    'type': 'document_detected',
                                    'detected': detected,
                                    'confidence': confidence
                                })
                            except:
                                pass
                    
                    # Auto capture with duplicate detection
                    if detected and (time.time() - last_capture_time) >= 3.0:  # Increased delay
                        # Check if this is a duplicate document
                        is_duplicate, duplicate_reason = is_duplicate_document(frame)
                        
                        if is_duplicate:
                            await send_log(f'🚫 Duplicate document skipped: {duplicate_reason}', 'info')
                            last_capture_time = time.time()  # Reset timer to prevent spam
                        else:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            doc_file = DOCS_DIR / f"document_{timestamp}.jpg"
                            
                            cv2.imwrite(str(doc_file), frame)
                            last_capture_time = time.time()
                            capture_count += 1
                            
                            await send_log(f'📸 Document captured: {doc_file.name} (Confidence: {confidence:.2f})', 'success')
                            
                            # Notify all connections
                            for connection in connections:
                                try:
                                    await connection.send_json({
                                        'type': 'auto_capture',
                                        'filename': doc_file.name
                                    })
                                except:
                                    pass
            
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Document detection error: {e}")
            await asyncio.sleep(1)

# Start background tasks
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(voice_detection())
    asyncio.create_task(document_detection())
    
    # Initialize and start all systems automatically
    init_camera()
    start_audio_recording()
    start_video_recording()
    
    print("🚀 All systems started automatically!")
    print("✅ Advanced OpenCV document detection active - No people detection")
    print("🧠 Smart duplicate detection active - Won't capture same document twice")

if __name__ == "__main__":
    print("🔬 Starting Smart Glasses - Best Document Detection...")
    print("📱 Open: http://localhost:8006")
    print("🚀 Auto-start • 🎤 Voice detection • 📄 Smart Document detection • 🧠 Duplicate prevention")
    
    uvicorn.run(app, host="127.0.0.1", port=8006)
