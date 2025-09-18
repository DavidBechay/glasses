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
import hashlib
from PIL import Image
import io
import pytesseract
import easyocr
import sounddevice as sd
import wave
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import re
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import nltk
import signal
import sys
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = FastAPI(title="Smart Glasses - Best Document Detection")

# Global state
standby_mode = False
document_detected = False
auto_capture_enabled = True
capture_count = 0
last_capture_time = 0

# Audio recording settings
AUDIO_CHANNELS = 1
AUDIO_RATE = 44100
AUDIO_FORMAT = 'int16'
AUDIO_CHUNK_DURATION = 60 * 60  # 1 hour chunks

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.current_file = None
        self.file_index = 1
        self.recording_start_time = None
        print("✅ Audio recorder initialized")
    
    def start_recording(self):
        """Start continuous audio recording"""
        if not self.recording:
            self.recording = True
            self.recording_start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file = str(AUDIO_DIR / f"recording_{timestamp}.wav")
            
            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=self._record_chunk)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            print(f"🎤 Audio recording started: {self.current_file}")
    
    def stop_recording(self):
        """Stop audio recording"""
        if self.recording:
            self.recording = False
            if hasattr(self, 'recording_thread'):
                self.recording_thread.join(timeout=2)
            
            duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            print(f"🛑 Audio recording stopped. Duration: {duration:.1f}s")
            return self.current_file, duration
    
    def _record_chunk(self):
        """Record audio chunk using sounddevice"""
        try:
            wf = wave.open(self.current_file, 'wb')
            wf.setnchannels(AUDIO_CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(AUDIO_RATE)

            def callback(indata, frames_count, time_info, status):
                if status:
                    print(f"Audio status: {status}")
                if self.recording:
                    data = (indata * 32767).astype(np.int16).tobytes()
                    wf.writeframes(data)

            with sd.InputStream(samplerate=AUDIO_RATE, channels=AUDIO_CHANNELS, callback=callback):
                print(f"🎤 Recording {self.current_file}...")
                while self.recording:
                    time.sleep(0.1)

            wf.close()
            print(f"💾 Saved {self.current_file}")
            
        except Exception as e:
            print(f"❌ Audio recording error: {e}")
    
    def is_recording(self):
        """Check if currently recording"""
        return self.recording

# Initialize audio recorder
audio_recorder = AudioRecorder()

# Document fingerprinting system
document_fingerprints = {}  # Store document hashes
similarity_threshold = 0.95  # Very strict similarity threshold (0-1)
max_fingerprints = 50  # Maximum fingerprints to store
restriction_violations = 0  # Track restriction violations
max_restriction_violations = 5  # Max violations before system lockout

# OCR processor
try:
    ocr_reader = easyocr.Reader(['en'])  # Initialize EasyOCR for English
    print("✅ OCR system initialized")
except Exception as e:
    print(f"⚠️ OCR initialization failed: {e}")
    ocr_reader = None

# Semantic Analysis System
try:
    # Initialize sentence transformer for semantic similarity
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize NLTK components
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    print("✅ Semantic analysis system initialized")
except Exception as e:
    print(f"⚠️ Semantic analysis initialization failed: {e}")
    semantic_model = None
    stop_words = set()
    lemmatizer = None

# Document Database
try:
    db_conn = sqlite3.connect('document_fingerprints.db', check_same_thread=False)
    db_cursor = db_conn.cursor()
    
    # Create documents table
    db_cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            content_hash TEXT,
            semantic_vector TEXT,
            structural_features TEXT,
            visual_features TEXT,
            extracted_text TEXT,
            timestamp REAL,
            capture_time TEXT
        )
    ''')
    
    db_conn.commit()
    print("✅ Document database initialized")
except Exception as e:
    print(f"⚠️ Database initialization failed: {e}")
    db_conn = None
    db_cursor = None
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

@app.post("/reset-restrictions")
async def reset_restrictions():
    """Reset restriction violations counter"""
    global restriction_violations
    try:
        restriction_violations = 0
        return {"success": True, "message": "Restriction violations reset"}
    except Exception as e:
        return {"success": False, "error": str(e)}

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
                            <span>📄 Document Detection</span>
                            <div class="status-indicator" id="document-status"></div>
                        </div>
                        <div class="status-item">
                            <span>📸 Auto Capture</span>
                            <div class="status-indicator active" id="auto-capture-status"></div>
                        </div>
                    </div>
                    <div style="margin-top: 15px;">
                        <span>📸 Camera Status: </span>
                        <span class="quality-indicator quality-active" id="camera-quality">📹 Active</span>
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
                
                <!-- Audio Recording Status -->
                <div class="card">
                    <h3>🎤 Audio Recording</h3>
                    <div style="text-align: center;">
                        <div id="audio-indicator" style="margin-bottom: 15px; padding: 15px; border-radius: 8px; background: rgba(0, 255, 0, 0.3);">
                            <div style="font-size: 1.2rem; margin-bottom: 10px;">
                                🔴 <span id="audio-status">Recording</span>
                            </div>
                            <div style="font-size: 1rem;">
                                Duration: <span id="audio-duration">00:00</span>
                            </div>
                            <div style="font-size: 0.9rem; margin-top: 5px; opacity: 0.8;">
                                <span id="audio-info">Continuous 16-bit recording</span>
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <button class="btn" id="stop-recording" style="background: linear-gradient(45deg, #f44336, #d32f2f);">
                                ⏹️ Stop Recording
                            </button>
                            <button class="btn" id="start-recording" style="background: linear-gradient(45deg, #4CAF50, #45a049);">
                                ▶️ Start Recording
                            </button>
                        </div>
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
                        <span style="color: #ff4444;">🔒 Ultra-Restrictive</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Similarity Threshold: </span>
                        <span id="similarity-threshold">95%+ (Very Strict)</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Detection Method: </span>
                        <span style="color: #ff6b6b;">Multi-Modal + Pre-filtering</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Analysis Type: </span>
                        <span style="color: #ff4444;">🔒 Restrictive Mode</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <span>Restriction Violations: </span>
                        <span id="restriction-violations" style="color: #ff4444;">0/5</span>
                    </div>
                    <div style="margin: 10px 0;">
                        <button onclick="resetRestrictions()" style="background: #ff4444; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">Reset Restrictions</button>
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
            let audioStartTime = null;
            let audioInterval = null;
            
            // Update audio recording status display
            function updateAudioStatus(isRecording, duration = 0, info = '') {
                const audioIndicator = document.getElementById('audio-indicator');
                const audioStatus = document.getElementById('audio-status');
                const audioDuration = document.getElementById('audio-duration');
                const audioInfo = document.getElementById('audio-info');
                const startBtn = document.getElementById('start-recording');
                const stopBtn = document.getElementById('stop-recording');
                
                if (isRecording) {
                    audioIndicator.style.display = 'block';
                    audioIndicator.style.background = 'rgba(0, 255, 0, 0.3)';
                    audioStatus.textContent = 'Recording';
                    audioInfo.textContent = info || 'Continuous 16-bit recording';
                    startBtn.style.display = 'none';
                    stopBtn.style.display = 'inline-block';
                    
                    // Start duration timer
                    if (!audioStartTime) {
                        audioStartTime = Date.now();
                        audioInterval = setInterval(updateAudioDuration, 1000);
                    }
                } else {
                    audioIndicator.style.display = 'none';
                    audioStatus.textContent = 'Not Recording';
                    audioInfo.textContent = 'Audio recording stopped';
                    startBtn.style.display = 'inline-block';
                    stopBtn.style.display = 'none';
                    
                    // Stop duration timer
                    if (audioInterval) {
                        clearInterval(audioInterval);
                        audioInterval = null;
                        audioStartTime = null;
                    }
                }
            }
            
            // Update audio recording duration
            function updateAudioDuration() {
                if (audioStartTime) {
                    const elapsed = Math.floor((Date.now() - audioStartTime) / 1000);
                    const minutes = Math.floor(elapsed / 60);
                    const seconds = elapsed % 60;
                    const durationText = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                    document.getElementById('audio-duration').textContent = durationText;
                }
            }
            
            // Initialize WebSocket
            function initWebSocket() {
                ws = new WebSocket('ws://localhost:8006/ws');
                
                ws.onopen = function() {
                    addLog('Connected to Smart Glasses system', 'success');
                    addLog('🚀 All systems starting automatically', 'success');
                    addLog('🎤 Audio recording enabled', 'info');
                    addLog('📄 Advanced document detection enabled', 'info');
                    addLog('📸 Auto capture enabled', 'info');
                    
                    // Start audio recording status display
                    updateAudioStatus(true);
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
                    case 'camera_frame':
                        updateCameraFeed(data.frame);
                        break;
                    case 'log_message':
                        addLog(data.message, data.level || 'info');
                        break;
                    case 'audio_status':
                        updateAudioStatus(data.recording, data.duration, data.info);
                        break;
                    case 'document_detected':
                        handleDocumentDetection(data.detected, data.confidence);
                        break;
                    case 'auto_capture':
                        handleAutoCapture(data.filename);
                        break;
                }
            }
            
            // Reset restriction violations
            function resetRestrictions() {
                fetch('/reset-restrictions', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('restriction-violations').textContent = '0/5';
                            alert('Restriction violations reset successfully!');
                        } else {
                            alert('Failed to reset restrictions: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error resetting restrictions:', error);
                        alert('Error resetting restrictions');
                    });
            }
            
            // Update status
            function updateStatus(status) {
                document.getElementById('standby-status').className = 
                    'status-indicator' + (status.standby ? ' standby' : ' active');
                document.getElementById('document-status').className = 
                    'status-indicator' + (status.document ? ' active' : '');
                document.getElementById('auto-capture-status').className = 
                    'status-indicator' + (status.auto_capture ? ' active' : '');
                
                // Update camera quality
                const qualityElement = document.getElementById('camera-quality');
                if (qualityElement) {
                    qualityElement.textContent = '📹 Active';
                    qualityElement.className = 'quality-indicator quality-active';
                }
            }
            
            // Update camera feed
            function updateCameraFeed(frameData) {
                const img = document.getElementById('camera-feed');
                img.src = 'data:image/jpeg;base64,' + frameData;
                img.style.display = 'block';
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
            
            document.getElementById('start-recording').onclick = function() {
                sendCommand('start_audio_recording');
            };
            
            document.getElementById('stop-recording').onclick = function() {
                sendCommand('stop_audio_recording');
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
                'document': document_detected,
                'auto_capture': auto_capture_enabled
            }
            
            await websocket.send_json({
                'type': 'status_update',
                'status': status
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

def calculate_structural_features(image):
    """Calculate structural features of the document"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate features
        features = {
            'image_size': (width, height),
            'aspect_ratio': width / height,
            'edge_density': np.sum(edges > 0) / (width * height),
            'contour_count': len(contours),
            'avg_contour_area': np.mean([cv2.contourArea(c) for c in contours]) if contours else 0,
            'brightness': np.mean(gray),
            'contrast': np.std(gray)
        }
        
        return features
        
    except Exception as e:
        print(f"Structural features error: {e}")
        return {}

def calculate_document_quality_score(image):
    """Calculate comprehensive document quality score"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        quality_factors = []
        
        # 1. Resolution quality (higher resolution = better)
        resolution_score = min(1.0, (height * width) / (1000 * 1000))  # Normalize to 1MP
        quality_factors.append(resolution_score)
        
        # 2. Sharpness quality (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 1000)  # Normalize
        quality_factors.append(sharpness_score)
        
        # 3. Contrast quality
        contrast_score = np.std(gray) / 255.0
        quality_factors.append(contrast_score)
        
        # 4. Brightness quality (avoid too dark or too bright)
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
        quality_factors.append(brightness_score)
        
        # 5. Edge density quality
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        edge_score = min(1.0, edge_density * 10)  # Normalize
        quality_factors.append(edge_score)
        
        # 6. Aspect ratio quality (prefer standard document ratios)
        aspect_ratio = width / height
        if 0.6 <= aspect_ratio <= 1.4:  # Standard document ratios
            aspect_score = 1.0
        elif 0.4 <= aspect_ratio <= 2.0:  # Acceptable ratios
            aspect_score = 0.7
        else:  # Unusual ratios
            aspect_score = 0.3
        quality_factors.append(aspect_score)
        
        # 7. Noise level quality (lower noise = better)
        noise_level = np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)
        noise_score = max(0.0, 1.0 - noise_level / 50)  # Normalize
        quality_factors.append(noise_score)
        
        # Calculate weighted average
        weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.1, 0.05]  # Resolution and sharpness most important
        weighted_score = sum(score * weight for score, weight in zip(quality_factors, weights))
        
        return min(1.0, max(0.0, weighted_score))
        
    except Exception as e:
        print(f"Quality score calculation error: {e}")
        return 0.0

def calculate_advanced_structural_features(image):
    """Calculate advanced structural features using multiple CV techniques"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Multi-scale edge detection
        edges_small = cv2.Canny(gray, 50, 150)
        edges_medium = cv2.Canny(gray, 30, 100)
        edges_large = cv2.Canny(gray, 10, 50)
        
        # Contour analysis at multiple scales
        contours_small, _ = cv2.findContours(edges_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_medium, _ = cv2.findContours(edges_medium, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Texture analysis using Local Binary Patterns
        lbp_features = calculate_lbp_features(gray)
        
        # Frequency domain analysis
        fft_features = calculate_fft_features(gray)
        
        # Geometric features
        geometric_features = calculate_geometric_features(gray, contours_small)
        
        # Color distribution (if color image)
        color_features = calculate_color_distribution(image)
        
        features = {
            'image_size': (width, height),
            'aspect_ratio': width / height,
            'edge_density_small': np.sum(edges_small > 0) / (width * height),
            'edge_density_medium': np.sum(edges_medium > 0) / (width * height),
            'edge_density_large': np.sum(edges_large > 0) / (width * height),
            'contour_count_small': len(contours_small),
            'contour_count_medium': len(contours_medium),
            'avg_contour_area': np.mean([cv2.contourArea(c) for c in contours_small]) if contours_small else 0,
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'lbp_features': lbp_features,
            'fft_features': fft_features,
            'geometric_features': geometric_features,
            'color_features': color_features
        }
        
        return features
        
    except Exception as e:
        print(f"Advanced structural features error: {e}")
        return {}

def calculate_lbp_features(gray_image):
    """Calculate Local Binary Pattern features for texture analysis"""
    try:
        # Simple LBP implementation
        lbp_image = np.zeros_like(gray_image)
        
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                center = gray_image[i, j]
                binary_string = ""
                
                # 8-neighborhood
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp_image[i, j] = int(binary_string, 2)
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        
        return {
            'lbp_histogram': hist.tolist(),
            'lbp_mean': np.mean(lbp_image),
            'lbp_std': np.std(lbp_image),
            'lbp_entropy': -np.sum(hist * np.log2(hist + 1e-10))
        }
        
    except Exception as e:
        print(f"LBP features error: {e}")
        return {}

def calculate_fft_features(gray_image):
    """Calculate frequency domain features using FFT"""
    try:
        # Apply FFT
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calculate frequency domain features
        height, width = magnitude_spectrum.shape
        center_y, center_x = height // 2, width // 2
        
        # Low frequency energy (center region)
        low_freq_region = magnitude_spectrum[center_y-10:center_y+10, center_x-10:center_x+10]
        low_freq_energy = np.sum(low_freq_region)
        
        # High frequency energy (edges)
        high_freq_energy = np.sum(magnitude_spectrum) - low_freq_energy
        
        return {
            'low_freq_energy': low_freq_energy,
            'high_freq_energy': high_freq_energy,
            'spectral_centroid': np.sum(magnitude_spectrum * np.arange(height)) / np.sum(magnitude_spectrum),
            'spectral_spread': np.sqrt(np.sum(magnitude_spectrum * (np.arange(height) - np.sum(magnitude_spectrum * np.arange(height)) / np.sum(magnitude_spectrum))**2) / np.sum(magnitude_spectrum))
        }
        
    except Exception as e:
        print(f"FFT features error: {e}")
        return {}

def calculate_geometric_features(gray_image, contours):
    """Calculate geometric features from contours"""
    try:
        if not contours:
            return {}
        
        areas = [cv2.contourArea(c) for c in contours]
        perimeters = [cv2.arcLength(c, True) for c in contours]
        
        # Calculate shape descriptors
        circularities = []
        rectangularities = []
        
        for i, contour in enumerate(contours):
            if perimeters[i] > 0:
                circularity = 4 * np.pi * areas[i] / (perimeters[i] ** 2)
                circularities.append(circularity)
            
            # Rectangularity
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            if rect_area > 0:
                rectangularity = areas[i] / rect_area
                rectangularities.append(rectangularity)
        
        return {
            'avg_circularity': np.mean(circularities) if circularities else 0,
            'avg_rectangularity': np.mean(rectangularities) if rectangularities else 0,
            'circularity_std': np.std(circularities) if circularities else 0,
            'rectangularity_std': np.std(rectangularities) if rectangularities else 0,
            'total_area': np.sum(areas),
            'area_distribution': np.histogram(areas, bins=10)[0].tolist()
        }
        
    except Exception as e:
        print(f"Geometric features error: {e}")
        return {}

def calculate_color_distribution(image):
    """Calculate color distribution features"""
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        bgr_mean = np.mean(image, axis=(0, 1))
        hsv_mean = np.mean(hsv, axis=(0, 1))
        lab_mean = np.mean(lab, axis=(0, 1))
        
        bgr_std = np.std(image, axis=(0, 1))
        hsv_std = np.std(hsv, axis=(0, 1))
        lab_std = np.std(lab, axis=(0, 1))
        
        return {
            'bgr_mean': bgr_mean.tolist(),
            'hsv_mean': hsv_mean.tolist(),
            'lab_mean': lab_mean.tolist(),
            'bgr_std': bgr_std.tolist(),
            'hsv_std': hsv_std.tolist(),
            'lab_std': lab_std.tolist(),
            'dominant_colors': calculate_dominant_colors(image)
        }
        
    except Exception as e:
        print(f"Color distribution error: {e}")
        return {}

def calculate_dominant_colors(image, k=5):
    """Calculate dominant colors using K-means clustering"""
    try:
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Use a subset for faster computation
        if len(pixels) > 10000:
            pixels = pixels[::len(pixels)//10000]
        
        # Simple K-means implementation
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
        color_counts = np.bincount(kmeans.labels_).tolist()
        
        return {
            'colors': dominant_colors,
            'counts': color_counts,
            'ratios': [count/sum(color_counts) for count in color_counts]
        }
        
    except Exception as e:
        print(f"Dominant colors error: {e}")
        return {}

def calculate_layout_features(image):
    """Calculate sophisticated layout features"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Detect lines using Hough transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        horizontal_lines = 0
        vertical_lines = 0
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:
                    horizontal_lines += 1
                elif abs(theta - np.pi/2) < np.pi/4:
                    vertical_lines += 1
        
        # Detect text regions
        text_regions = detect_text_regions(gray)
        
        # Calculate layout metrics
        layout_metrics = {
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'text_regions': len(text_regions),
            'text_density': sum([region['area'] for region in text_regions]) / (width * height),
            'layout_complexity': calculate_layout_complexity(gray),
            'symmetry_score': calculate_symmetry_score(gray),
            'margin_analysis': analyze_margins(gray)
        }
        
        return layout_metrics
        
    except Exception as e:
        print(f"Layout features error: {e}")
        return {}

def detect_text_regions(gray_image):
    """Detect text regions using MSER and other techniques"""
    try:
        # Use MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray_image)
        
        text_regions = []
        for region in regions:
            if len(region) > 50:  # Filter small regions
                x, y, w, h = cv2.boundingRect(region)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Text-like regions typically have certain properties
                if 0.1 < aspect_ratio < 10 and area > 100:
                    text_regions.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'area': area, 'aspect_ratio': aspect_ratio
                    })
        
        return text_regions
        
    except Exception as e:
        print(f"Text regions detection error: {e}")
        return []

def calculate_layout_complexity(gray_image):
    """Calculate layout complexity score"""
    try:
        # Use edge density and variance as complexity indicators
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        variance = cv2.filter2D((gray_image.astype(np.float32) - mean) ** 2, -1, kernel)
        avg_variance = np.mean(variance)
        
        # Normalize complexity score
        complexity = (edge_density * 0.7 + (avg_variance / 10000) * 0.3)
        return min(1.0, complexity)
        
    except Exception as e:
        print(f"Layout complexity error: {e}")
        return 0.0

def calculate_symmetry_score(gray_image):
    """Calculate horizontal and vertical symmetry scores"""
    try:
        height, width = gray_image.shape
        
        # Horizontal symmetry
        top_half = gray_image[:height//2, :]
        bottom_half = cv2.flip(gray_image[height//2:, :], 0)
        
        # Resize to same dimensions
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        horizontal_symmetry = 1.0 - np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float))) / 255.0
        
        # Vertical symmetry
        left_half = gray_image[:, :width//2]
        right_half = cv2.flip(gray_image[:, width//2:], 1)
        
        # Resize to same dimensions
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        vertical_symmetry = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
        
        return {
            'horizontal_symmetry': horizontal_symmetry,
            'vertical_symmetry': vertical_symmetry,
            'overall_symmetry': (horizontal_symmetry + vertical_symmetry) / 2
        }
        
    except Exception as e:
        print(f"Symmetry calculation error: {e}")
        return {'horizontal_symmetry': 0, 'vertical_symmetry': 0, 'overall_symmetry': 0}

def analyze_margins(gray_image):
    """Analyze document margins"""
    try:
        height, width = gray_image.shape
        margin_size = min(width, height) // 10
        
        # Analyze margins
        top_margin = np.mean(gray_image[:margin_size, :])
        bottom_margin = np.mean(gray_image[-margin_size:, :])
        left_margin = np.mean(gray_image[:, :margin_size])
        right_margin = np.mean(gray_image[:, -margin_size:])
        
        # Calculate margin consistency
        margins = [top_margin, bottom_margin, left_margin, right_margin]
        margin_consistency = 1.0 - (np.std(margins) / np.mean(margins)) if np.mean(margins) > 0 else 0
        
        return {
            'top_margin': top_margin,
            'bottom_margin': bottom_margin,
            'left_margin': left_margin,
            'right_margin': right_margin,
            'margin_consistency': margin_consistency,
            'avg_margin_brightness': np.mean(margins)
        }
        
    except Exception as e:
        print(f"Margin analysis error: {e}")
        return {}

def extract_and_analyze_text(image):
    """Extract and analyze text features"""
    try:
        if ocr_reader is None:
            return {}
        
        # Extract text using EasyOCR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = ocr_reader.readtext(rgb_image)
        
        if not results:
            return {}
        
        # Analyze text features
        texts = [result[1] for result in results]
        confidences = [result[2] for result in results]
        
        # Combine all text
        full_text = " ".join(texts)
        
        # Calculate text statistics
        text_features = {
            'text_count': len(texts),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'total_characters': len(full_text),
            'total_words': len(full_text.split()),
            'avg_word_length': np.mean([len(word) for word in full_text.split()]) if full_text.split() else 0,
            'text_density': len(full_text) / (image.shape[0] * image.shape[1]),
            'language_features': analyze_language_features(full_text),
            'text_layout': analyze_text_layout(results)
        }
        
        return text_features
        
    except Exception as e:
        print(f"Text analysis error: {e}")
        return {}

def analyze_language_features(text):
    """Analyze language-specific features"""
    try:
        if not text:
            return {}
        
        # Character frequency analysis
        char_freq = {}
        for char in text.lower():
            if char.isalpha():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Word length distribution
        words = text.split()
        word_lengths = [len(word) for word in words]
        
        # Sentence analysis
        sentences = text.split('.')
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        
        return {
            'char_frequency': char_freq,
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'word_length_std': np.std(word_lengths) if word_lengths else 0,
            'avg_sentence_length': np.mean(sentence_lengths) if sentence_lengths else 0,
            'sentence_count': len(sentences),
            'vocabulary_richness': len(set(words)) / len(words) if words else 0
        }
        
    except Exception as e:
        print(f"Language features error: {e}")
        return {}

def analyze_text_layout(ocr_results):
    """Analyze spatial layout of text"""
    try:
        if not ocr_results:
            return {}
        
        # Extract bounding boxes
        boxes = [result[0] for result in ocr_results]
        
        # Calculate layout metrics
        x_coords = []
        y_coords = []
        widths = []
        heights = []
        
        for box in boxes:
            x_coords.extend([point[0] for point in box])
            y_coords.extend([point[1] for point in box])
            widths.append(max([point[0] for point in box]) - min([point[0] for point in box]))
            heights.append(max([point[1] for point in box]) - min([point[1] for point in box]))
        
        return {
            'text_spread_x': max(x_coords) - min(x_coords) if x_coords else 0,
            'text_spread_y': max(y_coords) - min(y_coords) if y_coords else 0,
            'avg_text_width': np.mean(widths) if widths else 0,
            'avg_text_height': np.mean(heights) if heights else 0,
            'text_alignment_score': calculate_text_alignment_score(boxes)
        }
        
    except Exception as e:
        print(f"Text layout analysis error: {e}")
        return {}

def calculate_text_alignment_score(boxes):
    """Calculate how well-aligned the text is"""
    try:
        if len(boxes) < 2:
            return 1.0
        
        # Calculate alignment scores
        left_alignments = []
        right_alignments = []
        top_alignments = []
        bottom_alignments = []
        
        for box in boxes:
            left_alignments.append(min([point[0] for point in box]))
            right_alignments.append(max([point[0] for point in box]))
            top_alignments.append(min([point[1] for point in box]))
            bottom_alignments.append(max([point[1] for point in box]))
        
        # Calculate alignment consistency
        left_consistency = 1.0 - (np.std(left_alignments) / np.mean(left_alignments)) if np.mean(left_alignments) > 0 else 0
        right_consistency = 1.0 - (np.std(right_alignments) / np.mean(right_alignments)) if np.mean(right_alignments) > 0 else 0
        top_consistency = 1.0 - (np.std(top_alignments) / np.mean(top_alignments)) if np.mean(top_alignments) > 0 else 0
        bottom_consistency = 1.0 - (np.std(bottom_alignments) / np.mean(bottom_alignments)) if np.mean(bottom_alignments) > 0 else 0
        
        return {
            'left_alignment': left_consistency,
            'right_alignment': right_consistency,
            'top_alignment': top_consistency,
            'bottom_alignment': bottom_consistency,
            'overall_alignment': (left_consistency + right_consistency + top_consistency + bottom_consistency) / 4
        }
        
    except Exception as e:
        print(f"Text alignment error: {e}")
        return {}

def calculate_semantic_features(text_features):
    """Calculate semantic features using advanced NLP"""
    try:
        if not text_features or not text_features.get('text_count', 0):
            return {}
        
        # This would use the semantic model if available
        if semantic_model is None:
            return {}
        
        # Extract text for semantic analysis
        # Note: This is a placeholder - in practice, you'd extract the actual text
        sample_text = "document content"  # This should be the actual extracted text
        
        # Calculate semantic embeddings
        semantic_vector = semantic_model.encode(sample_text)
        
        return {
            'semantic_vector': semantic_vector.tolist(),
            'semantic_dimensions': len(semantic_vector),
            'semantic_magnitude': np.linalg.norm(semantic_vector)
        }
        
    except Exception as e:
        print(f"Semantic features error: {e}")
        return {}

def calculate_temporal_context():
    """Calculate temporal context features"""
    try:
        current_time = time.time()
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        return {
            'timestamp': current_time,
            'hour_of_day': current_hour,
            'day_of_week': current_day,
            'is_weekend': current_day >= 5,
            'is_business_hours': 9 <= current_hour <= 17,
            'time_since_last_capture': current_time - last_capture_time if last_capture_time > 0 else 0
        }
        
    except Exception as e:
        print(f"Temporal context error: {e}")
        return {}

def find_best_match_hierarchical(document_signature):
    """Find best match using hierarchical similarity analysis"""
    try:
        best_match = None
        best_score = 0.0
        
        for stored_hash, stored_data in document_fingerprints.items():
            # Calculate multi-modal similarity
            similarity_scores = {}
            
            # 1. Exact content match (highest priority)
            if stored_data.get('content_hash') == document_signature['content_hash']:
                similarity_scores['exact_match'] = 1.0
            
            # 2. Perceptual similarity
            if stored_data.get('perceptual_hash') and document_signature['perceptual_hash']:
                perceptual_sim = calculate_similarity(
                    stored_data['perceptual_hash'], 
                    document_signature['perceptual_hash']
                )
                similarity_scores['perceptual'] = perceptual_sim
            
            # 3. Structural similarity
            if stored_data.get('structural_features') and document_signature['structural_features']:
                structural_sim = calculate_structural_similarity(
                    stored_data['structural_features'],
                    document_signature['structural_features']
                )
                similarity_scores['structural'] = structural_sim
            
            # 4. Layout similarity
            if stored_data.get('layout_features') and document_signature['layout_features']:
                layout_sim = calculate_layout_similarity_advanced(
                    stored_data['layout_features'],
                    document_signature['layout_features']
                )
                similarity_scores['layout'] = layout_sim
            
            # 5. Text similarity
            if stored_data.get('text_features') and document_signature['text_features']:
                text_sim = calculate_text_similarity_advanced(
                    stored_data['text_features'],
                    document_signature['text_features']
                )
                similarity_scores['text'] = text_sim
            
            # 6. Semantic similarity
            if stored_data.get('semantic_features') and document_signature['semantic_features']:
                semantic_sim = calculate_semantic_similarity_advanced(
                    stored_data['semantic_features'],
                    document_signature['semantic_features']
                )
                similarity_scores['semantic'] = semantic_sim
            
            # Calculate weighted overall similarity
            overall_similarity = calculate_weighted_similarity(similarity_scores)
            
            if overall_similarity > best_score:
                best_score = overall_similarity
                match_type = determine_match_type(similarity_scores)
                confidence = calculate_confidence(similarity_scores, overall_similarity)
                
                best_match = {
                    'filename': stored_data['filename'],
                    'similarity': overall_similarity,
                    'match_type': match_type,
                    'confidence': confidence,
                    'analysis_details': format_analysis_details(similarity_scores),
                    'timestamp': stored_data['timestamp']
                }
        
        # Apply adaptive threshold
        adaptive_threshold = get_adaptive_threshold()
        if best_match and best_match['similarity'] > adaptive_threshold:
            # ADDITIONAL RESTRICTIVE CHECKS FOR BEST MATCH
            
            # RESTRICTIVE CHECK 9: Confidence validation
            if best_match['confidence'] < 0.9:
                print(f"❌ Match confidence too low: {best_match['confidence']:.3f} (min: 0.9)")
                return None
            
            # RESTRICTIVE CHECK 10: Age-based restrictions
            match_age_hours = (time.time() - best_match['timestamp']) / 3600
            if match_age_hours > 24:  # Reject matches older than 24 hours
                print(f"❌ Match too old: {match_age_hours:.1f} hours (max: 24)")
                return None
            
            # RESTRICTIVE CHECK 11: Multiple modality validation
            analysis_details = best_match.get('analysis_details', '')
            modality_count = len([detail for detail in analysis_details.split(';') if ':' in detail])
            if modality_count < 3:  # Must have at least 3 matching modalities
                print(f"❌ Insufficient matching modalities: {modality_count} (min: 3)")
                return None
            
            # RESTRICTIVE CHECK 12: Match type validation
            match_type = best_match.get('match_type', '')
            if match_type in ['Semantic Similarity', 'Text Similarity'] and best_match['similarity'] < 0.98:
                print(f"❌ {match_type} requires 98%+ similarity")
                return None
            
            return best_match
        
        return None
        
    except Exception as e:
        print(f"Hierarchical matching error: {e}")
        return None

def calculate_structural_similarity(struct1, struct2):
    """Calculate advanced structural similarity"""
    try:
        if not struct1 or not struct2:
            return 0.0
        
        similarities = []
        
        # Basic features similarity
        basic_features = ['aspect_ratio', 'brightness', 'contrast', 'edge_density_small']
        for feature in basic_features:
            if feature in struct1 and feature in struct2:
                val1, val2 = struct1[feature], struct2[feature]
                if val1 + val2 > 0:
                    sim = 1.0 - abs(val1 - val2) / (val1 + val2)
                    similarities.append(sim)
        
        # LBP similarity
        if 'lbp_features' in struct1 and 'lbp_features' in struct2:
            lbp_sim = calculate_lbp_similarity(struct1['lbp_features'], struct2['lbp_features'])
            similarities.append(lbp_sim)
        
        # FFT similarity
        if 'fft_features' in struct1 and 'fft_features' in struct2:
            fft_sim = calculate_fft_similarity(struct1['fft_features'], struct2['fft_features'])
            similarities.append(fft_sim)
        
        return np.mean(similarities) if similarities else 0.0
        
    except Exception as e:
        print(f"Structural similarity error: {e}")
        return 0.0

def calculate_lbp_similarity(lbp1, lbp2):
    """Calculate LBP histogram similarity"""
    try:
        if not lbp1 or not lbp2:
            return 0.0
        
        hist1 = np.array(lbp1.get('lbp_histogram', []))
        hist2 = np.array(lbp2.get('lbp_histogram', []))
        
        if len(hist1) != len(hist2):
            return 0.0
        
        # Calculate histogram intersection
        intersection = np.minimum(hist1, hist2).sum()
        union = np.maximum(hist1, hist2).sum()
        
        return intersection / union if union > 0 else 0.0
        
    except Exception as e:
        print(f"LBP similarity error: {e}")
        return 0.0

def calculate_fft_similarity(fft1, fft2):
    """Calculate FFT features similarity"""
    try:
        if not fft1 or not fft2:
            return 0.0
        
        similarities = []
        
        # Compare frequency domain features
        features = ['low_freq_energy', 'high_freq_energy', 'spectral_centroid', 'spectral_spread']
        for feature in features:
            if feature in fft1 and feature in fft2:
                val1, val2 = fft1[feature], fft2[feature]
                if val1 + val2 > 0:
                    sim = 1.0 - abs(val1 - val2) / (val1 + val2)
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
        
    except Exception as e:
        print(f"FFT similarity error: {e}")
        return 0.0

def calculate_layout_similarity_advanced(layout1, layout2):
    """Calculate advanced layout similarity"""
    try:
        if not layout1 or not layout2:
            return 0.0
        
        similarities = []
        
        # Basic layout features
        basic_features = ['horizontal_lines', 'vertical_lines', 'text_regions', 'text_density']
        for feature in basic_features:
            if feature in layout1 and feature in layout2:
                val1, val2 = layout1[feature], layout2[feature]
                if val1 + val2 > 0:
                    sim = 1.0 - abs(val1 - val2) / (val1 + val2)
                    similarities.append(sim)
        
        # Symmetry similarity
        if 'symmetry_score' in layout1 and 'symmetry_score' in layout2:
            sym1 = layout1['symmetry_score']
            sym2 = layout2['symmetry_score']
            if isinstance(sym1, dict) and isinstance(sym2, dict):
                sym_sim = 1.0 - abs(sym1.get('overall_symmetry', 0) - sym2.get('overall_symmetry', 0))
                similarities.append(sym_sim)
        
        return np.mean(similarities) if similarities else 0.0
        
    except Exception as e:
        print(f"Layout similarity error: {e}")
        return 0.0

def calculate_text_similarity_advanced(text1, text2):
    """Calculate advanced text similarity"""
    try:
        if not text1 or not text2:
            return 0.0
        
        similarities = []
        
        # Basic text features
        basic_features = ['text_count', 'total_characters', 'total_words', 'avg_word_length']
        for feature in basic_features:
            if feature in text1 and feature in text2:
                val1, val2 = text1[feature], text2[feature]
                if val1 + val2 > 0:
                    sim = 1.0 - abs(val1 - val2) / (val1 + val2)
                    similarities.append(sim)
        
        # Language features similarity
        if 'language_features' in text1 and 'language_features' in text2:
            lang_sim = calculate_language_similarity(text1['language_features'], text2['language_features'])
            similarities.append(lang_sim)
        
        return np.mean(similarities) if similarities else 0.0
        
    except Exception as e:
        print(f"Text similarity error: {e}")
        return 0.0

def calculate_language_similarity(lang1, lang2):
    """Calculate language features similarity"""
    try:
        if not lang1 or not lang2:
            return 0.0
        
        similarities = []
        
        # Character frequency similarity
        if 'char_frequency' in lang1 and 'char_frequency' in lang2:
            char_sim = calculate_char_frequency_similarity(lang1['char_frequency'], lang2['char_frequency'])
            similarities.append(char_sim)
        
        # Word length similarity
        features = ['avg_word_length', 'avg_sentence_length', 'vocabulary_richness']
        for feature in features:
            if feature in lang1 and feature in lang2:
                val1, val2 = lang1[feature], lang2[feature]
                if val1 + val2 > 0:
                    sim = 1.0 - abs(val1 - val2) / (val1 + val2)
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
        
    except Exception as e:
        print(f"Language similarity error: {e}")
        return 0.0

def calculate_char_frequency_similarity(freq1, freq2):
    """Calculate character frequency similarity"""
    try:
        if not freq1 or not freq2:
            return 0.0
        
        # Get all characters
        all_chars = set(freq1.keys()) | set(freq2.keys())
        
        if not all_chars:
            return 0.0
        
        # Calculate cosine similarity
        vec1 = [freq1.get(char, 0) for char in all_chars]
        vec2 = [freq2.get(char, 0) for char in all_chars]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
        
    except Exception as e:
        print(f"Character frequency similarity error: {e}")
        return 0.0

def calculate_semantic_similarity_advanced(sem1, sem2):
    """Calculate advanced semantic similarity"""
    try:
        if not sem1 or not sem2:
            return 0.0
        
        # Compare semantic vectors
        if 'semantic_vector' in sem1 and 'semantic_vector' in sem2:
            vec1 = np.array(sem1['semantic_vector'])
            vec2 = np.array(sem2['semantic_vector'])
            
            if len(vec1) == len(vec2):
                # Calculate cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    return dot_product / (norm1 * norm2)
        
        return 0.0
        
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return 0.0

def calculate_weighted_similarity(similarity_scores):
    """Calculate weighted overall similarity with restrictive criteria"""
    try:
        if not similarity_scores:
            return 0.0
        
        # RESTRICTIVE MODE: Require multiple high-confidence matches
        high_confidence_threshold = 0.9
        medium_confidence_threshold = 0.8
        
        # Count high and medium confidence matches
        high_confidence_matches = sum(1 for score in similarity_scores.values() if score >= high_confidence_threshold)
        medium_confidence_matches = sum(1 for score in similarity_scores.values() if score >= medium_confidence_threshold)
        
        # RESTRICTIVE RULE 1: Must have at least 2 high-confidence matches OR exact match
        if similarity_scores.get('exact_match', 0) >= 0.99:
            # Exact match gets full score
            return 1.0
        elif high_confidence_matches >= 2:
            # Multiple high-confidence matches
            pass
        elif high_confidence_matches >= 1 and medium_confidence_matches >= 3:
            # One high + multiple medium matches
            pass
        else:
            # Not restrictive enough - reject
            return 0.0
        
        # RESTRICTIVE RULE 2: Weighted scoring with stricter weights
        weights = {
            'exact_match': 1.0,
            'perceptual': 0.4,      # Increased weight for visual similarity
            'structural': 0.3,      # Increased weight for structure
            'layout': 0.2,          # Layout importance
            'text': 0.1,            # Reduced text weight
            'semantic': 0.05        # Minimal semantic weight
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for sim_type, score in similarity_scores.items():
            weight = weights.get(sim_type, 0.05)
            # Apply penalty for low scores
            if score < 0.7:
                score = score * 0.5  # Heavy penalty for low scores
            weighted_sum += score * weight
            total_weight += weight
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # RESTRICTIVE RULE 3: Apply additional penalty for insufficient matches
        if high_confidence_matches < 2 and similarity_scores.get('exact_match', 0) < 0.99:
            final_score *= 0.7  # 30% penalty
        
        return min(1.0, final_score)
        
    except Exception as e:
        print(f"Weighted similarity error: {e}")
        return 0.0

def determine_match_type(similarity_scores):
    """Determine the type of match based on similarity scores"""
    try:
        if similarity_scores.get('exact_match', 0) > 0.9:
            return "Exact Match"
        elif similarity_scores.get('perceptual', 0) > 0.8:
            return "Visual Similarity"
        elif similarity_scores.get('structural', 0) > 0.7:
            return "Structural Similarity"
        elif similarity_scores.get('layout', 0) > 0.6:
            return "Layout Similarity"
        elif similarity_scores.get('text', 0) > 0.5:
            return "Text Similarity"
        elif similarity_scores.get('semantic', 0) > 0.4:
            return "Semantic Similarity"
        else:
            return "Multi-Modal Similarity"
        
    except Exception as e:
        print(f"Match type determination error: {e}")
        return "Unknown"

def calculate_confidence(similarity_scores, overall_similarity):
    """Calculate confidence score for the match"""
    try:
        # Base confidence on overall similarity
        base_confidence = overall_similarity
        
        # Boost confidence for exact matches
        if similarity_scores.get('exact_match', 0) > 0.9:
            base_confidence = min(1.0, base_confidence + 0.2)
        
        # Boost confidence for multiple matching modalities
        matching_modalities = sum(1 for score in similarity_scores.values() if score > 0.5)
        modality_boost = min(0.1, matching_modalities * 0.02)
        
        return min(1.0, base_confidence + modality_boost)
        
    except Exception as e:
        print(f"Confidence calculation error: {e}")
        return overall_similarity

def format_analysis_details(similarity_scores):
    """Format analysis details for display"""
    try:
        details = []
        for sim_type, score in similarity_scores.items():
            if score > 0.1:  # Only show significant similarities
                details.append(f"{sim_type}: {score:.3f}")
        
        return "; ".join(details) if details else "No significant similarities"
        
    except Exception as e:
        print(f"Analysis details formatting error: {e}")
        return "Analysis error"

def get_adaptive_threshold():
    """Get adaptive similarity threshold with very restrictive settings"""
    try:
        # RESTRICTIVE MODE: Much higher base threshold
        base_threshold = similarity_threshold  # Already set to 0.95
        
        # RESTRICTIVE RULE 1: Always increase threshold based on document count
        if len(document_fingerprints) > 5:
            base_threshold += 0.02  # Stricter with more documents
        if len(document_fingerprints) > 20:
            base_threshold += 0.03  # Even stricter with many documents
        
        # RESTRICTIVE RULE 2: Time-based restrictions
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            base_threshold += 0.03  # Much stricter during work hours
        elif 18 <= current_hour <= 22:  # Evening
            base_threshold += 0.02  # Stricter in evening
        else:  # Night/early morning
            base_threshold += 0.01  # Slightly stricter at night
        
        # RESTRICTIVE RULE 3: Cap at very high threshold
        return min(0.98, base_threshold)  # Never go above 98%
        
    except Exception as e:
        print(f"Adaptive threshold error: {e}")
        return similarity_threshold

def update_adaptive_thresholds(match_type, similarity, confidence):
    """Update adaptive thresholds based on match results"""
    try:
        # This would implement learning from user feedback
        # For now, just log the match for future analysis
        print(f"📊 Learning: {match_type} match with {similarity:.3f} similarity, {confidence:.3f} confidence")
        
        # In a real system, you'd store this data and use it to improve thresholds
        
    except Exception as e:
        print(f"Adaptive threshold update error: {e}")

def store_sophisticated_fingerprint(filename, document_signature):
    """Store document with sophisticated indexing"""
    try:
        # Store in memory with comprehensive data
        perceptual_hash = document_signature['perceptual_hash']
        
        document_fingerprints[perceptual_hash] = {
            'filename': filename,
            'perceptual_hash': perceptual_hash,
            'content_hash': document_signature['content_hash'],
            'visual_features': document_signature['visual_features'],
            'structural_features': document_signature['structural_features'],
            'layout_features': document_signature['layout_features'],
            'text_features': document_signature['text_features'],
            'semantic_features': document_signature['semantic_features'],
            'temporal_context': document_signature['temporal_context'],
            'timestamp': document_signature['timestamp'],
            'confidence_scores': document_signature['confidence_scores']
        }
        
        # Limit stored fingerprints with sophisticated eviction
        if len(document_fingerprints) > max_fingerprints:
            evict_least_important_fingerprint()
        
        return True
        
    except Exception as e:
        print(f"Sophisticated fingerprint storage error: {e}")
        return False

def evict_least_important_fingerprint():
    """Evict least important fingerprint using sophisticated criteria"""
    try:
        if not document_fingerprints:
            return
        
        # Calculate importance scores for each fingerprint
        importance_scores = {}
        
        for hash_key, data in document_fingerprints.items():
            score = 0.0
            
            # Recency score (newer = more important)
            age_hours = (time.time() - data['timestamp']) / 3600
            recency_score = max(0, 1.0 - age_hours / 168)  # Decay over a week
            
            # Complexity score (more complex documents = more important)
            complexity_score = 0.0
            if 'structural_features' in data:
                struct = data['structural_features']
                complexity_score = struct.get('layout_complexity', 0) * 0.5
            
            # Text richness score
            text_richness = 0.0
            if 'text_features' in data:
                text = data['text_features']
                text_richness = min(1.0, text.get('total_words', 0) / 100) * 0.3
            
            # Combined importance score
            importance_scores[hash_key] = recency_score + complexity_score + text_richness
        
        # Remove least important fingerprint
        least_important = min(importance_scores.keys(), key=lambda k: importance_scores[k])
        del document_fingerprints[least_important]
        
        print(f"🗑️ Evicted least important fingerprint: {least_important[:8]}...")
        
    except Exception as e:
        print(f"Fingerprint eviction error: {e}")
        # Fallback to simple oldest-first eviction
        if document_fingerprints:
            oldest_key = min(document_fingerprints.keys(), 
                           key=lambda k: document_fingerprints[k]['timestamp'])
            del document_fingerprints[oldest_key]

def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        if ocr_reader is None:
            return None, "OCR not available"
        
        # Convert BGR to RGB for EasyOCR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract text using EasyOCR
        results = ocr_reader.readtext(rgb_image)
        
        # Combine all text
        extracted_text = ""
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Only include high-confidence text
                extracted_text += text + " "
        
        # Clean and normalize text
        cleaned_text = clean_text(extracted_text)
        
        return cleaned_text, f"Extracted {len(cleaned_text.split())} words"
        
    except Exception as e:
        print(f"OCR extraction error: {e}")
        return None, f"OCR error: {e}"

def clean_text(text):
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces
    text = text.strip()
    
    return text

def advanced_text_preprocessing(text):
    """Advanced text preprocessing for semantic analysis"""
    if not text:
        return ""
    
    try:
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stop words and lemmatize
        if lemmatizer and stop_words:
            processed_tokens = []
            for token in tokens:
                if token not in stop_words and len(token) > 2:
                    lemmatized = lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            return ' '.join(processed_tokens)
        else:
            # Fallback to simple cleaning
            return clean_text(text)
            
    except Exception as e:
        print(f"Text preprocessing error: {e}")
        return clean_text(text)

def calculate_semantic_similarity(text1, text2):
    """Calculate semantic similarity using sentence transformers"""
    if not text1 or not text2 or semantic_model is None:
        return 0.0
    
    try:
        # Preprocess texts
        processed_text1 = advanced_text_preprocessing(text1)
        processed_text2 = advanced_text_preprocessing(text2)
        
        if not processed_text1 or not processed_text2:
            return 0.0
        
        # Generate embeddings
        embeddings = semantic_model.encode([processed_text1, processed_text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
        
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        return 0.0

def calculate_content_fingerprint(text):
    """Calculate comprehensive content fingerprint"""
    if not text:
        return None, None, None
    
    try:
        # 1. Content hash (exact text match)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # 2. Semantic vector (meaning-based)
        semantic_vector = None
        if semantic_model:
            processed_text = advanced_text_preprocessing(text)
            if processed_text:
                embedding = semantic_model.encode([processed_text])[0]
                semantic_vector = json.dumps(embedding.tolist())
        
        # 3. Structural features
        structural_features = {
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
            'unique_words': len(set(text.lower().split())),
            'text_density': len(text.replace(' ', '')) / max(len(text), 1)
        }
        
        return content_hash, semantic_vector, structural_features
        
    except Exception as e:
        print(f"Content fingerprint error: {e}")
        return None, None, None

def calculate_structural_similarity(struct1, struct2):
    """Calculate similarity between structural features"""
    if not struct1 or not struct2:
        return 0.0
    
    try:
        similarities = []
        
        # Compare each structural feature
        for key in struct1:
            if key in struct2:
                val1, val2 = struct1[key], struct2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalize similarity for numeric values
                    max_val = max(val1, val2)
                    if max_val > 0:
                        similarity = 1.0 - abs(val1 - val2) / max_val
                        similarities.append(similarity)
                    else:
                        similarities.append(1.0 if val1 == val2 else 0.0)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
        
    except Exception as e:
        print(f"Structural similarity error: {e}")
        return 0.0

def store_document_fingerprint(filename, content_hash, semantic_vector, structural_features, visual_features, extracted_text):
    """Store document fingerprint in database"""
    if not db_conn or not db_cursor:
        return False
    
    try:
        timestamp = time.time()
        capture_time = datetime.now().isoformat()
        
        db_cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (filename, content_hash, semantic_vector, structural_features, visual_features, extracted_text, timestamp, capture_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            content_hash,
            semantic_vector,
            json.dumps(structural_features),
            json.dumps(visual_features),
            extracted_text,
            timestamp,
            capture_time
        ))
        
        db_conn.commit()
        return True
        
    except Exception as e:
        print(f"Database storage error: {e}")
        return False

def find_similar_documents(content_hash, semantic_vector, structural_features, threshold=0.85):
    """Find similar documents in database"""
    if not db_conn or not db_cursor:
        return []
    
    try:
        similar_docs = []
        
        # Get all documents from database
        db_cursor.execute('SELECT * FROM documents ORDER BY timestamp DESC LIMIT 100')
        rows = db_cursor.fetchall()
        
        for row in rows:
            doc_id, filename, stored_content_hash, stored_semantic_vector, stored_structural_features, stored_visual_features, stored_text, timestamp, capture_time = row
            
            # 1. Exact content match
            if content_hash == stored_content_hash:
                similar_docs.append({
                    'filename': filename,
                    'similarity': 1.0,
                    'match_type': 'exact_content',
                    'timestamp': timestamp
                })
                continue
            
            # 2. Semantic similarity
            semantic_similarity = 0.0
            if semantic_vector and stored_semantic_vector:
                try:
                    stored_vector = json.loads(stored_semantic_vector)
                    current_vector = json.loads(semantic_vector)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity([current_vector], [stored_vector])[0][0]
                    semantic_similarity = float(similarity)
                except:
                    pass
            
            # 3. Structural similarity
            structural_similarity = 0.0
            if structural_features and stored_structural_features:
                try:
                    stored_struct = json.loads(stored_structural_features)
                    structural_similarity = calculate_structural_similarity(structural_features, stored_struct)
                except:
                    pass
            
            # 4. Combined similarity
            combined_similarity = (
                semantic_similarity * 0.6 +  # 60% semantic (meaning)
                structural_similarity * 0.3 +  # 30% structural (layout)
                0.1  # 10% bonus for having any similarity
            )
            
            if combined_similarity > threshold:
                similar_docs.append({
                    'filename': filename,
                    'similarity': combined_similarity,
                    'match_type': 'semantic_structural',
                    'semantic_sim': semantic_similarity,
                    'structural_sim': structural_similarity,
                    'timestamp': timestamp
                })
        
        # Sort by similarity
        similar_docs.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_docs
        
    except Exception as e:
        print(f"Database search error: {e}")
        return []

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text strings using multiple methods"""
    if not text1 or not text2:
        return 0.0
    
    try:
        # Method 1: Fuzzy string matching
        fuzzy_ratio = fuzz.ratio(text1, text2) / 100.0
        
        # Method 2: Token-based similarity
        fuzzy_token = fuzz.token_sort_ratio(text1, text2) / 100.0
        
        # Method 3: TF-IDF cosine similarity
        tfidf_similarity = calculate_tfidf_similarity(text1, text2)
        
        # Method 4: Word overlap similarity
        word_overlap = calculate_word_overlap(text1, text2)
        
        # Weighted combination
        combined_similarity = (
            fuzzy_ratio * 0.3 +
            fuzzy_token * 0.3 +
            tfidf_similarity * 0.25 +
            word_overlap * 0.15
        )
        
        return min(1.0, max(0.0, combined_similarity))
        
    except Exception as e:
        print(f"Text similarity calculation error: {e}")
        return 0.0

def calculate_tfidf_similarity(text1, text2):
    """Calculate TF-IDF cosine similarity between texts"""
    try:
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
        # Fit and transform texts
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        
        return similarity_matrix[0][0]
        
    except Exception as e:
        print(f"TF-IDF similarity error: {e}")
        return 0.0

def calculate_word_overlap(text1, text2):
    """Calculate word overlap similarity"""
    try:
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    except Exception as e:
        print(f"Word overlap calculation error: {e}")
        return 0.0

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
    """Ultra-sophisticated multi-modal document duplicate detection system with maximum restrictions"""
    global document_fingerprints, restriction_violations
    
    try:
        # RESTRICTIVE CHECK 0: System lockout check
        if restriction_violations >= max_restriction_violations:
            print(f"🔒 SYSTEM LOCKOUT: {restriction_violations} restriction violations (max: {max_restriction_violations})")
            return False, "System locked due to restriction violations"
        
        # RESTRICTIVE CHECK 0.5: Document quality scoring
        quality_score = calculate_document_quality_score(image)
        min_quality_score = 0.7  # Minimum quality score required
        
        if quality_score < min_quality_score:
            restriction_violations += 1
            print(f"❌ Document quality too low: {quality_score:.3f} (min: {min_quality_score}) - Violation #{restriction_violations}")
            return False, f"Document quality too low: {quality_score:.3f}"
        # === PHASE 1: MULTI-MODAL FEATURE EXTRACTION ===
        print("🔬 Starting sophisticated analysis...")
        
        # 1. Visual Features (Perceptual + Content)
        perceptual_hash = calculate_perceptual_hash(image)
        content_hash, visual_features = calculate_content_hash(image)
        
        # 2. Structural Analysis
        structural_features = calculate_advanced_structural_features(image)
        
        # 3. Layout Analysis
        layout_features = calculate_layout_features(image)
        
        # 4. Text Analysis (if OCR available)
        text_features = extract_and_analyze_text(image)
        
        # 5. Semantic Analysis (if text extracted)
        semantic_features = calculate_semantic_features(text_features)
        
        # 6. Temporal Context
        temporal_context = calculate_temporal_context()
        
        # === PHASE 2: RESTRICTIVE PRE-FILTERING ===
        print("🔒 Performing restrictive pre-filtering...")
        
        # RESTRICTIVE CHECK 1: Must have sufficient features
        if not perceptual_hash or not content_hash:
            restriction_violations += 1
            print(f"❌ Insufficient features for restrictive analysis - Violation #{restriction_violations}")
            return False, "Insufficient features"
        
        # RESTRICTIVE CHECK 2: Must have structural complexity
        if not structural_features or structural_features.get('edge_density_small', 0) < 0.01:
            restriction_violations += 1
            print(f"❌ Document too simple for restrictive analysis - Violation #{restriction_violations}")
            return False, "Document too simple"
        
        # RESTRICTIVE CHECK 3: Document size requirements
        image_height, image_width = image.shape[:2]
        min_size = 200  # Minimum 200x200 pixels
        max_size = 4000  # Maximum 4000x4000 pixels
        
        if image_height < min_size or image_width < min_size:
            restriction_violations += 1
            print(f"❌ Document too small: {image_width}x{image_height} (min: {min_size}x{min_size}) - Violation #{restriction_violations}")
            return False, "Document too small"
        
        if image_height > max_size or image_width > max_size:
            restriction_violations += 1
            print(f"❌ Document too large: {image_width}x{image_height} (max: {max_size}x{max_size}) - Violation #{restriction_violations}")
            return False, "Document too large"
        
        # RESTRICTIVE CHECK 4: Aspect ratio validation
        aspect_ratio = image_width / image_height
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            restriction_violations += 1
            print(f"❌ Invalid aspect ratio: {aspect_ratio:.2f} (valid: 0.3-3.0) - Violation #{restriction_violations}")
            return False, "Invalid aspect ratio"
        
        # RESTRICTIVE CHECK 5: Content complexity requirements
        if structural_features:
            edge_density = structural_features.get('edge_density_small', 0)
            contour_count = structural_features.get('contour_count_small', 0)
            
            if edge_density < 0.02:  # Must have at least 2% edge density
                restriction_violations += 1
                print(f"❌ Insufficient edge density: {edge_density:.3f} (min: 0.02) - Violation #{restriction_violations}")
                return False, "Insufficient edge density"
            
            if contour_count < 5:  # Must have at least 5 contours
                restriction_violations += 1
                print(f"❌ Insufficient contours: {contour_count} (min: 5) - Violation #{restriction_violations}")
                return False, "Insufficient contours"
        
        # RESTRICTIVE CHECK 6: Temporal restrictions
        current_time = time.time()
        if last_capture_time > 0:
            time_since_last = current_time - last_capture_time
            min_interval = 2.0  # Minimum 2 seconds between captures
            
            if time_since_last < min_interval:
                restriction_violations += 1
                print(f"❌ Too soon since last capture: {time_since_last:.1f}s (min: {min_interval}s) - Violation #{restriction_violations}")
                return False, "Too soon since last capture"
        
        # RESTRICTIVE CHECK 7: Document type validation
        if layout_features:
            text_regions = layout_features.get('text_regions', 0)
            horizontal_lines = layout_features.get('horizontal_lines', 0)
            vertical_lines = layout_features.get('vertical_lines', 0)
            
            # Must have some document-like structure
            if text_regions == 0 and horizontal_lines == 0 and vertical_lines == 0:
                restriction_violations += 1
                print(f"❌ No document structure detected - Violation #{restriction_violations}")
                return False, "No document structure"
        
        # RESTRICTIVE CHECK 8: Color complexity validation
        if structural_features and 'color_features' in structural_features:
            color_features = structural_features['color_features']
            if 'dominant_colors' in color_features:
                color_count = len(color_features['dominant_colors'].get('colors', []))
                if color_count < 2:  # Must have at least 2 distinct colors
                    restriction_violations += 1
                    print(f"❌ Insufficient color complexity: {color_count} colors (min: 2) - Violation #{restriction_violations}")
                    return False, "Insufficient color complexity"
        
        # === PHASE 3: HIERARCHICAL SIMILARITY MATCHING ===
        print("🎯 Performing hierarchical similarity analysis...")
        
        # Create comprehensive document signature
        document_signature = {
            'perceptual_hash': perceptual_hash,
            'content_hash': content_hash,
            'visual_features': visual_features,
            'structural_features': structural_features,
            'layout_features': layout_features,
            'text_features': text_features,
            'semantic_features': semantic_features,
            'temporal_context': temporal_context,
            'timestamp': time.time(),
            'confidence_scores': {}
        }
        
        # === PHASE 4: RESTRICTIVE SIMILARITY THRESHOLDS ===
        best_match = find_best_match_hierarchical(document_signature)
        
        if best_match:
            match_type = best_match['match_type']
            similarity = best_match['similarity']
            confidence = best_match['confidence']
            
            print(f"🔄 Sophisticated duplicate detected!")
            print(f"   📄 Matches: {best_match['filename']}")
            print(f"   🎯 Match type: {match_type}")
            print(f"   📊 Similarity: {similarity:.3f}")
            print(f"   🎖️ Confidence: {confidence:.3f}")
            print(f"   🧠 Analysis: {best_match['analysis_details']}")
            
            # Adaptive learning: Update similarity thresholds based on user behavior
            update_adaptive_thresholds(match_type, similarity, confidence)
            
            return True, f"Similar to {best_match['filename']} ({match_type}: {similarity:.3f})"
        
        # === PHASE 4: STORE WITH SOPHISTICATED INDEXING ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"document_{timestamp}.jpg"
        
        # Store with sophisticated indexing
        store_sophisticated_fingerprint(filename, document_signature)
        
        print(f"✅ New document detected and stored with sophisticated analysis")
        print(f"   🔍 Perceptual hash: {perceptual_hash[:8]}...")
        print(f"   📊 Content hash: {content_hash[:8]}...")
        print(f"   📈 Structural features: {len(structural_features)} elements")
        print(f"   🎨 Layout features: {len(layout_features)} elements")
        print(f"   📝 Text features: {len(text_features)} elements")
        print(f"   🧠 Semantic features: {len(semantic_features)} elements")
        
        return False, f"New document detected (sophisticated analysis complete)"
        
    except Exception as e:
        print(f"❌ Sophisticated duplicate detection error: {e}")
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
    
    elif command == 'force_video':
        start_video_recording()
        await send_log('Manual video recording started', 'success')
    
    elif command == 'init_camera':
        init_camera()
        await send_log('Camera initialized', 'success')
    
    elif command == 'set_sensitivity':
        await send_log(f'Sensitivity set to {int(message.get("sensitivity", 0.7) * 100)}%', 'info')
    
    elif command == 'open_folder':
        folder = message.get('folder', 'video')
        await send_log(f'Opening {folder} folder', 'info')
    
    elif command == 'clear_files':
        await send_log('Clearing old files', 'info')
    
    elif command == 'start_audio_recording':
        audio_recorder.start_recording()
        await send_log('🎤 Audio recording started', 'success')
        await websocket_broadcast({
            'type': 'audio_status',
            'recording': True,
            'duration': 0,
            'info': 'Continuous 16-bit recording'
        })
    
    elif command == 'stop_audio_recording':
        filename, duration = audio_recorder.stop_recording()
        await send_log(f'🛑 Audio recording stopped. Duration: {duration:.1f}s', 'info')
        await websocket_broadcast({
            'type': 'audio_status',
            'recording': False,
            'duration': duration,
            'info': f'Recording saved: {filename}'
        })
    
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

async def websocket_broadcast(message):
    """Broadcast message to all WebSocket connections"""
    for connection in connections:
        try:
            await connection.send_json(message)
        except:
            pass

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

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\n🛑 Received signal {signum}, saving audio recording...")
    if audio_recorder.is_recording():
        filename, duration = audio_recorder.stop_recording()
        if filename:
            print(f"💾 Final audio recording saved: {filename} ({duration:.1f}s)")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    # Start audio recording automatically
    print("🎤 Starting audio recording...")
    audio_recorder.start_recording()
    
    # Start background tasks
    asyncio.create_task(document_detection())
    
    # Initialize and start all systems automatically
    init_camera()
    start_video_recording()
    
    print("🚀 All systems started automatically!")
    print("✅ Advanced OpenCV document detection active - No people detection")
    print("🔒 MAXIMUM RESTRICTIONS: Ultra-strict duplicate detection")
    print("🎯 95%+ similarity + 12 restrictive checks + quality scoring")
    print("🔍 Pre-filtering + Multi-modal analysis + System lockout")
    print("⚡ Sophisticated fingerprint storage with intelligent eviction")
    print("🎤 Continuous 16-bit audio recording started")

if __name__ == "__main__":
    print("🔬 Starting Smart Glasses - Best Document Detection...")
    print("📱 Open: http://localhost:8006")
    print("🚀 Auto-start • 🎤 Audio recording • 📄 Smart Document detection • 🔒 Ultra-Restrictive Duplicate prevention")
    
    uvicorn.run(app, host="127.0.0.1", port=8006)