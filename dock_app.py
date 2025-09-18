#!/usr/bin/env python3
"""
Smart Glasses Dock Application
A simple FastAPI application for processing audio and documents from Smart Glasses
"""

import os
import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any
import asyncio
import threading

# Setup logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Audio processing imports
try:
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError as e:
    AUDIO_PROCESSING_AVAILABLE = False
    logger.warning(f"pydub not available - audio conversion disabled: {e}")

# Document processing imports
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSING_AVAILABLE = False
    logger.warning("OpenCV or PIL not available - document processing disabled")

# Transcription imports
try:
    import whisper
    import librosa
    import numpy as np
    TRANSCRIPTION_AVAILABLE = True
except ImportError:
    TRANSCRIPTION_AVAILABLE = False
    logger.warning("whisper or librosa not available - transcription disabled")

# Configuration
class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.audio_dir = self.data_dir / "audio"
        self.pictures_dir = self.data_dir / "pictures"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.documents_dir = self.data_dir / "documents"
        self.scanned_docs_dir = self.data_dir / "scanned_docs"
        self.db_path = self.base_dir / "dock.db"
        
        # Create directories
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.pictures_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.scanned_docs_dir.mkdir(parents=True, exist_ok=True)

config = Config()

# Audio Converter
class AudioConverter:
    def __init__(self):
        self.available = AUDIO_PROCESSING_AVAILABLE
        
    def convert_wav_to_mp3(self, input_path: Path, output_path: Path = None) -> Path:
        """Convert WAV file to MP3 format"""
        if not self.available:
            raise RuntimeError("Audio processing not available - pydub not installed")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.with_suffix('.mp3')
        
        try:
            logger.info(f"Converting {input_path} to MP3...")
            
            # Load WAV file
            audio = AudioSegment.from_wav(str(input_path))
            
            # Try to export as MP3, fallback to WAV if FFmpeg not available
            try:
                audio.export(
                    str(output_path),
                    format="mp3",
                    bitrate="192k",  # Good quality, reasonable file size
                    parameters=["-q:a", "2"]  # High quality encoding
                )
                logger.info(f"Successfully converted to MP3: {output_path}")
            except Exception as ffmpeg_error:
                logger.warning(f"FFmpeg not available for MP3 conversion: {ffmpeg_error}")
                logger.info("Creating compressed WAV instead...")
                
                # Fallback: Create a compressed WAV file
                compressed_path = input_path.with_suffix('.compressed.wav')
                audio.export(
                    str(compressed_path),
                    format="wav",
                    parameters=["-acodec", "pcm_s16le", "-ar", "22050"]  # Lower quality WAV
                )
                
                # Rename to MP3 extension for consistency
                compressed_path.rename(output_path)
                logger.info(f"Created compressed audio file: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise
    
    def get_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """Get audio file information"""
        if not self.available:
            return {}
        
        try:
            audio = AudioSegment.from_file(str(file_path))
            return {
                "duration": len(audio) / 1000.0,  # Duration in seconds
                "channels": audio.channels,
                "sample_rate": audio.frame_rate,
                "bitrate": audio.frame_rate * audio.channels * audio.sample_width * 8
            }
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}

# Initialize audio converter
audio_converter = AudioConverter()

# Transcription Service
class TranscriptionService:
    def __init__(self):
        self.available = TRANSCRIPTION_AVAILABLE
        self.model = None
        self.model_loaded = False
        
    def load_model(self, model_size="base"):
        """Load Whisper model"""
        if not self.available:
            raise RuntimeError("Transcription not available - whisper not installed")
        
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            self.model = whisper.load_model(model_size)
            self.model_loaded = True
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, file_path: Path) -> Dict[str, Any]:
        """Transcribe audio file to text using librosa (no FFmpeg required)"""
        if not self.available:
            raise RuntimeError("Transcription not available - whisper or librosa not installed")
        
        if not self.model_loaded:
            self.load_model()
        
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            logger.info(f"Transcribing audio file: {file_path.name}")
            
            # Load audio with librosa (no FFmpeg required)
            logger.info("Loading audio with librosa...")
            audio, sr = librosa.load(str(file_path), sr=16000)  # Whisper expects 16kHz
            
            logger.info(f"Audio loaded: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.1f}s")
            
            # Transcribe using the loaded audio array
            logger.info("Transcribing with Whisper...")
            result = self.model.transcribe(audio)
            
            # Extract transcription text
            transcription_text = result["text"].strip()
            
            # Get segments with timestamps
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
            
            # Generate SRT format
            srt_content = self._generate_srt(segments)
            
            logger.info(f"✅ Transcription completed: {len(transcription_text)} characters")
            
            return {
                "text": transcription_text,
                "srt": srt_content,
                "segments": segments,
                "language": result.get("language", "unknown"),
                "duration": result.get("duration", len(audio) / sr)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def _generate_srt(self, segments: List[Dict[str, Any]]) -> str:
        """Generate SRT subtitle format from segments"""
        srt_lines = []
        
        for i, segment in enumerate(segments, 1):
            start_time = self._format_srt_time(segment["start"])
            end_time = self._format_srt_time(segment["end"])
            text = segment["text"]
            
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between segments
        
        return "\n".join(srt_lines)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _transcribe_with_fallback(self, file_path: Path) -> Dict[str, Any]:
        """Fallback transcription using pydub"""
        try:
            from pydub import AudioSegment
            import tempfile
            import os
            
            logger.info("Using pydub fallback for transcription")
            
            # Load audio with pydub
            audio = AudioSegment.from_file(str(file_path))
            
            # Convert to WAV format that Whisper can handle
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Export as WAV with standard format
            audio.export(temp_path, format="wav")
            
            try:
                # Now transcribe the converted file
                result = self.model.transcribe(temp_path)
                
                # Extract transcription text
                transcription_text = result["text"].strip()
                
                # Get segments with timestamps
                segments = []
                for segment in result["segments"]:
                    segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip()
                    })
                
                logger.info(f"✅ Fallback transcription completed: {len(transcription_text)} characters")
                
                return {
                    "text": transcription_text,
                    "segments": segments,
                    "language": result.get("language", "unknown"),
                    "duration": result.get("duration", 0)
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Fallback transcription failed: {e}")
            raise
    
    def transcribe_async(self, file_path: Path) -> Dict[str, Any]:
        """Transcribe audio file asynchronously"""
        import asyncio
        import concurrent.futures
        
        # Run transcription in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.transcribe_audio, file_path)
            result = loop.run_in_executor(None, lambda: future.result())
            return result

# Initialize transcription service
transcription_service = TranscriptionService()

# Document Processor
class DocumentProcessor:
    def __init__(self):
        self.available = DOCUMENT_PROCESSING_AVAILABLE
        
    def process_document(self, image_path: Path) -> Dict[str, Any]:
        """Process document image like CamScanner - crop, enhance, and save"""
        if not self.available:
            raise RuntimeError("Document processing not available - OpenCV not installed")
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            logger.info(f"Processing document: {image_path.name}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour (document)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Approximate the contour
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # If we found 4 corners, we have a document
                if len(approx) == 4:
                    # Get the four corners
                    corners = approx.reshape(4, 2)
                    
                    # Order corners: top-left, top-right, bottom-right, bottom-left
                    ordered_corners = self._order_points(corners)
                    
                    # Get dimensions of the document
                    width = max(np.linalg.norm(ordered_corners[0] - ordered_corners[1]),
                               np.linalg.norm(ordered_corners[2] - ordered_corners[3]))
                    height = max(np.linalg.norm(ordered_corners[0] - ordered_corners[3]),
                                np.linalg.norm(ordered_corners[1] - ordered_corners[2]))
                    
                    # Define destination points
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]
                    ], dtype=np.float32)
                    
                    # Calculate perspective transform matrix
                    matrix = cv2.getPerspectiveTransform(ordered_corners, dst)
                    
                    # Apply perspective transform
                    warped = cv2.warpPerspective(image, matrix, (int(width), int(height)))
                    
                    # Enhance the image
                    enhanced = self._enhance_document(warped)
                    
                    # Save processed document
                    processed_path = config.scanned_docs_dir / f"scanned_{image_path.stem}.jpg"
                    cv2.imwrite(str(processed_path), enhanced)
                    
                    logger.info(f"✅ Document processed and saved: {processed_path.name}")
                    
                    return {
                        "success": True,
                        "processed_path": str(processed_path),
                        "original_size": image.shape[:2],
                        "processed_size": enhanced.shape[:2],
                        "corners_found": True
                    }
                else:
                    # No document detected, just enhance the original
                    enhanced = self._enhance_document(image)
                    processed_path = config.scanned_docs_dir / f"scanned_{image_path.stem}.jpg"
                    cv2.imwrite(str(processed_path), enhanced)
                    
                    logger.info(f"⚠️ No document corners found, enhanced original: {processed_path.name}")
                    
                    return {
                        "success": True,
                        "processed_path": str(processed_path),
                        "original_size": image.shape[:2],
                        "processed_size": enhanced.shape[:2],
                        "corners_found": False
                    }
            else:
                # No contours found, enhance original
                enhanced = self._enhance_document(image)
                processed_path = config.scanned_docs_dir / f"scanned_{image_path.stem}.jpg"
                cv2.imwrite(str(processed_path), enhanced)
                
                logger.info(f"⚠️ No contours found, enhanced original: {processed_path.name}")
                
                return {
                    "success": True,
                    "processed_path": str(processed_path),
                    "original_size": image.shape[:2],
                    "processed_size": enhanced.shape[:2],
                    "corners_found": False
                }
                
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _order_points(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _enhance_document(self, image):
        """Enhance document image for better readability"""
        # Convert to PIL for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(2.0)
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced

# Initialize document processor
document_processor = DocumentProcessor()

# Database Manager
class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Audio files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audio_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size INTEGER,
                    duration REAL,
                    channels INTEGER,
                    sample_rate INTEGER,
                    transcription TEXT,
                    srt_content TEXT,
                    transcription_status TEXT DEFAULT 'pending',
                    language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Document files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Scanned documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scanned_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_filename TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    scanned_filename TEXT NOT NULL,
                    scanned_path TEXT NOT NULL,
                    original_size TEXT,
                    scanned_size TEXT,
                    corners_found BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

# Initialize database
db_manager = DatabaseManager(config.db_path)

# FastAPI app
app = FastAPI(title="Smart Glasses Dock", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections
connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back the message
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        connections.remove(websocket)

async def broadcast_message(message: str):
    """Broadcast message to all WebSocket connections"""
    for connection in connections:
        try:
            await connection.send_text(message)
        except:
            pass

# API Routes
@app.get("/")
async def root():
    """Serve the main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Glasses Dock</title>
        <link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Roboto', Arial, sans-serif;
                background-color: #f8f9fa;
                color: #202124;
                line-height: 1.4;
            }
            
            /* Header */
            .header {
                background: white;
                border-bottom: 1px solid #dadce0;
                padding: 8px 24px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                position: sticky;
                top: 0;
                z-index: 100;
            }
            
            .header-left {
                display: flex;
                align-items: center;
                gap: 16px;
            }
            
            .logo {
                display: flex;
                align-items: center;
                gap: 8px;
                font-family: 'Google Sans', Arial, sans-serif;
                font-size: 22px;
                font-weight: 400;
                color: #5f6368;
            }
            
            .logo-icon {
                width: 24px;
                height: 24px;
                background: linear-gradient(135deg, #4285f4, #34a853);
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 14px;
            }
            
            .search-box {
                background: #f1f3f4;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                width: 400px;
                font-size: 14px;
                outline: none;
            }
            
            .search-box:focus {
                background: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .header-right {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            /* Main Content */
            .main-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 24px;
            }
            
            /* Status Card */
            .status-card {
                background: white;
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 1px solid #dadce0;
            }
            
            .status-title {
                font-family: 'Google Sans', Arial, sans-serif;
                font-size: 18px;
                font-weight: 500;
                color: #202124;
                margin-bottom: 16px;
            }
            
            .status-item {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 8px;
                font-size: 14px;
                color: #5f6368;
            }
            
            .status-icon {
                width: 16px;
                height: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            /* Action Buttons */
            .action-bar {
                display: flex;
                gap: 8px;
                margin-bottom: 24px;
            }
            
            .btn {
                background: #1a73e8;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .btn:hover {
                background: #1557b0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            }
            
            .btn-secondary {
                background: white;
                color: #5f6368;
                border: 1px solid #dadce0;
            }
            
            .btn-secondary:hover {
                background: #f8f9fa;
                border-color: #5f6368;
            }
            
            /* Files Section */
            .files-section {
                background: white;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border: 1px solid #dadce0;
                overflow: hidden;
            }
            
            .files-header {
                padding: 16px 24px;
                border-bottom: 1px solid #dadce0;
                background: #f8f9fa;
            }
            
            .files-title {
                font-family: 'Google Sans', Arial, sans-serif;
                font-size: 16px;
                font-weight: 500;
                color: #202124;
            }
            
            /* Tab Styles */
            .tab-buttons {
                display: flex;
                gap: 8px;
                margin-top: 12px;
            }
            
            .tab-btn {
                background: transparent;
                border: 1px solid #dadce0;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
                color: #5f6368;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .tab-btn:hover {
                background: #f8f9fa;
                border-color: #5f6368;
            }
            
            .tab-btn.active {
                background: #4285f4;
                border-color: #4285f4;
                color: white;
            }
            
            .tab-content {
                display: none;
                min-height: 200px;
            }
            
            .tab-content.active {
                display: block;
            }
            
            /* Search Styles */
            .search-section {
                margin-bottom: 20px;
                padding: 16px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e8eaed;
            }
            
            .search-container {
                position: relative;
                display: flex;
                align-items: center;
            }
            
            .search-input {
                width: 100%;
                padding: 12px 40px 12px 16px;
                border: 1px solid #dadce0;
                border-radius: 24px;
                font-size: 14px;
                background: white;
                transition: all 0.2s ease;
            }
            
            .search-input:focus {
                outline: none;
                border-color: #4285f4;
                box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.2);
            }
            
            .clear-search-btn {
                position: absolute;
                right: 12px;
                background: none;
                border: none;
                color: #5f6368;
                cursor: pointer;
                font-size: 16px;
                padding: 4px;
                border-radius: 50%;
                transition: all 0.2s ease;
            }
            
            .clear-search-btn:hover {
                background: #f1f3f4;
                color: #202124;
            }
            
            .search-results {
                margin-top: 16px;
                background: white;
                border-radius: 8px;
                border: 1px solid #e8eaed;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .search-results-header {
                padding: 12px 16px;
                background: #f8f9fa;
                border-bottom: 1px solid #e8eaed;
                font-weight: 500;
                color: #5f6368;
                font-size: 14px;
            }
            
            .search-results-list {
                padding: 8px;
            }
            
            .search-result-item {
                padding: 12px;
                border-radius: 6px;
                margin-bottom: 4px;
                background: #f8f9fa;
                border: 1px solid #e8eaed;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .search-result-item:hover {
                background: #e8f0fe;
                border-color: #4285f4;
            }
            
            .search-result-item:last-child {
                margin-bottom: 0;
            }
            
            .search-result-title {
                font-weight: 500;
                color: #202124;
                margin-bottom: 4px;
                font-size: 14px;
            }
            
            .search-result-date {
                color: #5f6368;
                font-size: 11px;
                margin-bottom: 6px;
                font-style: italic;
            }
            
            .search-result-preview {
                color: #5f6368;
                font-size: 12px;
                line-height: 1.4;
                margin-bottom: 8px;
            }
            
            .search-result-actions {
                display: flex;
                gap: 8px;
            }
            
            .search-result-btn {
                background: #4285f4;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .search-result-btn:hover {
                background: #3367d6;
            }
            
            .files-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 0;
                padding: 8px;
            }
            
            .file-card {
                padding: 16px;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s;
                border: 1px solid transparent;
                margin: 4px;
            }
            
            .file-card:hover {
                background: #f8f9fa;
                border-color: #dadce0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .file-icon {
                width: 40px;
                height: 40px;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 12px;
                font-size: 18px;
                color: white;
                font-weight: bold;
            }
            
            .file-icon.audio {
                background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            }
            
            .file-icon.document {
                background: linear-gradient(135deg, #4ecdc4, #44a08d);
            }
            
            .file-name {
                font-size: 14px;
                font-weight: 500;
                color: #202124;
                margin-bottom: 4px;
                word-break: break-word;
            }
            
            .file-info {
                font-size: 12px;
                color: #5f6368;
                line-height: 1.3;
            }
            
            .file-header {
                cursor: pointer;
            }
            
            .file-actions {
                margin-top: 12px;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            
            .action-btn {
                background: #4285f4;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
            }
            
            .action-btn:hover {
                background: #3367d6;
                transform: translateY(-1px);
            }
            
            .delete-btn {
                background: #ea4335;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
            }
            
            .delete-btn:hover {
                background: #d33b2c;
                transform: translateY(-1px);
            }
            
            .transcribe-btn {
                background: #34a853;
            }
            
            .transcribe-btn:hover {
                background: #2d8f47;
            }
            
            .view-transcription-btn {
                background: #ea4335;
            }
            
            .view-transcription-btn:hover {
                background: #d33b2c;
            }
            
            .download-srt-btn {
                background: #9c27b0;
            }
            
            .download-srt-btn:hover {
                background: #7b1fa2;
            }
            
            .download-transcript-btn {
                background: #ff9800;
            }
            
            .download-transcript-btn:hover {
                background: #f57c00;
            }
            
            .transcription-status {
                font-size: 11px;
                color: #5f6368;
                text-align: center;
                padding: 4px;
                border-radius: 4px;
                background: #f8f9fa;
            }
            
            /* Transcription Modal */
            .transcription-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
            }
            
            .modal-content {
                background: white;
                border-radius: 12px;
                width: 90%;
                max-width: 600px;
                max-height: 80vh;
                overflow: hidden;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            }
            
            .modal-header {
                padding: 20px 24px;
                border-bottom: 1px solid #dadce0;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .modal-header h3 {
                font-size: 18px;
                font-weight: 500;
                color: #202124;
                margin: 0;
            }
            
            .close-btn {
                background: none;
                border: none;
                font-size: 24px;
                color: #5f6368;
                cursor: pointer;
                padding: 0;
                width: 32px;
                height: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                transition: all 0.2s;
            }
            
            .close-btn:hover {
                background: #f8f9fa;
                color: #202124;
            }
            
            .modal-body {
                padding: 24px;
                max-height: 50vh;
                overflow-y: auto;
            }
            
            .transcription-info {
                background: #f8f9fa;
                padding: 16px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            
            .transcription-info p {
                margin: 4px 0;
                font-size: 14px;
                color: #5f6368;
            }
            
            .transcription-text h4 {
                font-size: 16px;
                font-weight: 500;
                color: #202124;
                margin-bottom: 12px;
            }
            
            .text-content {
                background: white;
                border: 1px solid #dadce0;
                border-radius: 8px;
                padding: 16px;
                font-size: 14px;
                line-height: 1.6;
                color: #202124;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            .modal-footer {
                padding: 16px 24px;
                border-top: 1px solid #dadce0;
                display: flex;
                gap: 12px;
                justify-content: flex-end;
            }
            
            .file-type {
                text-transform: uppercase;
                font-weight: 500;
                margin-bottom: 2px;
            }
            
            .file-size {
                margin-bottom: 2px;
            }
            
            .file-date {
                color: #9aa0a6;
            }
            
            /* Empty State */
            .empty-state {
                text-align: center;
                padding: 48px 24px;
                color: #5f6368;
            }
            
            .empty-icon {
                font-size: 48px;
                margin-bottom: 16px;
                opacity: 0.5;
            }
            
            .empty-title {
                font-size: 16px;
                font-weight: 500;
                margin-bottom: 8px;
                color: #202124;
            }
            
            .empty-description {
                font-size: 14px;
                margin-bottom: 24px;
            }
            
            /* Loading State */
            .loading {
                text-align: center;
                padding: 48px 24px;
                color: #5f6368;
            }
            
            .spinner {
                width: 24px;
                height: 24px;
                border: 2px solid #dadce0;
                border-top: 2px solid #1a73e8;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Responsive */
            @media (max-width: 768px) {
                .header {
                    padding: 8px 16px;
                }
                
                .search-box {
                    width: 200px;
                }
                
                .main-content {
                    padding: 16px;
                }
                
                .files-grid {
                    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                }
            }
        </style>
    </head>
    <body>
        <!-- Header -->
        <div class="header">
            <div class="header-left">
                <div class="logo">
                    <div class="logo-icon">SG</div>
                    Smart Glasses Dock
                </div>
                <input type="text" class="search-box" placeholder="Search files..." id="searchInput">
            </div>
            <div class="header-right">
                <button class="btn" onclick="scanDirectories()">
                    <span>📁</span> Scan Directories
                </button>
                <button class="btn btn-secondary" onclick="loadFiles()">
                    <span>🔄</span> Refresh
                </button>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <!-- Status Card -->
            <div class="status-card">
                <div class="status-title">System Status</div>
                <div class="status-item">
                    <div class="status-icon">✅</div>
                    <span>Backend running on port 8007</span>
                </div>
                <div class="status-item">
                    <div class="status-icon">📁</div>
                    <span>Audio directory: <span id="audio-status">Checking...</span></span>
                </div>
                <div class="status-item">
                    <div class="status-icon">📄</div>
                    <span>Pictures directory: <span id="doc-status">Checking...</span></span>
                    <span>Transcripts directory: <span id="transcripts-status">Checking...</span></span>
                    <span>Documents directory: <span id="documents-status">Checking...</span></span>
                </div>
            </div>
            
            <!-- Files Section with Tabs -->
            <div class="files-section">
                <div class="files-header">
                    <div class="files-title">Files</div>
                    <div class="tab-buttons">
                        <button class="tab-btn active" onclick="switchTab('audio')" id="audio-tab">
                            🎵 Audio (<span id="audio-count">0</span>)
                        </button>
                        <button class="tab-btn" onclick="switchTab('pictures')" id="pictures-tab">
                            📸 Pictures (<span id="pictures-count">0</span>)
                        </button>
                        <button class="tab-btn" onclick="switchTab('scanned')" id="scanned-tab">
                            📄 Scanned Docs (<span id="scanned-count">0</span>)
                        </button>
                    </div>
                </div>
                
                <!-- Audio Files Tab -->
                <div id="audio-tab-content" class="tab-content active">
                    <!-- Search Bar for Audio Tab -->
                    <div class="search-section">
                        <div class="search-container">
                            <input type="text" id="transcriptSearch" placeholder="🔍 Search through transcripts..." 
                                   class="search-input" oninput="searchTranscripts(this.value)">
                            <button onclick="clearSearch()" class="clear-search-btn" title="Clear search">✕</button>
                        </div>
                        <div id="search-results" class="search-results" style="display: none;">
                            <div class="search-results-header">
                                <span id="search-count">0</span> results found
                            </div>
                            <div id="search-results-list" class="search-results-list"></div>
                        </div>
                    </div>
                    
                    <div id="audio-files-grid" class="files-grid">
                        <div class="loading">
                            <div class="spinner"></div>
                            <div>Loading audio files...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Pictures Tab -->
                <div id="pictures-tab-content" class="tab-content">
                    <div id="pictures-files-grid" class="files-grid">
                        <div class="loading">
                            <div class="spinner"></div>
                            <div>Loading pictures...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Scanned Documents Tab -->
                <div id="scanned-tab-content" class="tab-content">
                    <div id="scanned-files-grid" class="files-grid">
                        <div class="loading">
                            <div class="spinner"></div>
                            <div>Loading scanned documents...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let ws = null;
            
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8007/ws');
                ws.onopen = function() {
                    console.log('WebSocket connected');
                };
                ws.onmessage = function(event) {
                    console.log('Message:', event.data);
                    try {
                        const message = JSON.parse(event.data);
                        if (message.type === 'file_update') {
                            // Refresh files when there's an update
                            loadFiles();
                        }
                    } catch (e) {
                        console.log('Raw message:', event.data);
                    }
                };
                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    setTimeout(connectWebSocket, 3000);
                };
            }
            
            async function loadFiles() {
                try {
                    // Load regular files
                    const response = await fetch('/api/files');
                    const data = await response.json();
                    
                    // Load scanned documents
                    const scannedResponse = await fetch('/api/scanned-documents');
                    const scannedData = await scannedResponse.json();
                    
                    // Combine files and scanned documents
                    const allFiles = [...data.files];
                    
                    // Add scanned documents with proper type
                    scannedData.scanned_documents.forEach(doc => {
                        allFiles.push({
                            id: doc.id,
                            name: doc.scanned_filename,
                            type: 'scanned',
                            size: 0, // We don't store size for scanned docs
                            created_at: doc.created_at,
                            corners_found: doc.corners_found,
                            original_filename: doc.original_filename,
                            scanned_path: doc.scanned_path
                        });
                    });
                    
                    displayFiles(allFiles);
                } catch (error) {
                    console.error('Error loading files:', error);
                    document.getElementById('audio-files-grid').innerHTML = '<p>Error loading files</p>';
                    document.getElementById('pictures-files-grid').innerHTML = '<p>Error loading files</p>';
                    document.getElementById('scanned-files-grid').innerHTML = '<p>Error loading files</p>';
                }
            }
            
            // Tab management
            let currentTab = 'audio';
            let allFiles = [];
            let searchResults = [];
            
            // Search functionality
            let searchTimeout;
            
            async function searchTranscripts(searchTerm) {
                // Clear previous timeout
                if (searchTimeout) {
                    clearTimeout(searchTimeout);
                }
                
                // If search term is empty or too short, hide results but don't clear input
                if (!searchTerm || searchTerm.length < 2) {
                    document.getElementById('search-results').style.display = 'none';
                    return;
                }
                
                // Debounce the search - wait 300ms after user stops typing
                searchTimeout = setTimeout(async () => {
                    try {
                        // Get all audio files with transcriptions
                        const response = await fetch('/api/files');
                        const data = await response.json();
                        const audioFiles = data.files.filter(f => f.type === 'audio' && f.transcription_status === 'completed');
                        
                        searchResults = [];
                        
                        for (const file of audioFiles) {
                            try {
                                const transcriptResponse = await fetch(`/api/transcription/${file.id}`);
                                if (transcriptResponse.ok) {
                                    const transcriptData = await transcriptResponse.json();
                                    const transcription = transcriptData.transcription || '';
                                    
                                    // Search in transcription text
                                    const searchIndex = transcription.toLowerCase().indexOf(searchTerm.toLowerCase());
                                    if (searchIndex !== -1) {
                                        // Get context around the match
                                        const start = Math.max(0, searchIndex - 50);
                                        const end = Math.min(transcription.length, searchIndex + searchTerm.length + 50);
                                        const preview = transcription.substring(start, end);
                                        
                                        searchResults.push({
                                            file: file,
                                            matchIndex: searchIndex,
                                            preview: preview,
                                            searchTerm: searchTerm
                                        });
                                    }
                                }
                            } catch (error) {
                                console.error(`Error searching file ${file.id}:`, error);
                            }
                        }
                        
                        displaySearchResults(searchResults);
                        
                    } catch (error) {
                        console.error('Error searching transcripts:', error);
                    }
                }, 300);
            }
            
            function displaySearchResults(results) {
                const searchResultsDiv = document.getElementById('search-results');
                const searchCountSpan = document.getElementById('search-count');
                const searchResultsList = document.getElementById('search-results-list');
                
                if (results.length === 0) {
                    searchResultsDiv.style.display = 'none';
                    return;
                }
                
                searchCountSpan.textContent = results.length;
                searchResultsDiv.style.display = 'block';
                
                searchResultsList.innerHTML = results.map(result => {
                    const highlightedPreview = highlightSearchTerm(result.preview, result.searchTerm);
                    const formattedDate = formatDate(result.file.created_at);
                    return `
                        <div class="search-result-item" onclick="openAudioFromSearch(${result.file.id})">
                            <div class="search-result-title">🎵 ${result.file.name}</div>
                            <div class="search-result-date">📅 ${formattedDate}</div>
                            <div class="search-result-preview">${highlightedPreview}</div>
                            <div class="search-result-actions">
                                <button class="search-result-btn" onclick="event.stopPropagation(); openAudioFromSearch(${result.file.id})">
                                    ▶️ Play Audio
                                </button>
                                <button class="search-result-btn" onclick="event.stopPropagation(); viewTranscription(${result.file.id})">
                                    👁️ View Full Text
                                </button>
                            </div>
                        </div>
                    `;
                }).join('');
            }
            
            function highlightSearchTerm(text, searchTerm) {
                const regex = new RegExp(`(${searchTerm})`, 'gi');
                return text.replace(regex, '<mark style="background: #ffeb3b; padding: 2px 4px; border-radius: 2px;">$1</mark>');
            }
            
            function clearSearch() {
                // Clear the search input
                document.getElementById('transcriptSearch').value = '';
                // Hide search results
                document.getElementById('search-results').style.display = 'none';
                // Clear search results array
                searchResults = [];
                // Clear any pending search timeout
                if (searchTimeout) {
                    clearTimeout(searchTimeout);
                }
            }
            
            async function openAudioFromSearch(fileId) {
                try {
                    const response = await fetch(`/api/audio/${fileId}`);
                    if (response.ok) {
                        const audioData = await response.json();
                        showAudioPlayer(audioData);
                    } else {
                        alert('Error loading audio file');
                    }
                } catch (error) {
                    alert(`Error opening audio: ${error.message}`);
                }
            }
            
            function switchTab(tabName) {
                currentTab = tabName;
                
                // Update tab buttons
                document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                document.getElementById(`${tabName}-tab`).classList.add('active');
                
                // Update tab content
                document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
                document.getElementById(`${tabName}-tab-content`).classList.add('active');
                
                // Display files for current tab
                displayFilesForTab(tabName);
            }
            
            function displayFilesForTab(tabName) {
                let files = [];
                let gridId = '';
                
                switch(tabName) {
                    case 'audio':
                        files = allFiles.filter(f => f.type === 'audio');
                        gridId = 'audio-files-grid';
                        break;
                    case 'pictures':
                        files = allFiles.filter(f => f.type === 'document');
                        gridId = 'pictures-files-grid';
                        break;
                    case 'scanned':
                        files = allFiles.filter(f => f.type === 'scanned');
                        gridId = 'scanned-files-grid';
                        break;
                }
                
                const grid = document.getElementById(gridId);
                if (files.length === 0) {
                    grid.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-icon">📁</div>
                            <div class="empty-title">No ${tabName} files found</div>
                            <div class="empty-description">Upload files or scan directories to get started</div>
                        </div>
                    `;
                    return;
                }
                
                grid.innerHTML = files.map(file => `
                    <div class="file-card">
                        <div class="file-header" onclick="openFile('${file.name}')">
                            <div class="file-icon ${file.type}">
                                ${file.type === 'audio' ? '🎵' : file.type === 'scanned' ? '📄' : '📸'}
                            </div>
                            <div class="file-name">${file.name}</div>
                        </div>
                        <div class="file-info">
                            <div class="file-type">${file.type}</div>
                            <div class="file-size">${formatFileSize(file.size)}</div>
                            <div class="file-date">${formatDate(file.created_at)}</div>
                        </div>
                        ${file.type === 'audio' ? `
                            <div class="file-actions">
                                <button class="action-btn play-btn" onclick="playAudio(${file.id})" 
                                        title="Play Audio">
                                    ▶️ Play
                                </button>
                                <button class="action-btn transcribe-btn" onclick="transcribeFile(${file.id})" 
                                        title="Transcribe Audio">
                                    📝 Transcribe
                                </button>
                                <button class="action-btn view-transcription-btn" onclick="viewTranscription(${file.id})" 
                                        title="View Transcription" style="display: ${file.transcription_status === 'completed' ? 'block' : 'none'}">
                                    👁️ View Text
                                </button>
                                <button class="action-btn download-srt-btn" onclick="downloadSRT(${file.id})" 
                                        title="Download SRT File" style="display: ${file.transcription_status === 'completed' ? 'block' : 'none'}">
                                    📄 Download SRT
                                </button>
                                <button class="action-btn download-transcript-btn" onclick="downloadTranscript(${file.id})" 
                                        title="Download Transcript" style="display: ${file.transcription_status === 'completed' ? 'block' : 'none'}">
                                    📝 Download Text
                                </button>
                                <button class="delete-btn" onclick="deleteFile(${file.id})" 
                                        title="Delete File">
                                    🗑️ Delete
                                </button>
                                <div class="transcription-status" id="status-${file.id}">
                                    ${file.transcription_status === 'processing' ? '🔄 Processing...' : 
                                      file.transcription_status === 'completed' ? '✅ Transcribed' : 
                                      file.transcription_status === 'failed' ? '❌ Failed' : '⏳ Pending'}
                                </div>
                            </div>
                        ` : file.type === 'document' ? `
                            <div class="file-actions">
                                <button class="action-btn view-doc-btn" onclick="viewDocument(${file.id})" 
                                        title="View Document">
                                    👁️ View
                                </button>
                                <button class="action-btn download-doc-btn" onclick="downloadDocument(${file.id})" 
                                        title="Download Document">
                                    📥 Download
                                </button>
                                <button class="delete-btn" onclick="deleteFile(${file.id})" 
                                        title="Delete File">
                                    🗑️ Delete
                                </button>
                                <div class="processing-status" id="doc-status-${file.id}">
                                    ${file.processed ? '✅ Processed' : '⏳ Processing...'}
                                </div>
                            </div>
                        ` : file.type === 'scanned' ? `
                            <div class="file-actions">
                                <button class="action-btn view-doc-btn" onclick="viewScannedDocument(${file.id})" 
                                        title="View Scanned Document">
                                    👁️ View
                                </button>
                                <button class="action-btn download-doc-btn" onclick="downloadScannedDocument(${file.id})" 
                                        title="Download Scanned Document">
                                    📥 Download
                                </button>
                                <button class="delete-btn" onclick="deleteFile(${file.id})" 
                                        title="Delete File">
                                    🗑️ Delete
                                </button>
                                <div class="processing-status" id="scanned-status-${file.id}">
                                    ${file.corners_found ? '✅ Corners Detected' : '⚠️ Enhanced Only'}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                `).join('');
            }
            
            function displayFiles(files) {
                allFiles = files;
                
                // Update tab counts
                const audioCount = files.filter(f => f.type === 'audio').length;
                const picturesCount = files.filter(f => f.type === 'document').length;
                const scannedCount = files.filter(f => f.type === 'scanned').length;
                
                document.getElementById('audio-count').textContent = audioCount;
                document.getElementById('pictures-count').textContent = picturesCount;
                document.getElementById('scanned-count').textContent = scannedCount;
                
                // Display files for current tab
                displayFilesForTab(currentTab);
            }
            
            // File deletion
            async function deleteFile(fileId) {
                if (!confirm('Are you sure you want to delete this file? This action cannot be undone.')) {
                    return;
                }
                
                try {
                    const response = await fetch(`/api/files/${fileId}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(`File deleted: ${result.message}`);
                        
                        // Refresh files to update the interface
                        loadFiles();
                    } else {
                        const error = await response.text();
                        alert(`Error deleting file: ${error}`);
                    }
                } catch (error) {
                    alert(`Error deleting file: ${error.message}`);
                }
            }
            
            // Scanned document functions
            async function viewScannedDocument(docId) {
                try {
                    const response = await fetch(`/api/scanned-document/${docId}/view`);
                    
                    if (response.ok) {
                        const docData = await response.json();
                        showDocumentViewer(docData);
                    } else {
                        const error = await response.text();
                        alert(`Error loading scanned document: ${error}`);
                    }
                } catch (error) {
                    alert(`Error loading scanned document: ${error.message}`);
                }
            }
            
            async function downloadScannedDocument(docId) {
                try {
                    const response = await fetch(`/api/scanned-document/${docId}/view`);
                    
                    if (response.ok) {
                        const docData = await response.json();
                        
                        // Create download link
                        const a = document.createElement('a');
                        a.href = `/files/scanned/${docData.filename}`;
                        a.download = docData.filename;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        
                        alert(`Scanned document downloaded: ${docData.filename}`);
                    } else {
                        const error = await response.text();
                        alert(`Error downloading scanned document: ${error}`);
                    }
                } catch (error) {
                    alert(`Error downloading scanned document: ${error.message}`);
                }
            }
            
            function formatDate(dateString) {
                const date = new Date(dateString);
                const now = new Date();
                const diffTime = Math.abs(now - date);
                const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
                
                if (diffDays === 1) return 'Today';
                if (diffDays === 2) return 'Yesterday';
                if (diffDays <= 7) return `${diffDays - 1} days ago`;
                return date.toLocaleDateString();
            }
            
            function openFile(filename) {
                alert(`Opening file: ${filename}`);
                // Add file opening logic here
            }
            
            async function transcribeFile(fileId) {
                try {
                    const response = await fetch(`/api/transcribe/${fileId}`, {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(`Transcription started: ${result.message}`);
                        
                        // Update status display
                        const statusElement = document.getElementById(`status-${fileId}`);
                        if (statusElement) {
                            statusElement.textContent = '🔄 Processing...';
                        }
                        
                        // Refresh files after a delay
                        setTimeout(loadFiles, 2000);
                    } else {
                        const error = await response.text();
                        alert(`Error: ${error}`);
                    }
                } catch (error) {
                    alert(`Error starting transcription: ${error.message}`);
                }
            }
            
            async function viewTranscription(fileId) {
                try {
                    const response = await fetch(`/api/transcription/${fileId}`);
                    
                    if (response.ok) {
                        const result = await response.json();
                        
                        // Create modal to display transcription
                        const modal = document.createElement('div');
                        modal.className = 'transcription-modal';
                        modal.innerHTML = `
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h3>📝 Transcription: ${result.filename}</h3>
                                    <button class="close-btn" onclick="closeModal()">&times;</button>
                                </div>
                                <div class="modal-body">
                                    <div class="transcription-info">
                                        <p><strong>Status:</strong> ${result.status}</p>
                                        <p><strong>Language:</strong> ${result.language || 'Unknown'}</p>
                                        <p><strong>Duration:</strong> ${result.duration ? Math.round(result.duration) + 's' : 'Unknown'}</p>
                                    </div>
                                    <div class="transcription-text">
                                        <h4>Transcription:</h4>
                                        <div class="text-content">${result.transcription || 'No transcription available'}</div>
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button class="action-btn" onclick="copyTranscription('${result.transcription.replace(/'/g, "\\'")}')">📋 Copy Text</button>
                                    <button class="action-btn" onclick="closeModal()">Close</button>
                                </div>
                            </div>
                        `;
                        
                        document.body.appendChild(modal);
                        
                    } else {
                        const error = await response.text();
                        alert(`Error: ${error}`);
                    }
                } catch (error) {
                    alert(`Error loading transcription: ${error.message}`);
                }
            }
            
            function closeModal() {
                const modal = document.querySelector('.transcription-modal');
                if (modal) {
                    modal.remove();
                }
            }
            
            function copyTranscription(text) {
                navigator.clipboard.writeText(text).then(() => {
                    alert('Transcription copied to clipboard!');
                }).catch(err => {
                    alert('Failed to copy text');
                });
            }
            
            async function downloadSRT(fileId) {
                try {
                    const response = await fetch(`/api/srt/${fileId}`);
                    
                    if (response.ok) {
                        const result = await response.json();
                        
                        // Create and download SRT file
                        const blob = new Blob([result.content], { type: 'text/plain' });
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = result.filename;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);
                        
                        alert(`SRT file downloaded: ${result.filename}`);
                    } else {
                        const error = await response.text();
                        alert(`Error downloading SRT: ${error}`);
                    }
                } catch (error) {
                    alert(`Error downloading SRT: ${error.message}`);
                }
            }
            
            async function downloadTranscript(fileId) {
                try {
                    const response = await fetch(`/api/transcript/${fileId}`);
                    
                    if (response.ok) {
                        const result = await response.json();
                        
                        // Create and download transcript file
                        const blob = new Blob([result.content], { type: 'text/plain' });
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = result.filename;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);
                        
                        alert(`Transcript file downloaded: ${result.filename}`);
                    } else {
                        const error = await response.text();
                        alert(`Error downloading transcript: ${error}`);
                    }
                } catch (error) {
                    alert(`Error downloading transcript: ${error.message}`);
                }
            }
            
            // Audio playback with synced transcription
            let currentAudio = null;
            let currentTranscription = null;
            
            async function playAudio(fileId) {
                try {
                    const response = await fetch(`/api/audio/${fileId}`);
                    
                    if (response.ok) {
                        const audioData = await response.json();
                        
                        // Get transcription if available
                        if (audioData.transcription_status === 'completed') {
                            currentTranscription = audioData.transcription;
                            showAudioPlayer(audioData);
                        } else {
                            alert('Audio transcription not available yet. Please transcribe first.');
                        }
                    } else {
                        const error = await response.text();
                        alert(`Error loading audio: ${error}`);
                    }
                } catch (error) {
                    alert(`Error loading audio: ${error.message}`);
                }
            }
            
            function showAudioPlayer(audioData) {
                // Create audio player modal
                const modal = document.createElement('div');
                modal.className = 'audio-player-modal';
                modal.innerHTML = `
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>🎵 ${audioData.filename}</h3>
                            <button class="close-btn" onclick="closeAudioPlayer()">&times;</button>
                        </div>
                        <div class="audio-controls">
                            <audio id="audioPlayer" controls>
                                <source src="/files/audio/${audioData.filename}" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                        </div>
                        <div class="transcription-display">
                            <h4>📝 Transcription</h4>
                            <div class="transcription-text" id="transcriptionText">
                                ${audioData.transcription || 'No transcription available'}
                            </div>
                        </div>
                        <div class="audio-actions">
                            <button onclick="downloadSRT(${audioData.id})" class="action-btn">📄 Download SRT</button>
                            <button onclick="downloadTranscript(${audioData.id})" class="action-btn">📝 Download Text</button>
                        </div>
                    </div>
                `;
                
                // Add CSS for modal
                const style = document.createElement('style');
                style.textContent = `
                    .audio-player-modal {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.8);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        z-index: 1000;
                    }
                    .modal-content {
                        background: white;
                        border-radius: 12px;
                        padding: 24px;
                        max-width: 600px;
                        width: 90%;
                        max-height: 80vh;
                        overflow-y: auto;
                    }
                    .modal-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 20px;
                        padding-bottom: 12px;
                        border-bottom: 1px solid #e0e0e0;
                    }
                    .close-btn {
                        background: none;
                        border: none;
                        font-size: 24px;
                        cursor: pointer;
                        color: #666;
                    }
                    .audio-controls {
                        margin-bottom: 20px;
                    }
                    .audio-controls audio {
                        width: 100%;
                    }
                    .transcription-display {
                        margin-bottom: 20px;
                    }
                    .transcription-display h4 {
                        margin-bottom: 12px;
                        color: #333;
                    }
                    .transcription-text {
                        background: #f8f9fa;
                        padding: 16px;
                        border-radius: 8px;
                        border: 1px solid #e0e0e0;
                        line-height: 1.6;
                        max-height: 200px;
                        overflow-y: auto;
                    }
                    .audio-actions {
                        display: flex;
                        gap: 12px;
                        justify-content: center;
                    }
                `;
                document.head.appendChild(style);
                document.body.appendChild(modal);
                
                // Setup audio event listeners for sync
                const audio = document.getElementById('audioPlayer');
                if (audio && currentTranscription) {
                    setupTranscriptionSync(audio, currentTranscription);
                }
            }
            
            function setupTranscriptionSync(audio, transcription) {
                // This is a simplified sync - in a real implementation, you'd parse SRT timestamps
                const transcriptionElement = document.getElementById('transcriptionText');
                
                audio.addEventListener('timeupdate', function() {
                    // Highlight current part of transcription based on time
                    // This is a basic implementation - you'd need SRT parsing for precise sync
                    const currentTime = audio.currentTime;
                    const duration = audio.duration;
                    const progress = currentTime / duration;
                    
                    // Simple word highlighting based on progress
                    if (transcriptionElement) {
                        const words = transcription.split(' ');
                        const currentWordIndex = Math.floor(progress * words.length);
                        
                        let highlightedText = '';
                        words.forEach((word, index) => {
                            if (index === currentWordIndex) {
                                highlightedText += `<span style="background: #ffeb3b; padding: 2px 4px; border-radius: 3px;">${word}</span> `;
                            } else {
                                highlightedText += word + ' ';
                            }
                        });
                        
                        transcriptionElement.innerHTML = highlightedText;
                    }
                });
            }
            
            function closeAudioPlayer() {
                const modal = document.querySelector('.audio-player-modal');
                if (modal) {
                    modal.remove();
                }
                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio = null;
                }
                currentTranscription = null;
            }
            
            // Document viewing
            async function viewDocument(fileId) {
                try {
                    const response = await fetch(`/api/scanned-document/${fileId}/view`);
                    
                    if (response.ok) {
                        const docData = await response.json();
                        showDocumentViewer(docData);
                    } else {
                        const error = await response.text();
                        alert(`Error loading document: ${error}`);
                    }
                } catch (error) {
                    alert(`Error loading document: ${error.message}`);
                }
            }
            
            function showDocumentViewer(docData) {
                // Create document viewer modal
                const modal = document.createElement('div');
                modal.className = 'document-viewer-modal';
                modal.innerHTML = `
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>📄 ${docData.filename}</h3>
                            <button class="close-btn" onclick="closeDocumentViewer()">&times;</button>
                        </div>
                        <div class="document-display">
                            <img src="/files/scanned/${docData.filename}" 
                                 alt="${docData.filename}" 
                                 style="max-width: 100%; height: auto; border-radius: 8px;">
                        </div>
                        <div class="document-actions">
                            <button onclick="downloadDocument(${docData.id})" class="action-btn">📥 Download</button>
                        </div>
                    </div>
                `;
                
                // Add CSS for modal
                const style = document.createElement('style');
                style.textContent = `
                    .document-viewer-modal {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.9);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        z-index: 1000;
                    }
                    .document-viewer-modal .modal-content {
                        background: white;
                        border-radius: 12px;
                        padding: 24px;
                        max-width: 90%;
                        max-height: 90%;
                        overflow: auto;
                    }
                    .document-display {
                        text-align: center;
                        margin: 20px 0;
                    }
                    .document-display img {
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    }
                `;
                document.head.appendChild(style);
                document.body.appendChild(modal);
            }
            
            function closeDocumentViewer() {
                const modal = document.querySelector('.document-viewer-modal');
                if (modal) {
                    modal.remove();
                }
            }
            
            async function downloadDocument(fileId) {
                try {
                    // Get file info from the files API
                    const response = await fetch('/api/files');
                    const data = await response.json();
                    const file = data.files.find(f => f.id === fileId && f.type === 'document');
                    
                    if (file) {
                        // Create download link for regular document
                        const a = document.createElement('a');
                        a.href = `/files/pictures/${file.name}`;
                        a.download = file.name;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        
                        alert(`Document downloaded: ${file.name}`);
                    } else {
                        alert('Document not found');
                    }
                } catch (error) {
                    alert(`Error downloading document: ${error.message}`);
                }
            }
            
            // Search functionality
            document.getElementById('searchInput').addEventListener('input', function(e) {
                const searchTerm = e.target.value.toLowerCase();
                const fileCards = document.querySelectorAll('.file-card');
                
                fileCards.forEach(card => {
                    const fileName = card.querySelector('.file-name').textContent.toLowerCase();
                    if (fileName.includes(searchTerm)) {
                        card.style.display = 'block';
                    } else {
                        card.style.display = 'none';
                    }
                });
            });
            
            function formatFileSize(bytes) {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            }
            
            async function scanDirectories() {
                try {
                    const response = await fetch('/api/scan', { method: 'POST' });
                    const result = await response.json();
                    alert(result.message);
                    loadFiles();
                } catch (error) {
                    console.error('Error scanning directories:', error);
                    alert('Error scanning directories');
                }
            }
            
            async function checkDirectories() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    document.getElementById('audio-status').textContent = 
                        `${data.audio_files} files in ${data.audio_dir}`;
                    document.getElementById('doc-status').textContent = 
                        `${data.document_files} files in ${data.pictures_dir}`;
                    document.getElementById('transcripts-status').textContent = 
                        `0 files in ${data.transcripts_dir}`;
                    document.getElementById('documents-status').textContent = 
                        `0 files in ${data.documents_dir}`;
                } catch (error) {
                    console.error('Error checking directories:', error);
                }
            }
            
            // Initialize
            connectWebSocket();
            loadFiles();
            checkDirectories();
            
            // Refresh every 30 seconds
            setInterval(() => {
                loadFiles();
                checkDirectories();
            }, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/files")
async def get_files():
    """Get all files from database"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get audio files
            cursor.execute('''
                SELECT id, filename, file_path, size, duration, created_at, processed, 
                       transcription_status, language
                FROM audio_files ORDER BY created_at DESC
            ''')
            audio_files = [dict(row) for row in cursor.fetchall()]
            
            # Get document files
            cursor.execute('''
                SELECT id, filename, file_path, size, width, height, created_at, processed
                FROM documents ORDER BY created_at DESC
            ''')
            document_files = [dict(row) for row in cursor.fetchall()]
            
            # Combine and format files
            all_files = []
            for file in audio_files:
                all_files.append({
                    'id': file['id'],
                    'name': file['filename'],
                    'type': 'audio',
                    'size': file['size'],
                    'duration': file['duration'],
                    'created_at': file['created_at'],
                    'processed': file['processed'],
                    'transcription_status': file['transcription_status'],
                    'language': file['language']
                })
            
            for file in document_files:
                all_files.append({
                    'id': file['id'],
                    'name': file['filename'],
                    'type': 'document',
                    'size': file['size'],
                    'width': file['width'],
                    'height': file['height'],
                    'created_at': file['created_at'],
                    'processed': file['processed']
                })
            
            return {"files": all_files}
            
    except Exception as e:
        logger.error(f"Error getting files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scanned-documents")
async def get_scanned_documents():
    """Get all scanned documents"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, original_filename, scanned_filename, scanned_path, 
                       original_size, scanned_size, corners_found, created_at
                FROM scanned_documents ORDER BY created_at DESC
            ''')
            scanned_docs = [dict(row) for row in cursor.fetchall()]
            return {"scanned_documents": scanned_docs}
    except Exception as e:
        logger.error(f"Error getting scanned documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scanned-document/{doc_id}")
async def get_scanned_document(doc_id: int):
    """Get specific scanned document"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM scanned_documents WHERE id = ?
            ''', (doc_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Scanned document not found")
            
            return dict(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scanned document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio/{file_id}")
async def get_audio_file(file_id: int):
    """Get audio file with transcription for playback"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, file_path, transcription, srt_content, 
                       duration, transcription_status, language
                FROM audio_files WHERE id = ?
            ''', (file_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            return dict(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audio file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio/{file_id}/play")
async def play_audio_file(file_id: int):
    """Stream audio file for playback"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT file_path FROM audio_files WHERE id = ?', (file_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            file_path = Path(result['file_path'])
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="Audio file not found on disk")
            
            # Return file path for streaming
            return {"file_path": str(file_path), "filename": file_path.name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting audio file for playback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{file_id}")
async def delete_file(file_id: int):
    """Delete a file from the system"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get file info first
            cursor.execute('SELECT * FROM audio_files WHERE id = ?', (file_id,))
            audio_file = cursor.fetchone()
            
            if audio_file:
                # Delete audio file
                file_path = Path(audio_file['file_path'])
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted audio file: {file_path}")
                
                cursor.execute('DELETE FROM audio_files WHERE id = ?', (file_id,))
                conn.commit()
                
                await broadcast_message(json.dumps({"type": "file_update", "message": f"Audio file deleted: {audio_file['filename']}"}))
                return {"message": f"Audio file deleted: {audio_file['filename']}"}
            
            # Check documents table
            cursor.execute('SELECT * FROM documents WHERE id = ?', (file_id,))
            document_file = cursor.fetchone()
            
            if document_file:
                # Delete document file
                file_path = Path(document_file['file_path'])
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted document file: {file_path}")
                
                cursor.execute('DELETE FROM documents WHERE id = ?', (file_id,))
                conn.commit()
                
                await broadcast_message(json.dumps({"type": "file_update", "message": f"Document file deleted: {document_file['filename']}"}))
                return {"message": f"Document file deleted: {document_file['filename']}"}
            
            # Check scanned documents table
            cursor.execute('SELECT * FROM scanned_documents WHERE id = ?', (file_id,))
            scanned_doc = cursor.fetchone()
            
            if scanned_doc:
                # Delete scanned document file
                scanned_path = Path(scanned_doc['scanned_path'])
                if scanned_path.exists():
                    scanned_path.unlink()
                    logger.info(f"Deleted scanned document: {scanned_path}")
                
                cursor.execute('DELETE FROM scanned_documents WHERE id = ?', (file_id,))
                conn.commit()
                
                await broadcast_message(json.dumps({"type": "file_update", "message": f"Scanned document deleted: {scanned_doc['scanned_filename']}"}))
                return {"message": f"Scanned document deleted: {scanned_doc['scanned_filename']}"}
            
            raise HTTPException(status_code=404, detail="File not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scanned-document/{doc_id}/view")
async def view_scanned_document(doc_id: int):
    """Get scanned document image for viewing"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT scanned_path FROM scanned_documents WHERE id = ?', (doc_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Scanned document not found")
            
            file_path = Path(result['scanned_path'])
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="Scanned document not found on disk")
            
            return {"file_path": str(file_path), "filename": file_path.name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scanned document for viewing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/scanned/{filename}")
async def serve_scanned_file(filename: str):
    """Serve scanned document files"""
    try:
        file_path = config.scanned_docs_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(str(file_path), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error serving scanned file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/audio/{filename}")
async def serve_audio_file(filename: str):
    """Serve audio files"""
    try:
        file_path = config.audio_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(str(file_path), media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Error serving audio file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files/pictures/{filename}")
async def serve_picture_file(filename: str):
    """Serve picture files"""
    try:
        file_path = config.pictures_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(str(file_path), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error serving picture file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """Get system status"""
    try:
        audio_files = list(config.audio_dir.glob("*")) if config.audio_dir.exists() else []
        document_files = list(config.pictures_dir.glob("*")) if config.pictures_dir.exists() else []
        
        return {
            "audio_dir": str(config.audio_dir),
            "pictures_dir": str(config.pictures_dir),
            "transcripts_dir": str(config.transcripts_dir),
            "documents_dir": str(config.documents_dir),
            "audio_files": len(audio_files),
            "document_files": len(document_files),
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/scan")
async def scan_directories():
    """Scan directories for new files"""
    try:
        new_files = 0
        
        # Scan audio directory
        if config.audio_dir.exists():
            for file_path in config.audio_dir.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
                    # Check if file already exists in database
                    with db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('SELECT id FROM audio_files WHERE file_path = ?', (str(file_path),))
                        if not cursor.fetchone():
                            # Check if it's a WAV file that needs conversion
                            if file_path.suffix.lower() == '.wav':
                                logger.info(f"Detected new WAV file: {file_path}")
                                
                                # Convert WAV to MP3
                                mp3_file = file_path.with_suffix('.mp3')
                                conversion_success = False
                                
                                try:
                                    if AUDIO_PROCESSING_AVAILABLE:
                                        logger.info(f"Converting {file_path.name} to MP3...")
                                        audio = AudioSegment.from_wav(str(file_path))
                                        audio.export(str(mp3_file), format="mp3", bitrate="128k")
                                        
                                        if mp3_file.exists() and mp3_file.stat().st_size > 0:
                                            logger.info(f"✅ MP3 conversion successful: {mp3_file.name}")
                                            
                                            # Delete original WAV file
                                            file_path.unlink()
                                            logger.info(f"🗑️ Deleted original WAV file: {file_path.name}")
                                            
                                            # Update file_path to point to MP3
                                            file_path = mp3_file
                                            conversion_success = True
                                        else:
                                            logger.warning(f"⚠️ MP3 conversion failed - keeping WAV file")
                                    else:
                                        logger.warning(f"⚠️ Audio processing not available - keeping WAV file")
                                        
                                except Exception as e:
                                    logger.error(f"❌ MP3 conversion failed: {e}")
                                    logger.info(f"📁 Keeping original WAV file: {file_path.name}")
                            
                            # Add to database (either original WAV or converted MP3)
                            cursor.execute('''
                                INSERT INTO audio_files (filename, file_path, size, created_at)
                                VALUES (?, ?, ?, ?)
                            ''', (file_path.name, str(file_path), file_path.stat().st_size, datetime.now()))
                            conn.commit()
                            new_files += 1
                            logger.info(f"Added new audio file: {file_path}")
        
        # Scan documents directory
        if config.pictures_dir.exists():
            for file_path in config.pictures_dir.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf']:
                    # Check if file already exists in database
                    with db_manager.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('SELECT id FROM documents WHERE file_path = ?', (str(file_path),))
                        if not cursor.fetchone():
                            # Process document with CamScanner-like functionality
                            logger.info(f"Processing document: {file_path.name}")
                            
                            try:
                                if DOCUMENT_PROCESSING_AVAILABLE:
                                    result = document_processor.process_document(file_path)
                                    
                                    if result["success"]:
                                        # Add original to documents table
                                        cursor.execute('''
                                            INSERT INTO documents (filename, file_path, size, width, height, created_at, processed)
                                            VALUES (?, ?, ?, ?, ?, ?, ?)
                                        ''', (file_path.name, str(file_path), file_path.stat().st_size, 
                                             result["original_size"][1], result["original_size"][0], datetime.now(), True))
                                        
                                        # Add scanned document to scanned_documents table
                                        scanned_path = Path(result["processed_path"])
                                        cursor.execute('''
                                            INSERT INTO scanned_documents 
                                            (original_filename, original_path, scanned_filename, scanned_path, 
                                             original_size, scanned_size, corners_found, created_at)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (file_path.name, str(file_path), scanned_path.name, str(scanned_path),
                                             f"{result['original_size'][1]}x{result['original_size'][0]}",
                                             f"{result['processed_size'][1]}x{result['processed_size'][0]}",
                                             result["corners_found"], datetime.now()))
                                        
                                        conn.commit()
                                        new_files += 1
                                        logger.info(f"✅ Document processed and saved: {scanned_path.name}")
                                    else:
                                        logger.error(f"❌ Document processing failed: {result.get('error', 'Unknown error')}")
                                        # Still add to database as unprocessed
                                        cursor.execute('''
                                            INSERT INTO documents (filename, file_path, size, created_at, processed)
                                            VALUES (?, ?, ?, ?, ?)
                                        ''', (file_path.name, str(file_path), file_path.stat().st_size, datetime.now(), False))
                                        conn.commit()
                                        new_files += 1
                                        logger.info(f"Added unprocessed document: {file_path.name}")
                                else:
                                    logger.warning(f"⚠️ Document processing not available - adding as-is")
                                    cursor.execute('''
                                        INSERT INTO documents (filename, file_path, size, created_at, processed)
                                        VALUES (?, ?, ?, ?, ?)
                                    ''', (file_path.name, str(file_path), file_path.stat().st_size, datetime.now(), False))
                                    conn.commit()
                                    new_files += 1
                                    logger.info(f"Added document (no processing): {file_path.name}")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error processing document {file_path.name}: {e}")
                                # Add to database as unprocessed
                                cursor.execute('''
                                    INSERT INTO documents (filename, file_path, size, created_at, processed)
                                    VALUES (?, ?, ?, ?, ?)
                                ''', (file_path.name, str(file_path), file_path.stat().st_size, datetime.now(), False))
                                conn.commit()
                                new_files += 1
                                logger.info(f"Added document (processing failed): {file_path.name}")
        
        await broadcast_message(f"Scan completed: {new_files} new files found")
        
        return {"message": f"Scan completed: {new_files} new files found"}
        
    except Exception as e:
        logger.error(f"Error scanning directories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/convert-audio/{file_id}")
async def convert_audio_file(file_id: int):
    """Convert an existing WAV file to MP3"""
    try:
        if not audio_converter.available:
            raise HTTPException(status_code=400, detail="Audio conversion not available")
        
        # Get file info from database
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT filename, file_path FROM audio_files WHERE id = ?', (file_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="File not found")
            
            filename, file_path = result
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File not found on disk")
            
            if not filename.lower().endswith('.wav'):
                raise HTTPException(status_code=400, detail="File is not a WAV file")
        
        # Convert to MP3
        mp3_path = audio_converter.convert_wav_to_mp3(file_path)
        
        # Get audio info
        audio_info = audio_converter.get_audio_info(mp3_path)
        
        # Update database with new file info
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE audio_files 
                SET filename = ?, file_path = ?, size = ?, duration = ?, channels = ?, sample_rate = ?
                WHERE id = ?
            ''', (
                mp3_path.name,
                str(mp3_path),
                mp3_path.stat().st_size,
                audio_info.get('duration', 0),
                audio_info.get('channels', 0),
                audio_info.get('sample_rate', 0),
                file_id
            ))
            conn.commit()
        
        # Remove original WAV file
        file_path.unlink()
        
        await broadcast_message(f"Converted {filename} to MP3")
        
        return {
            "message": f"Successfully converted {filename} to MP3",
            "new_filename": mp3_path.name,
            "audio_info": audio_info
        }
        
    except Exception as e:
        logger.error(f"Error converting audio file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def transcribe_audio_background(file_id: int, file_path: Path):
    """Transcribe audio file in background"""
    try:
        logger.info(f"Starting background transcription for file ID {file_id}")
        
        # Update status to processing
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE audio_files 
                SET transcription_status = 'processing'
                WHERE id = ?
            ''', (file_id,))
            conn.commit()
        
        await broadcast_message(f"Transcribing audio file ID {file_id}...")
        
        # Transcribe the audio
        result = transcription_service.transcribe_audio(file_path)
        
        # Save transcription as text file
        base_name = Path(file_path).stem
        transcript_filename = f"{base_name}_transcript.txt"
        transcript_path = config.transcripts_dir / transcript_filename
        
        try:
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            logger.info(f"✅ Transcription saved to: {transcript_path}")
        except Exception as e:
            logger.error(f"Failed to save transcription file: {e}")
        
        # Update database with transcription
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE audio_files 
                SET transcription = ?, srt_content = ?, transcription_status = 'completed', language = ?
                WHERE id = ?
            ''', (result['text'], result['srt'], result['language'], file_id))
            conn.commit()
        
        logger.info(f"✅ Transcription completed for file ID {file_id}")
        await broadcast_message(f"Transcription completed for file ID {file_id}")
        
    except Exception as e:
        logger.error(f"Error transcribing audio file ID {file_id}: {e}")
        
        # Update status to failed
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE audio_files 
                SET transcription_status = 'failed'
                WHERE id = ?
            ''', (file_id,))
            conn.commit()
        
        await broadcast_message(f"Transcription failed for file ID {file_id}")

@app.get("/api/transcription/{file_id}")
async def get_transcription(file_id: int):
    """Get transcription for an audio file"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT filename, transcription, srt_content, transcription_status, language, duration
                FROM audio_files 
                WHERE id = ?
            ''', (file_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="File not found")
            
            filename, transcription, srt_content, status, language, duration = result
            
            return {
                "file_id": file_id,
                "filename": filename,
                "transcription": transcription,
                "srt": srt_content,
                "status": status,
                "language": language,
                "duration": duration
            }
            
    except Exception as e:
        logger.error(f"Error getting transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/srt/{file_id}")
async def get_srt_file(file_id: int):
    """Download SRT file for an audio file"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT filename, srt_content, transcription_status
                FROM audio_files 
                WHERE id = ?
            ''', (file_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="File not found")
            
            filename, srt_content, status = result
            
            if status != 'completed':
                raise HTTPException(status_code=400, detail="Transcription not completed")
            
            if not srt_content:
                raise HTTPException(status_code=404, detail="SRT content not available")
            
            # Generate SRT filename
            base_name = Path(filename).stem
            srt_filename = f"{base_name}.srt"
            
            return {
                "filename": srt_filename,
                "content": srt_content,
                "content_type": "text/plain"
            }
            
    except Exception as e:
        logger.error(f"Error getting SRT file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transcript/{file_id}")
async def get_transcript_file(file_id: int):
    """Download transcript text file for an audio file"""
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT filename, transcription, transcription_status
                FROM audio_files 
                WHERE id = ?
            ''', (file_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="File not found")
            
            filename, transcription, status = result
            
            if status != 'completed':
                raise HTTPException(status_code=400, detail="Transcription not completed")
            
            if not transcription:
                raise HTTPException(status_code=404, detail="Transcription not available")
            
            # Generate transcript filename
            base_name = Path(filename).stem
            transcript_filename = f"{base_name}_transcript.txt"
            
            return {
                "filename": transcript_filename,
                "content": transcription,
                "content_type": "text/plain"
            }
            
    except Exception as e:
        logger.error(f"Error getting transcript file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe/{file_id}")
async def transcribe_file(file_id: int):
    """Manually transcribe an audio file"""
    try:
        if not transcription_service.available:
            raise HTTPException(status_code=400, detail="Transcription not available")
        
        # Get file info from database
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT filename, file_path FROM audio_files WHERE id = ?', (file_id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="File not found")
            
            filename, file_path = result
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Start transcription in background
        asyncio.create_task(transcribe_audio_background(file_id, file_path))
        
        return {
            "message": f"Transcription started for {filename}",
            "file_id": file_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and convert WAV to MP3 if needed"""
    try:
        # Determine file type and directory
        if file.content_type and file.content_type.startswith('audio/'):
            upload_dir = config.audio_dir
            file_type = 'audio'
        elif file.content_type and file.content_type.startswith('image/'):
            upload_dir = config.pictures_dir
            file_type = 'document'
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Save original file
        original_file_path = upload_dir / file.filename
        with open(original_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process audio files
        final_file_path = original_file_path
        file_size = len(content)
        
        if file_type == 'audio' and file.filename.lower().endswith('.wav'):
            if audio_converter.available:
                try:
                    # Convert WAV to MP3
                    mp3_path = audio_converter.convert_wav_to_mp3(original_file_path)
                    
                    # Get MP3 file size
                    file_size = mp3_path.stat().st_size
                    final_file_path = mp3_path
                    
                    # Get audio info
                    audio_info = audio_converter.get_audio_info(mp3_path)
                    
                    # Remove original WAV file
                    original_file_path.unlink()
                    
                    logger.info(f"Converted {file.filename} to MP3: {mp3_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to convert WAV to MP3: {e}")
                    # Keep original file if conversion fails
                    final_file_path = original_file_path
            else:
                logger.warning("Audio conversion not available - keeping original WAV file")
        
        # Add to database
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            if file_type == 'audio':
                # Get audio info if available
                audio_info = audio_converter.get_audio_info(final_file_path) if audio_converter.available else {}
                
                cursor.execute('''
                    INSERT INTO audio_files (filename, file_path, size, created_at, duration, channels, sample_rate, transcription_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    final_file_path.name, 
                    str(final_file_path), 
                    file_size, 
                    datetime.now(),
                    audio_info.get('duration', 0),
                    audio_info.get('channels', 0),
                    audio_info.get('sample_rate', 0),
                    'pending'
                ))
                
                # Get the file ID for transcription
                file_id = cursor.lastrowid
                
            else:
                cursor.execute('''
                    INSERT INTO documents (filename, file_path, size, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (final_file_path.name, str(final_file_path), file_size, datetime.now()))
            conn.commit()
        
        # Start transcription in background if it's an audio file
        if file_type == 'audio' and transcription_service.available:
            asyncio.create_task(transcribe_audio_background(file_id, final_file_path))
        
        await broadcast_message(f"File uploaded: {final_file_path.name}")
        
        return {
            "message": f"File {final_file_path.name} uploaded successfully",
            "converted": final_file_path != original_file_path,
            "original_format": "WAV" if file.filename.lower().endswith('.wav') else "Other",
            "final_format": "MP3" if final_file_path.suffix.lower() == '.mp3' else "Original"
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Smart Glasses Dock Application")
    logger.info(f"Audio directory: {config.audio_dir}")
    logger.info(f"Pictures directory: {config.pictures_dir}")
    logger.info(f"Transcripts directory: {config.transcripts_dir}")
    logger.info(f"Documents directory: {config.documents_dir}")
    logger.info(f"Scanned documents directory: {config.scanned_docs_dir}")
    logger.info(f"Database: {config.db_path}")
    
    if audio_converter.available:
        logger.info("✅ Audio conversion (WAV to MP3) is available")
    else:
        logger.warning("❌ Audio conversion not available - install pydub for WAV to MP3 conversion")
    
    if transcription_service.available:
        logger.info("✅ Audio transcription (Whisper) is available")
    else:
        logger.warning("❌ Audio transcription not available - install whisper for transcription")
    
    if document_processor.available:
        logger.info("✅ Document processing (CamScanner-like) is available")
    else:
        logger.warning("❌ Document processing not available - install opencv-python for document scanning")
    
    print("\n" + "="*60)
    print("🎉 Smart Glasses Dock Application Started!")
    print("="*60)
    print(f"🌐 Web Interface: http://localhost:8007")
    print(f"📱 API Endpoint:  http://localhost:8007/api")
    print(f"🔌 WebSocket:     ws://localhost:8007/ws")
    if audio_converter.available:
        print("🎵 Audio Conversion: WAV → MP3 (Automatic)")
    else:
        print("🎵 Audio Conversion: Not Available")
    if transcription_service.available:
        print("📝 Audio Transcription: Whisper (Automatic)")
    else:
        print("📝 Audio Transcription: Not Available")
    if document_processor.available:
        print("📄 Document Processing: CamScanner-like (Automatic)")
    else:
        print("📄 Document Processing: Not Available")
    print("="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8007)
