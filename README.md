# Smart Glasses System - Best Document Detection

## 🚀 Quick Start

```bash
python start.py
```

## 🌐 Access

Open your browser to: **http://localhost:8006**

## ✨ Features

- **📄 Advanced Document Detection**: Uses OpenCV to detect rectangular documents
- **🚫 No People Detection**: Completely ignores people - only detects documents
- **🎤 Voice Detection**: Automatically increases audio quality when speech detected
- **📸 Auto-Capture**: Automatically captures documents when detected
- **🔊 Audio Recording**: Continuous audio recording with quality adjustment
- **📺 Live Camera Feed**: Real-time camera feed with detection overlay

## 🎯 What It Detects

- ✅ Papers and documents (rectangular shapes)
- ✅ Books (rectangular with good aspect ratio)
- ✅ Screens and monitors (rectangular displays)
- ✅ Tablets and phones (when held flat)

## 🚫 What It Won't Detect

- ❌ People (completely ignored)
- ❌ Furniture (chairs, tables, etc.)
- ❌ Irregular shapes (non-rectangular objects)
- ❌ Small objects (below size threshold)

## 📁 File Structure

```
prototype/
├── smart_glasses_best.py    # Main system file
├── start.py                 # Startup script
├── requirements.txt         # Dependencies
└── data/                   # Generated data
    ├── audio/              # Audio recordings
    ├── video/              # Video recordings
    └── documents/          # Captured documents
```

## 🔧 Requirements

- Python 3.7+
- Webcam
- Microphone
- Modern web browser

## 📦 Dependencies

- FastAPI (web framework)
- OpenCV (computer vision)
- PyAudio (audio processing)
- NumPy (numerical computing)
- Uvicorn (web server)

## 🎮 Usage

1. Run `python start.py`
2. Open http://localhost:8006 in your browser
3. Show documents to the camera
4. System will automatically detect and capture them
5. Speak to trigger higher audio quality

## 🛠️ Technical Details

- **Document Detection**: Advanced OpenCV with adaptive thresholding, morphological operations, and rectangularity analysis
- **Voice Detection**: RMS + Zero-crossing rate analysis
- **Web Interface**: Modern HTML5/CSS3/JavaScript with WebSocket communication
- **Real-time Processing**: 30 FPS camera processing with 0.5s detection intervals
- **Auto-capture Delay**: 3 seconds between captures to prevent spam

## 🎉 Success!

The system uses pure computer vision to detect rectangular document shapes and will never capture photos of people!