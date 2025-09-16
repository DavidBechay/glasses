# Smart Glasses System - Complete Simulation

A comprehensive Smart Glasses system simulation built with PySimpleGUI, featuring both a Dock Station control interface and a Glasses Simulation interface.

## 🚀 Features

### Dock Station Interface
- **System Status Monitoring**: Real-time status of dock, glasses, audio, and video
- **Audio Recording**: Start/stop audio recording with level visualization
- **Video Recording**: Start/stop video recording with live camera feed
- **Camera Management**: Initialize and control camera functionality
- **File Management**: Open audio/video folders, clear old files
- **System Logging**: Real-time system log with save/clear functionality
- **System Information**: Display system specs and file counts

### Glasses Simulation Interface
- **Connection Management**: Connect/disconnect from dock station
- **Recording Controls**: Start/stop recording with dock integration
- **Live Camera Feed**: Real-time camera view simulation
- **Computer Vision Features**: Object detection, text recognition, face detection, scene analysis
- **Audio Controls**: Audio recording with level monitoring
- **Settings Management**: Brightness, volume, display mode, power save
- **Status Display**: Real-time status updates and notifications

## 📋 Requirements

- Python 3.7+
- PySimpleGUI
- OpenCV
- NumPy
- PyAudio
- Standard Python libraries (threading, socket, json, etc.)

## 🛠️ Installation

1. **Clone or download the project files**
2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Option 1: Start Everything at Once
```bash
python start_system.py
```
This will:
- Install required packages
- Create data directories
- Start both Dock Station and Glasses Simulation
- Provide system management

### Option 2: Start Individual Components
```bash
# Start Dock Station only
python dock_station.py

# Start Glasses Simulation only
python glasses_simulation.py
```

## 📁 Project Structure

```
smart-glasses-system/
├── config.py                 # System configuration and settings
├── audio_processor.py        # Audio recording and processing
├── camera_processor.py       # Camera capture and computer vision
├── dock_station.py          # Dock Station GUI (main control)
├── glasses_simulation.py    # Glasses Simulation GUI
├── start_system.py          # Main startup script
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── data/                   # Data storage directory
    ├── audio/              # Audio recordings (.wav files)
    ├── video/              # Video recordings (.mp4 files)
    └── logs/               # System logs
```

## 🎮 How to Use

### 1. Start the System
Run `python start_system.py` to launch both interfaces.

### 2. Dock Station Operations
- **Initialize Camera**: Click "Initialize Camera" to set up camera
- **Start Recording**: Use "Start Audio Recording" and "Start Video Recording"
- **Monitor Status**: Watch real-time status indicators
- **File Management**: Open folders or clear old files
- **System Log**: Monitor system activity in real-time

### 3. Glasses Simulation Operations
- **Connect to Dock**: Click "Connect to Dock" to establish connection
- **Start Recording**: Use recording controls (requires dock connection)
- **Computer Vision**: Test object detection, text recognition, etc.
- **Settings**: Adjust brightness, volume, display mode
- **Status Monitoring**: Watch battery level and connection status

### 4. Integration
- Connect glasses to dock for full functionality
- Recording from glasses is saved to dock station
- Real-time communication between interfaces

## 🔧 Configuration

Edit `config.py` to modify:
- Audio settings (sample rate, chunk size)
- Video settings (resolution, FPS)
- Communication ports
- GUI theme and window size
- File management settings

## 📊 System Features

### Audio Processing
- Real-time audio recording
- Audio level visualization
- WAV file output
- Background recording threads

### Video Processing
- Live camera feed
- Video recording with timestamps
- Computer vision processing (edge detection)
- Object detection simulation
- MP4 file output

### Computer Vision
- Edge detection and enhancement
- Object detection simulation
- Frame processing with timestamps
- Real-time visualization

### File Management
- Automatic file naming with timestamps
- Organized storage in data directories
- Old file cleanup functionality
- File count monitoring

## 🐛 Troubleshooting

### Common Issues

1. **Camera not working**:
   - Check camera permissions
   - Ensure no other applications are using the camera
   - Try "Initialize Camera" button

2. **Audio not recording**:
   - Check microphone permissions
   - Ensure PyAudio is properly installed
   - Check audio device availability

3. **GUI not responding**:
   - Check system resources
   - Restart the application
   - Check for error messages in logs

### Error Messages
- All errors are logged in the system log
- Check the status display for real-time error information
- Save logs for debugging purposes

## 🔄 Updates and Maintenance

### Automatic Cleanup
- Old files are automatically cleaned based on settings
- Log files can be saved and cleared
- System info shows current file counts

### Performance Monitoring
- Real-time status indicators
- Battery level simulation
- Connection status monitoring

## 📝 Development Notes

### Architecture
- Modular design with separate processors
- Threaded GUI updates for real-time performance
- Centralized configuration management
- Clean separation of concerns

### Extensibility
- Easy to add new computer vision features
- Modular audio/video processing
- Configurable system settings
- Pluggable GUI components

## 🎯 Future Enhancements

- Network communication between dock and glasses
- Advanced computer vision algorithms
- Machine learning integration
- Cloud storage integration
- Mobile app companion

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to modify and extend the system for your needs!

---

**Smart Glasses System** - Complete simulation with PySimpleGUI interfaces
