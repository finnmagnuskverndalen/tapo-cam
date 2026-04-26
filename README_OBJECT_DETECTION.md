# Tapo C200 Camera with Object Detection

A Rust application for viewing and controlling a TP-Link Tapo C200 IP camera with real-time object detection capabilities.

## Features

### Core Features
- **Live RTSP Video Streaming**: View camera feed in real-time
- **Object Detection**: Detect humans, animals, and more using OpenCV Haar cascades
- **PTZ Control**: Pan/Tilt/Zoom control via arrow keys (if authentication works)
- **Local HTTP API**: Direct communication with camera without cloud dependency

### Object Detection Capabilities
- **Human Face Detection**: Using frontal face cascade
- **Full Body Detection**: Using full body cascade  
- **Upper Body Detection**: Using upper body cascade
- **Animal Detection**: Cat face detection (if cascade files available)
- **Real-time Bounding Boxes**: Objects are outlined with color-coded boxes
- **Statistics Display**: Frame count and object counts shown on screen

## Requirements

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libopencv-dev libclang-dev clang opencv-data

# Install OpenCV data files for cascade classifiers
sudo apt-get install -y opencv-data
```

### Rust Toolchain
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Setup

1. **Clone and enter the repository**
   ```bash
   cd tapo-cam
   ```

2. **Configure camera credentials**
   ```bash
   echo "TAPO_EMAIL=your_email@example.com" > .env
   echo "TAPO_PASSWORD=your_password" >> .env
   ```

3. **Build the application**
   ```bash
   ./build.sh
   ```
   
   Or manually:
   ```bash
   # Use object detection version
   cp src/main_object_detection.rs src/main.rs
   cargo build --release
   ```

## Usage

### Basic Usage
```bash
./target/release/tapo-cam
```

### Controls
- **Q** or **ESC**: Quit application
- **H**: Calibrate/return camera to home position (if PTZ enabled)
- **Arrow Keys**: Pan/Tilt control (if PTZ enabled)

### Modes
1. **With Authentication**: Full PTZ controls + object detection
2. **Without Authentication**: Object detection only (video streaming still works)

## Object Detection Details

### Detection Methods
The application uses OpenCV's pre-trained Haar cascade classifiers:
- **Human faces**: `haarcascade_frontalface_default.xml`
- **Full bodies**: `haarcascade_fullbody.xml`  
- **Upper bodies**: `haarcascade_upperbody.xml`
- **Cat faces**: `haarcascade_frontalcatface.xml`

### Performance Optimization
- Detection runs every 3rd frame to reduce CPU load
- Grayscale conversion and histogram equalization for better detection
- Configurable detection parameters (scale factor, min neighbors)

### Visualization
- **Green boxes**: Humans
- **Blue boxes**: Animals
- **Labels**: Object type with confidence percentage
- **Statistics**: Frame count and object counts in top-left corner

## Troubleshooting

### Common Issues

1. **"Could not find OpenCV cascade files"**
   ```bash
   sudo apt-get install -y opencv-data
   ```

2. **Authentication fails but video works**
   - Check `.env` file credentials
   - Camera may not be linked to your Tapo account
   - PTZ controls will be disabled, but object detection still works

3. **High CPU usage**
   - Detection runs every 3rd frame by default
   - Can adjust detection frequency in code

4. **Poor detection accuracy**
   - Ensure good lighting conditions
   - Objects should be clearly visible
   - Adjust detection parameters in code if needed

## Project Structure

```
tapo-cam/
├── Cargo.toml              # Rust dependencies
├── src/
│   ├── main.rs                    # Main application (current)
│   ├── main_object_detection.rs   # Object detection version
│   └── camera.rs                  # Camera API client
├── .env                    # Camera credentials (gitignored)
├── build.sh               # Build script
└── README_OBJECT_DETECTION.md
```

## Development

### Switching Versions
```bash
# Use object detection version
cp src/main_object_detection.rs src/main.rs
cargo build --release

# Use original version (without object detection)
cp src/main_backup.rs src/main.rs  # if you have backup
# or check git history for original
```

### Extending Detection
To add more object types:
1. Add new cascade classifier files to system
2. Update `ObjectDetector::new()` to load new cascade
3. Add detection logic in `detect_objects()` method
4. Update `ObjectType` enum and visualization colors

### Performance Tuning
Key parameters in `ObjectDetector::detect_objects()`:
- `scaleFactor`: Detection scale (default: 1.05-1.1)
- `minNeighbors`: Minimum neighbors for detection (default: 3)
- `minSize`: Minimum object size (default: 30x30)
- Detection frequency (currently every 3rd frame)

## Security Notes

- Credentials stored in `.env` file (in `.gitignore`)
- No hardcoded credentials in source code
- Local network communication only
- Self-signed certificate acceptance for local camera
- Change camera password if credentials were ever exposed

## License

MIT License - See LICENSE file for details

## Acknowledgments

- OpenCV for computer vision libraries
- TP-Link for Tapo camera API (reverse engineered)
- Rust community for excellent crates ecosystem