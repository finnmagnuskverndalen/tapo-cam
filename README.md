# Tapo C200 Camera Control with Object Detection

A modern Rust application for controlling TP-Link Tapo C200 IP cameras with real-time object detection capabilities.

![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## ✨ Features

### 🎥 **Core Camera Features**
- **Live RTSP Video Streaming** - Real-time camera feed display
- **PTZ Control** - Pan/Tilt/Zoom via keyboard controls
- **Local HTTP API** - Direct communication without cloud dependency
- **Multiple Authentication Methods** - Supports various Tapo firmware versions

### 🤖 **Object Detection**
- **Human Face Detection** - Using OpenCV Haar cascades
- **Real-time Visualization** - Color-coded bounding boxes
- **Performance Optimized** - Configurable detection frequency
- **Statistics Display** - Frame count and detection metrics

### 🔒 **Security Features**
- **Secure Credential Storage** - Environment variables only
- **No Hardcoded Credentials** - Safe for public repositories
- **Local Network Focus** - Self-signed certificate handling

## 🚀 Quick Start

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y libopencv-dev libclang-dev clang opencv-data

# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Installation
```bash
# Clone and setup
git clone https://github.com/yourusername/tapo-cam.git
cd tapo-cam

# Configure credentials (creates .env file)
echo "TAPO_EMAIL=your_email@example.com" > .env
echo "TAPO_PASSWORD=your_password" >> .env

# Build the application
cargo build --release
```

### Running
```bash
# Run the application
./target/release/tapo-cam
```

## 🎮 Controls

| Key | Action | Description |
|-----|--------|-------------|
| **Q** or **ESC** | Quit | Exit the application |
| **H** | Calibrate | Return camera to home position |
| **← →** | Pan | Move camera left/right |
| **↑ ↓** | Tilt | Move camera up/down |

## 📁 Project Structure

```
tapo-cam/
├── src/
│   ├── main.rs                    # Main application with object detection
│   ├── camera.rs                  # Camera API client
│   ├── main_simple_detection.rs   # Simple face detection version
│   └── main_object_detection.rs   # Full object detection version
├── Cargo.toml                     # Rust dependencies
├── Cargo.lock                     # Dependency locks
├── README.md                      # This documentation
├── .gitignore                     # Git ignore rules
├── build.sh                       # Build automation script
├── switch_version.sh              # Version switcher
└── .env                           # Camera credentials (not in git)
```

## 🔧 Configuration

### Camera Settings
Create a `.env` file in the project root:
```env
TAPO_EMAIL=your_email@example.com
TAPO_PASSWORD=your_password
```

### Camera IP Address
Edit `src/main.rs` to match your camera's IP:
```rust
const CAM_IP: &str = "192.168.10.185";  // Change to your camera's IP
```

## 🔍 Object Detection

### How It Works
1. **Frame Capture**: RTSP stream captured using OpenCV
2. **Preprocessing**: Grayscale conversion and histogram equalization
3. **Detection**: Haar cascade classifiers scan for features
4. **Visualization**: Bounding boxes and labels added
5. **Display**: Processed frame shown in window

### Supported Detectors
- **Face Detection**: `haarcascade_frontalface_default.xml`
- **Full Body Detection**: `haarcascade_fullbody.xml`
- **Upper Body Detection**: `haarcascade_upperbody.xml`
- **Animal Detection**: `haarcascade_frontalcatface.xml`

## 🛠️ Development

### Building
```bash
# Development build
cargo build

# Release build (recommended)
cargo build --release

# Clean build
cargo clean && cargo build --release
```

### Version Management
Switch between different application versions:
```bash
./switch_version.sh simple-detection    # Face detection (default)
./switch_version.sh full-detection      # Full object detection
./switch_version.sh original            # Basic camera control
./switch_version.sh list                # List available versions
```

### Testing
```bash
# Run unit tests
cargo test

# Check for errors
cargo check

# Format code
cargo fmt
```

## 🐛 Troubleshooting

### Common Issues

#### "Could not find OpenCV cascade files"
```bash
sudo apt install -y opencv-data
```

#### Authentication fails
- Verify credentials in `.env` file
- Check camera is linked to Tapo account
- Ensure camera is on same network

#### Video doesn't stream
- Confirm RTSP URL format
- Check network connectivity
- Verify camera supports RTSP

#### High CPU usage
- Adjust detection frequency in code
- Reduce frame resolution if possible
- Use release builds for better performance

## 🔒 Security Best Practices

1. **Never commit credentials** - `.env` is in `.gitignore`
2. **Use strong passwords** - Change default camera passwords
3. **Network isolation** - Keep cameras on isolated VLAN
4. **Regular updates** - Keep dependencies patched
5. **Monitor access** - Review camera access logs

## 📊 Performance Tips

- **Detection Frequency**: Modify `frame_count % 5` in `main.rs`
- **Resolution**: Lower resolution for faster processing
- **Cascades**: Use only needed detection cascades
- **Build Type**: Always use `--release` for production

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Style
- Follow Rust conventions
- Use meaningful variable names
- Add documentation for public APIs
- Include tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **TP-Link** for Tapo camera API
- **Rust Community** for excellent tooling
- **All Contributors** for improvements

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/tapo-cam/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/tapo-cam/discussions)
- **Wiki**: [Project Wiki](https://github.com/yourusername/tapo-cam/wiki)

---

**Disclaimer**: This project is for educational and personal use. Always comply with local privacy laws and regulations when using camera software.

<p align="center">
  <strong>Made with ❤️ using Rust and OpenCV</strong>
</p>