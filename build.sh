#!/bin/bash

# Build script for tapo-cam with object detection

echo "Building tapo-cam with object detection..."

# Create a symbolic link for the main file
if [ -f "src/main_object_detection.rs" ]; then
    if [ -f "src/main.rs" ]; then
        echo "Backing up original main.rs..."
        cp src/main.rs src/main_backup.rs
    fi
    echo "Using object detection version..."
    cp src/main_object_detection.rs src/main.rs
fi

# Build the project
echo "Building with cargo..."
cargo build --release

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "To run with object detection:"
    echo "  ./target/release/tapo-cam"
    echo ""
    echo "Features:"
    echo "  - Human face detection"
    echo "  - Full body detection"
    echo "  - Animal (cat) detection"
    echo "  - RTSP video streaming"
    echo "  - PTZ control (if authentication works)"
    echo ""
    echo "Controls:"
    echo "  Q / ESC: Quit"
    echo "  H: Calibrate camera (if PTZ enabled)"
    echo "  Arrow keys: Pan/Tilt (if PTZ enabled)"
else
    echo ""
    echo "❌ Build failed!"
    echo "Make sure you have OpenCV development libraries installed:"
    echo "  sudo apt-get install libopencv-dev"
    echo ""
fi

# Restore original main.rs if we backed it up
if [ -f "src/main_backup.rs" ]; then
    echo "Restoring original main.rs..."
    mv src/main_backup.rs src/main.rs
fi