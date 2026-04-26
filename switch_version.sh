#!/bin/bash

# Script to switch between different versions of tapo-cam

VERSION="$1"

case "$VERSION" in
    "original")
        echo "Switching to original version (no object detection)..."
        if [ -f "src/main_backup.rs" ]; then
            cp src/main_backup.rs src/main.rs
            echo "✅ Switched to original version"
        else
            echo "❌ Original backup not found. Looking for other versions..."
            if [ -f "src/main_simple_detection.rs" ]; then
                # Create a version without object detection
                echo "Creating original version from simple detection..."
                sed '/face_cascade/d' src/main_simple_detection.rs | \
                  sed '/face detection/d' | \
                  sed '/faces/d' > src/main.rs
                echo "✅ Created original version"
            else
                echo "❌ No version files found"
            fi
        fi
        ;;
    "simple-detection")
        echo "Switching to simple face detection version..."
        if [ -f "src/main_simple_detection.rs" ]; then
            # Backup current main.rs
            if [ -f "src/main.rs" ]; then
                cp src/main.rs src/main_backup.rs
            fi
            cp src/main_simple_detection.rs src/main.rs
            echo "✅ Switched to simple face detection version"
        else
            echo "❌ Simple detection version not found"
        fi
        ;;
    "full-detection")
        echo "Switching to full object detection version..."
        if [ -f "src/main_object_detection.rs" ]; then
            # Backup current main.rs
            if [ -f "src/main.rs" ]; then
                cp src/main.rs src/main_backup.rs
            fi
            cp src/main_object_detection.rs src/main.rs
            echo "✅ Switched to full object detection version"
        else
            echo "❌ Full detection version not found"
        fi
        ;;
    "list")
        echo "Available versions:"
        echo "  original          - Basic camera control without object detection"
        echo "  simple-detection  - Face detection only (recommended)"
        echo "  full-detection    - Full object detection (humans, animals, etc.)"
        echo ""
        echo "Usage: ./switch_version.sh <version>"
        echo "Example: ./switch_version.sh simple-detection"
        ;;
    *)
        echo "Usage: ./switch_version.sh <version>"
        echo ""
        echo "Versions:"
        echo "  original          - Basic camera control"
        echo "  simple-detection  - Face detection (current)"
        echo "  full-detection    - Full object detection"
        echo "  list              - List available versions"
        echo ""
        echo "Current version is configured for: simple-detection"
        ;;
esac