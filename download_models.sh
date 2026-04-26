#!/bin/bash
# Download MobileNet SSD model for object detection

MODEL_DIR="models"
mkdir -p "$MODEL_DIR"

echo "Downloading MobileNet SSD model files..."

# Model config (prototxt)
echo "Downloading model config..."
wget -O "$MODEL_DIR/MobileNetSSD_deploy.prototxt" \
    https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt

# Model weights (caffemodel)
echo "Downloading model weights..."
wget -O "$MODEL_DIR/MobileNetSSD_deploy.caffemodel" \
    https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc

echo "Download complete!"
echo "Model files saved to: $MODEL_DIR/"