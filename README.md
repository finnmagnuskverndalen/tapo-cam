# tapo-cam

A Rust application for viewing and remotely controlling a TP-Link Tapo C200 IP camera using OpenCV and the camera's local HTTP API.

## Features

- Live RTSP video stream rendered with OpenCV
- Pan/tilt control via arrow keys (direct local API, no cloud)
- Return-to-home command
- Debounced PTZ input to avoid flooding the camera

## Requirements

### System dependencies

```bash
sudo apt install -y libopencv-dev libclang-dev clang
```

### Rust

Rust 1.70+ (install via [rustup](https://rustup.rs))

## Configuration

Edit the constants at the top of `src/main.rs`:

```rust
const CAM_IP: &str = "192.168.10.185";   // your camera's IP
const RTSP_URL: &str = "rtsp://192.168.10.185/stream1";
const TAPO_EMAIL: &str = "YOUR_TAPO_EMAIL";
const TAPO_PASSWORD: &str = "YOUR_TAPO_PASSWORD";
```

`TAPO_EMAIL` and `TAPO_PASSWORD` are the credentials for your Tapo/TP-Link account.

## Build & run

```bash
cargo run --release
```

First build will take a few minutes — the OpenCV bindings are large.

## Controls

| Key | Action |
|-----|--------|
| ← → | Pan left / right |
| ↑ ↓ | Tilt up / down |
| H | Return to home position |
| Q / ESC | Quit |

## How it works

**Video**: OpenCV connects to the camera's RTSP stream (`/stream1`) via FFmpeg and renders frames in a native window.

**PTZ control**: The C200 exposes a local HTTPS API on port 443. This app authenticates using the Tapo passthrough protocol (SHA1+MD5 hashed email → base64, base64 password), retrieves a session token (`stok`), then sends `motor_move` commands directly to the camera on your LAN — no cloud involved.

## Project structure

```
src/
├── main.rs     — video loop, keyboard input, PTZ dispatch
└── camera.rs   — Tapo C200 auth and API calls
```

## Finding your camera's IP

```bash
nmap -sn 192.168.1.0/24
```

Look for a host named `C200` or with a TP-Link MAC address.
