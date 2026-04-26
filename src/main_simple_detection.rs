mod camera;

use anyhow::Result;
use camera::TapoCamera;
use opencv::{
    core, highgui, imgproc, objdetect, prelude::*, videoio,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const CAM_IP: &str = "192.168.10.185";
const WINDOW: &str = "Tapo C200 - Object Detection";
const PAN_SPEED: i32 = 40;
const TILT_SPEED: i32 = 40;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let email = std::env::var("TAPO_EMAIL")
        .map_err(|_| anyhow::anyhow!("TAPO_EMAIL not set in .env"))?;
    let password = std::env::var("TAPO_PASSWORD")
        .map_err(|_| anyhow::anyhow!("TAPO_PASSWORD not set in .env"))?;

    println!("Connecting to Tapo C200 at {CAM_IP}...");

    // Try to connect to camera for PTZ control
    let camera = match TapoCamera::connect(CAM_IP, &email, &password).await {
        Ok(cam) => {
            let info = cam.get_device_info().await?;
            println!("Connected: {}", info["device_info"]["basic_info"]["device_alias"]
                .as_str()
                .unwrap_or("C200"));
            println!("PTZ controls: ENABLED");
            Some(Arc::new(Mutex::new(cam)))
        }
        Err(e) => {
            println!("Camera authentication failed: {}. PTZ controls will be disabled.", e);
            println!("Video streaming will continue...");
            None
        }
    };

    // Try to load face cascade for object detection
    let face_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    let mut face_cascade = match objdetect::CascadeClassifier::new(face_cascade_path) {
        Ok(cascade) => {
            println!("Face detection: ENABLED");
            Some(cascade)
        }
        Err(e) => {
            println!("Face detection: DISABLED (could not load cascade: {})", e);
            None
        }
    };

    // Construct RTSP URL from credentials
    let rtsp_url = format!("rtsp://{}:{}@{}:554/stream1", email, password, CAM_IP);
    println!("Opening RTSP stream: rtsp://{}:****@{}:554/stream1", email, CAM_IP);
    
    let mut cap = videoio::VideoCapture::from_file(&rtsp_url, videoio::CAP_FFMPEG)?;
    if !cap.is_opened()? {
        anyhow::bail!("Failed to open RTSP stream. Check credentials and network connection.");
    }

    highgui::named_window(WINDOW, highgui::WINDOW_AUTOSIZE)?;

    let mut frame_count = 0u64;
    let mut last_key_check = Instant::now();
    let debounce_delay = Duration::from_millis(250);
    let mut frame = Mat::default();

    println!("Starting object detection. Press Q or ESC to quit.");
    println!("If PTZ is enabled: Arrow keys for pan/tilt, H for calibrate");

    loop {
        if !cap.read(&mut frame)? || frame.size()?.width == 0 {
            println!("End of stream or empty frame");
            break;
        }

        frame_count += 1;
        
        // Run face detection every 5th frame to reduce CPU load
        let mut faces = core::Vector::<core::Rect>::new();
        if frame_count % 5 == 0 {
            if let Some(cascade) = &mut face_cascade {
                let mut gray = Mat::default();
                imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
                
                let mut gray_eq = Mat::default();
                imgproc::equalize_hist(&gray, &mut gray_eq)?;
                
                cascade.detect_multi_scale(
                    &gray_eq,
                    &mut faces,
                    1.1,
                    3,
                    objdetect::CASCADE_SCALE_IMAGE,
                    core::Size::new(30, 30),
                    core::Size::new(0, 0),
                ).ok(); // Ignore errors
            }
        }

        // Draw detected faces
        let mut display_frame = frame.clone();
        for face in &faces {
            let color = core::Scalar::new(0.0, 255.0, 0.0, 0.0); // Green for faces
            let rect = core::Rect::new(face.x, face.y, face.width, face.height);
            imgproc::rectangle(&mut display_frame, rect, color, 2, imgproc::LINE_8, 0)?;
            
            // Draw label
            let label = format!("Face");
            let label_pos = core::Point::new(face.x, face.y - 5);
            imgproc::put_text(
                &mut display_frame,
                &label,
                label_pos,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }

        // Draw statistics
        let stats = format!(
            "Frame: {} | Faces: {}",
            frame_count,
            faces.len()
        );
        
        imgproc::put_text(
            &mut display_frame,
            &stats,
            core::Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        // Draw instructions
        let instructions = if camera.is_some() {
            "Q/ESC: Quit | H: Calibrate | Arrow keys: Pan/Tilt"
        } else {
            "Q/ESC: Quit | Face detection active"
        };
        
        imgproc::put_text(
            &mut display_frame,
            instructions,
            core::Point::new(10, 60),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;

        // Show frame
        highgui::imshow(WINDOW, &display_frame)?;

        // Check for keyboard input
        let key = highgui::wait_key(30)?;
        let now = Instant::now();
        
        if now.duration_since(last_key_check) > debounce_delay {
            match key {
                27 | 113 => break, // ESC or 'q'
                104 | 72 if camera.is_some() => { // 'h' or 'H'
                    if let Some(cam) = &camera {
                        let cam = cam.lock().await;
                        if let Err(e) = cam.calibrate().await {
                            eprintln!("Calibration failed: {}", e);
                        } else {
                            println!("Camera calibration/return to home initiated");
                        }
                    }
                }
                81 | 82 | 83 | 84 if camera.is_some() => { // Arrow keys
                    if let Some(cam) = &camera {
                        let cam = cam.lock().await;
                        let (pan, tilt) = match key {
                            81 => (-PAN_SPEED, 0),    // Left arrow
                            82 => (PAN_SPEED, 0),     // Right arrow  
                            83 => (0, TILT_SPEED),    // Up arrow
                            84 => (0, -TILT_SPEED),   // Down arrow
                            _ => (0, 0),
                        };
                        
                        if pan != 0 || tilt != 0 {
                            if let Err(e) = cam.move_motor(pan, tilt).await {
                                eprintln!("PTZ move failed: {}", e);
                            }
                        }
                    }
                }
                _ => {}
            }
            last_key_check = now;
        }
    }

    println!("Shutting down...");
    highgui::destroy_window(WINDOW)?;
    Ok(())
}