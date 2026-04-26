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
const WINDOW: &str = "Tapo C200 - Face & Eye Detection";
const PAN_SPEED: i32 = 40;
const TILT_SPEED: i32 = 40;

struct FaceDetector {
    face_cascade: objdetect::CascadeClassifier,
    eye_cascade: objdetect::CascadeClassifier,
    scale_factor: f64,
    min_neighbors: i32,
    min_size: core::Size,
    max_size: core::Size,
}

impl FaceDetector {
    fn new() -> Result<Self> {
        // Load face cascade
        let mut face_cascade = objdetect::CascadeClassifier::new(
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        )?;
        
        // Load eye cascade
        let mut eye_cascade = objdetect::CascadeClassifier::new(
            "/usr/share/opencv4/haarcascades/haarcascade_eye.xml",
        )?;
        
        Ok(Self {
            face_cascade,
            eye_cascade,
            scale_factor: 1.1,
            min_neighbors: 3,
            min_size: core::Size::new(30, 30),
            max_size: core::Size::new(300, 300),
        })
    }
    
    fn detect_faces(&mut self, frame: &Mat) -> Result<Vec<core::Rect>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Equalize histogram for better contrast
        let mut gray_eq = Mat::default();
        imgproc::equalize_hist(&gray, &mut gray_eq)?;
        
        // Detect faces
        let mut faces = core::Vector::new();
        self.face_cascade.detect_multi_scale(
            &gray_eq,
            &mut faces,
            self.scale_factor,
            self.min_neighbors,
            0, // flags
            self.min_size,
            self.max_size,
        )?;
        
        Ok(faces.to_vec())
    }
    
    fn detect_eyes(&mut self, face_roi: &Mat) -> Result<Vec<core::Rect>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(face_roi, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        let mut eyes = core::Vector::new();
        self.eye_cascade.detect_multi_scale(
            &gray,
            &mut eyes,
            1.1, // scale factor
            2,   // min neighbors
            0,   // flags
            core::Size::new(20, 20), // min size
            core::Size::new(100, 100), // max size
        )?;
        
        Ok(eyes.to_vec())
    }
}

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

    // Initialize face detector
    println!("Initializing face detector...");
    let mut detector = FaceDetector::new()?;
    println!("Face detector ready with OpenCV Haar cascades");

    // Construct RTSP URL from credentials
    let rtsp_url = format!("rtsp://{}:{}@{}:554/stream1", email, password, CAM_IP);
    println!("Opening RTSP stream: rtsp://{}:****@{}:554/stream1", email, CAM_IP);
    
    let mut cap = videoio::VideoCapture::from_file(&rtsp_url, videoio::CAP_FFMPEG)?;
    if !cap.is_opened()? {
        anyhow::bail!("Failed to open RTSP stream. Check credentials and network connection.");
    }

    // Get frame info
    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    println!("Stream info: {}x{} @ {:.1} FPS", frame_width, frame_height, fps);

    highgui::named_window(WINDOW, highgui::WINDOW_AUTOSIZE)?;

    let mut frame_count = 0u64;
    let mut last_key_check = Instant::now();
    let debounce_delay = Duration::from_millis(250);
    let mut frame = Mat::default();

    println!("Starting face detection. Press Q or ESC to quit.");
    println!("If PTZ is enabled: Arrow keys for pan/tilt, H for calibrate");

    // Performance tracking
    let mut fps_counter = 0u32;
    let mut last_fps_time = Instant::now();
    let mut processing_times = Vec::new();

    loop {
        let frame_start = Instant::now();
        
        if !cap.read(&mut frame)? || frame.size()?.width == 0 {
            println!("End of stream or empty frame");
            break;
        }

        frame_count += 1;
        fps_counter += 1;
        
        // Create display frame
        let mut display_frame = frame.clone();
        
        // Run face detection every 3rd frame for better performance
        let mut faces = Vec::new();
        if frame_count % 3 == 0 {
            match detector.detect_faces(&frame) {
                Ok(detected_faces) => {
                    faces = detected_faces;
                }
                Err(e) => {
                    eprintln!("Face detection error: {}", e);
                }
            }
        }
        
        // Draw face detections
        for (i, face) in faces.iter().enumerate() {
            // Draw face rectangle (green)
            imgproc::rectangle(
                &mut display_frame, 
                *face, 
                core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green
                2, 
                imgproc::LINE_8, 
                0
            )?;
            
            // Draw face label
            let label = format!("Face {}", i + 1);
            imgproc::put_text(
                &mut display_frame,
                &label,
                core::Point::new(face.x, face.y - 5),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green
                1,
                imgproc::LINE_AA,
                false,
            )?;
            
            // Try to detect eyes within the face ROI
            if face.width > 50 && face.height > 50 { // Only for larger faces
                let face_roi = Mat::roi(&frame, *face)?;
                match detector.detect_eyes(&face_roi) {
                    Ok(eyes) => {
                        for eye in eyes {
                            // Adjust eye coordinates relative to the whole frame
                            let eye_rect = core::Rect::new(
                                face.x + eye.x,
                                face.y + eye.y,
                                eye.width,
                                eye.height,
                            );
                            
                            // Draw eye rectangle (blue)
                            imgproc::rectangle(
                                &mut display_frame,
                                eye_rect,
                                core::Scalar::new(255.0, 0.0, 0.0, 0.0), // Blue
                                1,
                                imgproc::LINE_8,
                                0,
                            )?;
                        }
                    }
                    Err(_) => {
                        // Eye detection failed, skip
                    }
                }
            }
            
            // Draw face center point (red)
            let face_center = core::Point::new(
                face.x + face.width / 2,
                face.y + face.height / 2,
            );
            imgproc::circle(
                &mut display_frame,
                face_center,
                3,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red
                2,
                imgproc::LINE_AA,
                0,
            )?;
        }

        // Calculate FPS
        let now = Instant::now();
        if now.duration_since(last_fps_time) > Duration::from_secs(1) {
            let fps = fps_counter as f32 / now.duration_since(last_fps_time).as_secs_f32();
            fps_counter = 0;
            last_fps_time = now;
            
            // Track processing time
            let process_time = frame_start.elapsed();
            processing_times.push(process_time);
            if processing_times.len() > 60 {
                processing_times.remove(0);
            }
            
            // Draw performance info
            let avg_process_time: Duration = processing_times.iter().sum::<Duration>() / processing_times.len() as u32;
            let stats = format!(
                "FPS: {:.1} | Faces: {} | Process: {:.1}ms",
                fps,
                faces.len(),
                avg_process_time.as_secs_f32() * 1000.0
            );
            
            imgproc::put_text(
                &mut display_frame,
                &stats,
                core::Point::new(10, 30),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }

        // Draw frame counter
        let frame_info = format!("Frame: {}", frame_count);
        imgproc::put_text(
            &mut display_frame,
            &frame_info,
            core::Point::new(10, 60),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;

        // Draw instructions
        let instructions = if camera.is_some() {
            "Q/ESC: Quit | H: Calibrate | Arrow keys: Pan/Tilt"
        } else {
            "Q/ESC: Quit | PTZ: Disabled (auth failed)"
        };
        
        imgproc::put_text(
            &mut display_frame,
            &instructions,
            core::Point::new(10, frame_height - 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;

        // Show frame
        highgui::imshow(WINDOW, &display_frame)?;

        // Handle keyboard input with debouncing
        let now = Instant::now();
        if now.duration_since(last_key_check) > debounce_delay {
            let key = highgui::wait_key(1)?; // Very short wait for responsive UI
            
            match key {
                113 | 27 => break, // 'q' or ESC
                104 | 72 if camera.is_some() => { // 'h' or 'H'
                    if let Some(cam) = &camera {
                        let cam_lock = cam.lock().await;
                        if let Err(e) = cam_lock.calibrate().await {
                            println!("Failed to calibrate: {}", e);
                        } else {
                            println!("Calibrating camera...");
                        }
                    }
                }
                81 | 82 if camera.is_some() => { // Left/Right arrow keys (Linux keycodes)
                    if let Some(cam) = &camera {
                        let cam_lock = cam.lock().await;
                        let _ = cam_lock.move_motor(-PAN_SPEED, 0).await;
                        println!("Panning left...");
                    }
                }
                83 | 84 if camera.is_some() => { // Up/Down arrow keys (Linux keycodes)
                    if let Some(cam) = &camera {
                        let cam_lock = cam.lock().await;
                        let _ = cam_lock.move_motor(0, TILT_SPEED).await;
                        println!("Tilting up...");
                    }
                }
                _ => {}
            }
            
            last_key_check = now;
        }
        
        // Check if we're running too slow
        let frame_time = frame_start.elapsed();
        if frame_time > Duration::from_millis(100) {
            println!("Warning: Frame processing took {:.1}ms", frame_time.as_secs_f32() * 1000.0);
        }
    }

    highgui::destroy_window(WINDOW)?;
    println!("Application closed after {} frames.", frame_count);
    Ok(())
}