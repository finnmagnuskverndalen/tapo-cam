mod camera;

use anyhow::Result;
use camera::TapoCamera;
use opencv::{
    core, highgui, imgproc, objdetect, prelude::*, types, videoio,
};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const CAM_IP: &str = "192.168.10.185";
const WINDOW: &str = "Tapo C200 - Object Detection";
const PAN_SPEED: i32 = 40;
const TILT_SPEED: i32 = 40;

// Object types we can detect
#[derive(Debug, Clone, Copy, PartialEq)]
enum ObjectType {
    Human,
    Animal,
    Vehicle,
    Unknown,
}

// Detected object with bounding box and type
#[derive(Debug, Clone)]
struct DetectedObject {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    object_type: ObjectType,
    confidence: f32,
}

struct ObjectDetector {
    face_cascade: objdetect::CascadeClassifier,
    fullbody_cascade: objdetect::CascadeClassifier,
    upperbody_cascade: objdetect::CascadeClassifier,
    cat_cascade: objdetect::CascadeClassifier,
}

impl ObjectDetector {
    fn new() -> Result<Self> {
        // Try to find cascade files in common locations
        let cascade_paths = [
            "/usr/share/opencv4/haarcascades/",
            "/usr/share/opencv/haarcascades/",
            "/opt/homebrew/share/opencv4/haarcascades/",
            "./cascades/",
        ];

        let mut found_path = None;
        for path in cascade_paths {
            let test_file = format!("{}haarcascade_frontalface_default.xml", path);
            if Path::new(&test_file).exists() {
                found_path = Some(path);
                break;
            }
        }

        let base_path = found_path.ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find OpenCV cascade files. Install with: sudo apt-get install opencv-data"
            )
        })?;

        // Load cascade classifiers
        let face_cascade = objdetect::CascadeClassifier::new(
            &format!("{}haarcascade_frontalface_default.xml", base_path),
        )?;
        
        let fullbody_cascade = objdetect::CascadeClassifier::new(
            &format!("{}haarcascade_fullbody.xml", base_path),
        )?;
        
        let upperbody_cascade = objdetect::CascadeClassifier::new(
            &format!("{}haarcascade_upperbody.xml", base_path),
        )?;
        
        // Try to load cat cascade (might not exist in all installations)
        let cat_cascade = objdetect::CascadeClassifier::new(
            &format!("{}haarcascade_frontalcatface_extended.xml", base_path),
        ).unwrap_or_else(|_| {
            objdetect::CascadeClassifier::new(
                &format!("{}haarcascade_frontalcatface.xml", base_path),
            ).unwrap_or_default()
        });

        Ok(Self {
            face_cascade,
            fullbody_cascade,
            upperbody_cascade,
            cat_cascade,
        })
    }

    fn detect_objects(&mut self, frame: &Mat) -> Result<Vec<DetectedObject>> {
        // Downscale for faster detection
        let mut small = Mat::default();
        imgproc::resize(frame, &mut small, core::Size::default(), 0.5, 0.5, imgproc::INTER_LINEAR)?;

        let mut gray = Mat::default();
        imgproc::cvt_color(&small, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        // equalize_hist requires separate src and dst — using same Mat is undefined behavior
        let mut gray_eq = Mat::default();
        imgproc::equalize_hist(&gray, &mut gray_eq)?;
        let gray = gray_eq;

        // Detected rects are in the downscaled image; scale back to original coordinates
        let scale_rect = |r: core::Rect| DetectedObject {
            x: (r.x * 2),
            y: (r.y * 2),
            width: (r.width * 2),
            height: (r.height * 2),
            object_type: ObjectType::Unknown, // overridden below
            confidence: 0.0,
        };

        let mut objects = Vec::new();

        // min_size is in the downscaled image: 15px here = 30px in the original
        let min_sz = core::Size::new(15, 15);

        let mut faces = types::VectorOfRect::new();
        self.face_cascade.detect_multi_scale(
            &gray, &mut faces, 1.1, 3,
            objdetect::CASCADE_SCALE_IMAGE, min_sz, core::Size::new(0, 0),
        )?;
        for face in faces {
            let mut obj = scale_rect(face);
            obj.object_type = ObjectType::Human;
            obj.confidence = 0.8;
            objects.push(obj);
        }

        let mut full_bodies = types::VectorOfRect::new();
        self.fullbody_cascade.detect_multi_scale(
            &gray, &mut full_bodies, 1.05, 3,
            objdetect::CASCADE_SCALE_IMAGE, min_sz, core::Size::new(0, 0),
        )?;
        for body in full_bodies {
            let mut obj = scale_rect(body);
            obj.object_type = ObjectType::Human;
            obj.confidence = 0.7;
            objects.push(obj);
        }

        let mut upper_bodies = types::VectorOfRect::new();
        self.upperbody_cascade.detect_multi_scale(
            &gray, &mut upper_bodies, 1.05, 3,
            objdetect::CASCADE_SCALE_IMAGE, min_sz, core::Size::new(0, 0),
        )?;
        for body in upper_bodies {
            let mut obj = scale_rect(body);
            obj.object_type = ObjectType::Human;
            obj.confidence = 0.6;
            objects.push(obj);
        }

        if !self.cat_cascade.empty()? {
            let mut cats = types::VectorOfRect::new();
            self.cat_cascade.detect_multi_scale(
                &gray, &mut cats, 1.05, 3,
                objdetect::CASCADE_SCALE_IMAGE, min_sz, core::Size::new(0, 0),
            )?;
            for cat in cats {
                let mut obj = scale_rect(cat);
                obj.object_type = ObjectType::Animal;
                obj.confidence = 0.7;
                objects.push(obj);
            }
        }

        Ok(objects)
    }
}

fn draw_objects(frame: &mut Mat, objects: &[DetectedObject], frame_count: u64) -> Result<()> {
    for obj in objects {
        let color = match obj.object_type {
            ObjectType::Human => core::Scalar::new(0.0, 255.0, 0.0, 0.0), // Green for humans
            ObjectType::Animal => core::Scalar::new(255.0, 0.0, 0.0, 0.0), // Blue for animals
            ObjectType::Vehicle => core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red for vehicles
            ObjectType::Unknown => core::Scalar::new(255.0, 255.0, 0.0, 0.0), // Yellow for unknown
        };

        let thickness = 2;
        let top_left = core::Point::new(obj.x, obj.y);
        let bottom_right = core::Point::new(obj.x + obj.width, obj.y + obj.height);
        
        // Draw bounding box
        imgproc::rectangle(frame, top_left, bottom_right, color, thickness, imgproc::LINE_8, 0)?;
        
        // Draw label
        let label = format!(
            "{:?} {:.1}%",
            obj.object_type,
            obj.confidence * 100.0
        );
        let label_pos = core::Point::new(obj.x, obj.y - 5);
        imgproc::put_text(
            frame,
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
        "Frame: {} | Objects: {} (H:{} A:{})",
        frame_count,
        objects.len(),
        objects.iter().filter(|o| o.object_type == ObjectType::Human).count(),
        objects.iter().filter(|o| o.object_type == ObjectType::Animal).count()
    );
    
    imgproc::put_text(
        frame,
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
    let instructions = "Q/ESC: Quit | H: Calibrate | Arrow keys: Pan/Tilt";
    imgproc::put_text(
        frame,
        instructions,
        core::Point::new(10, 60),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
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

    // Initialize object detector
    let mut detector = ObjectDetector::new()?;
    println!("Object detector initialized successfully");

    // Must be set before VideoCapture opens the stream
    // SAFETY: single-threaded at this point; no other threads read this env var
    unsafe {
        std::env::set_var(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
        );
    }

    let rtsp_url = format!("rtsp://{}:{}@{}:554/stream1", email, password, CAM_IP);
    println!("Opening RTSP stream: rtsp://{}:****@{}:554/stream1", email, CAM_IP);

    let mut cap = videoio::VideoCapture::from_file(&rtsp_url, videoio::CAP_FFMPEG)?;
    if !cap.is_opened()? {
        anyhow::bail!("Failed to open RTSP stream. Check credentials and network connection.");
    }
    cap.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

    highgui::named_window(WINDOW, highgui::WINDOW_AUTOSIZE)?;

    let mut frame_count = 0u64;
    let mut frame = Mat::default();
    // Persist detections across frames — avoids flicker on non-detection frames
    let mut last_objects: Vec<DetectedObject> = Vec::new();
    let mut ptz_debounce = Instant::now() - Duration::from_secs(1);
    let ptz_debounce_ms = Duration::from_millis(200);

    // Arrow key codes for Linux/X11
    const KEY_LEFT: i32 = 65361;
    const KEY_UP: i32 = 65362;
    const KEY_RIGHT: i32 = 65363;
    const KEY_DOWN: i32 = 65364;

    println!("Starting object detection. Press Q or ESC to quit.");

    loop {
        if !cap.read(&mut frame)? || frame.size()?.width == 0 {
            println!("End of stream or empty frame");
            break;
        }

        frame_count += 1;

        // Run detection every 3rd frame; carry last_objects on non-detection frames
        if frame_count % 3 == 0 {
            match detector.detect_objects(&frame) {
                Ok(objects) => last_objects = objects,
                Err(e) => eprintln!("Detection error: {}", e),
            }
        }

        let mut display_frame = frame.clone();
        draw_objects(&mut display_frame, &last_objects, frame_count)?;

        highgui::imshow(WINDOW, &display_frame)?;

        // wait_key must be called every frame for GUI events to process
        let key = highgui::wait_key(1)?;
        match key {
            27 | 113 => break, // ESC or 'q'
            _ => {}
        }

        // PTZ commands with debounce
        let now = Instant::now();
        if now.duration_since(ptz_debounce) > ptz_debounce_ms {
            if let Some(cam) = &camera {
                let handled = match key {
                    104 | 72 => {
                        cam.lock().await.calibrate().await.ok();
                        println!("Calibrating...");
                        true
                    }
                    KEY_LEFT => { cam.lock().await.move_motor(-PAN_SPEED, 0).await.ok(); true }
                    KEY_RIGHT => { cam.lock().await.move_motor(PAN_SPEED, 0).await.ok(); true }
                    KEY_UP => { cam.lock().await.move_motor(0, TILT_SPEED).await.ok(); true }
                    KEY_DOWN => { cam.lock().await.move_motor(0, -TILT_SPEED).await.ok(); true }
                    _ => false,
                };
                if handled {
                    ptz_debounce = Instant::now();
                }
            }
        }
    }

    println!("Shutting down...");
    highgui::destroy_window(WINDOW)?;
    Ok(())
}