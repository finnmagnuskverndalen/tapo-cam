mod camera;

use anyhow::Result;
use camera::TapoCamera;
use opencv::{
    core, highgui, imgproc, prelude::*, videoio,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const CAM_IP: &str = "192.168.10.185";
const WINDOW: &str = "Tapo C200 - Better Object Detection";
const PAN_SPEED: i32 = 40;
const TILT_SPEED: i32 = 40;

// Simple motion-based object detection
struct MotionDetector {
    background: Option<Mat>,
    motion_threshold: f64,
    min_motion_area: f32,
    learning_rate: f64,
}

impl MotionDetector {
    fn new(threshold: f64, min_area: f32) -> Self {
        Self {
            background: None,
            motion_threshold: threshold,
            min_motion_area: min_area,
            learning_rate: 0.01, // Slow background learning
        }
    }

    fn detect_motion(&mut self, frame: &Mat) -> Result<Vec<core::Rect>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Blur to reduce noise
        let mut blurred = Mat::default();
        imgproc::gaussian_blur(&gray, &mut blurred, core::Size::new(21, 21), 0.0, 0.0, core::BORDER_DEFAULT)?;
        
        // Initialize background if needed
        if self.background.is_none() {
            self.background = Some(blurred.clone());
            return Ok(Vec::new());
        }
        
        let background = self.background.as_ref().unwrap();
        
        // Compute absolute difference between current frame and background
        let mut diff = Mat::default();
        core::absdiff(background, &blurred, &mut diff)?;
        
        // Threshold the difference image
        let mut thresh = Mat::default();
        imgproc::threshold(&diff, &mut thresh, self.motion_threshold, 255.0, imgproc::THRESH_BINARY)?;
        
        // Dilate to fill in holes
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(5, 5),
            core::Point::new(-1, -1),
        )?;
        let mut dilated = Mat::default();
        imgproc::dilate(&thresh, &mut dilated, &kernel, core::Point::new(-1, -1), 1, core::BORDER_CONSTANT, core::Scalar::default())?;
        thresh = dilated;
        
        // Find contours
        let mut contours = core::Vector::<core::Vector<core::Point>>::new();
        imgproc::find_contours(&thresh, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
        
        // Filter contours by area and convert to bounding boxes
        let mut motion_regions = Vec::new();
        for contour in contours.iter() {
            let area = imgproc::contour_area(&contour, false)?;
            if area > self.min_motion_area as f64 {
                let bbox = imgproc::bounding_rect(&contour)?;
                motion_regions.push(bbox);
            }
        }
        
        // Update background model (slowly adapt to changes)
        if let Some(bg) = &mut self.background {
            let mut updated_bg = Mat::default();
            core::add_weighted(bg, 1.0 - self.learning_rate, &blurred, self.learning_rate, 0.0, &mut updated_bg, -1)?;
            *bg = updated_bg;
        }
        
        Ok(motion_regions)
    }
}

// Simple object tracker without heavy computations
struct SimpleTracker {
    objects: Vec<core::Rect>,
    last_positions: Vec<core::Point>,
    frame_count: u32,
}

impl SimpleTracker {
    fn new() -> Self {
        Self {
            objects: Vec::new(),
            last_positions: Vec::new(),
            frame_count: 0,
        }
    }
    
    fn update(&mut self, new_detections: Vec<core::Rect>) {
        self.frame_count += 1;
        
        // Simple tracking: if detection is close to previous position, keep it
        let mut tracked_objects = Vec::new();
        let mut tracked_positions = Vec::new();
        
        for det in new_detections {
            let center = core::Point::new(det.x + det.width / 2, det.y + det.height / 2);
            let mut matched = false;
            
            for (i, &last_pos) in self.last_positions.iter().enumerate() {
                let distance = ((center.x - last_pos.x).pow(2) + (center.y - last_pos.y).pow(2)) as f32;
                if distance < 5000.0 { // Threshold for matching
                    tracked_objects.push(self.objects[i]);
                    tracked_positions.push(center);
                    matched = true;
                    break;
                }
            }
            
            if !matched {
                tracked_objects.push(det);
                tracked_positions.push(center);
            }
        }
        
        self.objects = tracked_objects;
        self.last_positions = tracked_positions;
        
        // Clear if too many frames without updates
        if self.frame_count % 30 == 0 && self.objects.is_empty() {
            self.objects.clear();
            self.last_positions.clear();
        }
    }
    
    fn get_objects(&self) -> &[core::Rect] {
        &self.objects
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

    // Create motion detector and tracker
    let mut motion_detector = MotionDetector::new(25.0, 500.0); // Threshold 25, min area 500
    let mut tracker = SimpleTracker::new();

    highgui::named_window(WINDOW, highgui::WINDOW_AUTOSIZE)?;

    let mut frame_count = 0u64;
    let mut last_key_check = Instant::now();
    let debounce_delay = Duration::from_millis(250);
    let mut frame = Mat::default();

    println!("Starting motion-based object detection. Press Q or ESC to quit.");
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
        
        // Resize frame for faster processing (if needed)
        let mut processed_frame = frame.clone();
        let scale_factor = 0.5; // Process at half resolution for speed
        if scale_factor < 1.0 {
            let new_width = (frame_width as f32 * scale_factor) as i32;
            let new_height = (frame_height as f32 * scale_factor) as i32;
            imgproc::resize(&frame, &mut processed_frame, core::Size::new(new_width, new_height), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        }
        
        // Run motion detection every 3rd frame for better performance
        let mut motion_regions = Vec::new();
        if frame_count % 3 == 0 {
            match motion_detector.detect_motion(&processed_frame) {
                Ok(regions) => {
                    motion_regions = regions;
                    
                    // Scale regions back to original size if we scaled down
                    if scale_factor < 1.0 {
                        for region in motion_regions.iter_mut() {
                            region.x = (region.x as f32 / scale_factor) as i32;
                            region.y = (region.y as f32 / scale_factor) as i32;
                            region.width = (region.width as f32 / scale_factor) as i32;
                            region.height = (region.height as f32 / scale_factor) as i32;
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Motion detection error: {}", e);
                }
            }
            
            tracker.update(motion_regions);
        }
        
        // Draw tracked objects
        let mut display_frame = frame.clone();
        let objects = tracker.get_objects();
        
        for (i, obj) in objects.iter().enumerate() {
            // Color based on object index (for multiple objects)
            let colors = [
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),   // Green
                core::Scalar::new(255.0, 0.0, 0.0, 0.0),   // Blue
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),   // Red
                core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Yellow
                core::Scalar::new(255.0, 0.0, 255.0, 0.0), // Magenta
                core::Scalar::new(255.0, 255.0, 0.0, 0.0), // Cyan
            ];
            let color = colors[i % colors.len()];
            
            // Draw bounding box
            imgproc::rectangle(&mut display_frame, *obj, color, 2, imgproc::LINE_8, 0)?;
            
            // Draw label
            let label = format!("Motion {}", i + 1);
            let label_pos = core::Point::new(obj.x, obj.y - 5);
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
            
            // Draw center point
            let center = core::Point::new(obj.x + obj.width / 2, obj.y + obj.height / 2);
            imgproc::circle(
                &mut display_frame,
                center,
                3,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
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
                "FPS: {:.1} | Objects: {} | Process: {:.1}ms",
                fps,
                objects.len(),
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