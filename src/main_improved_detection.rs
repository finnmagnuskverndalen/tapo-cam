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
const WINDOW: &str = "Tapo C200 - Improved Object Detection";
const PAN_SPEED: i32 = 40;
const TILT_SPEED: i32 = 40;

// Struct to track detected objects with history
#[derive(Debug, Clone)]
struct TrackedObject {
    id: u32,
    bbox: core::Rect,
    confidence: f32,
    last_seen: Instant,
    age: Duration,
    history: Vec<core::Point>, // Center point history for smoothing
    stable: bool,
}

impl TrackedObject {
    fn new(id: u32, bbox: core::Rect, confidence: f32) -> Self {
        let center = core::Point::new(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        Self {
            id,
            bbox,
            confidence,
            last_seen: Instant::now(),
            age: Duration::from_secs(0),
            history: vec![center],
            stable: false,
        }
    }

    fn center(&self) -> core::Point {
        let bbox = &self.bbox;
        core::Point::new(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2)
    }

    fn update(&mut self, new_bbox: core::Rect, confidence: f32) {
        let new_center = core::Point::new(new_bbox.x + new_bbox.width / 2, new_bbox.y + new_bbox.height / 2);
        
        // Smooth the bbox position using moving average
        self.history.push(new_center);
        if self.history.len() > 10 {
            self.history.remove(0);
        }
        
        // Calculate smoothed center
        let sum_x: i32 = self.history.iter().map(|p| p.x).sum();
        let sum_y: i32 = self.history.iter().map(|p| p.y).sum();
        let avg_x = sum_x / self.history.len() as i32;
        let avg_y = sum_y / self.history.len() as i32;
        
        // Update bbox with smoothed position
        let half_width = new_bbox.width / 2;
        let half_height = new_bbox.height / 2;
        self.bbox = core::Rect::new(avg_x - half_width, avg_y - half_height, new_bbox.width, new_bbox.height);
        
        self.confidence = (self.confidence * 0.7) + (confidence * 0.3); // Moving average of confidence
        self.last_seen = Instant::now();
        self.age = self.last_seen.duration_since(Instant::now());
        
        // Mark as stable if we've seen it for a while
        if self.history.len() >= 5 {
            self.stable = true;
        }
    }

    fn should_remove(&self) -> bool {
        let since_seen = Instant::now().duration_since(self.last_seen);
        since_seen > Duration::from_secs(2) // Remove if not seen for 2 seconds
    }
}

struct ObjectTracker {
    objects: Vec<TrackedObject>,
    next_id: u32,
    frame_size: core::Size,
}

impl ObjectTracker {
    fn new(frame_width: i32, frame_height: i32) -> Self {
        Self {
            objects: Vec::new(),
            next_id: 1,
            frame_size: core::Size::new(frame_width, frame_height),
        }
    }

    fn update(&mut self, new_detections: Vec<core::Rect>, confidences: Vec<f32>) {
        let current_time = Instant::now();
        
        // Mark all objects as unmatched
        for obj in &mut self.objects {
            // Age the object
            obj.age = current_time.duration_since(obj.last_seen);
        }
        
        // Match new detections to existing objects using IoU
        let mut matched = vec![false; new_detections.len()];
        
        for i in 0..new_detections.len() {
            let detection = &new_detections[i];
            let confidence = confidences[i];
            
            let mut best_match_idx = None;
            let mut best_iou = 0.3; // Minimum IoU threshold
            
            for (j, obj) in self.objects.iter_mut().enumerate() {
                let iou = self.calculate_iou(detection, &obj.bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_match_idx = Some(j);
                }
            }
            
            if let Some(idx) = best_match_idx {
                self.objects[idx].update(*detection, confidence);
                matched[i] = true;
            }
        }
        
        // Create new objects for unmatched detections
        for i in 0..new_detections.len() {
            if !matched[i] && confidences[i] > 0.5 {
                let new_obj = TrackedObject::new(self.next_id, new_detections[i], confidences[i]);
                self.objects.push(new_obj);
                self.next_id += 1;
            }
        }
        
        // Remove old objects
        self.objects.retain(|obj| !obj.should_remove());
    }
    
    fn calculate_iou(&self, a: &core::Rect, b: &core::Rect) -> f32 {
        let intersection = self.rect_intersection(a, b);
        let union = (a.width * a.height) + (b.width * b.height) - intersection;
        
        if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        }
    }
    
    fn rect_intersection(&self, a: &core::Rect, b: &core::Rect) -> i32 {
        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.width).min(b.x + b.width);
        let y2 = (a.y + a.height).min(b.y + b.height);
        
        if x2 < x1 || y2 < y1 {
            0
        } else {
            (x2 - x1) * (y2 - y1)
        }
    }
    
    fn get_objects(&self) -> Vec<&TrackedObject> {
        self.objects.iter().collect()
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

    // Load multiple cascade classifiers for better detection
    let face_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    let profile_face_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml";
    
    let mut face_cascade = match objdetect::CascadeClassifier::new(face_cascade_path) {
        Ok(cascade) => {
            println!("Frontal face detection: ENABLED");
            Some(cascade)
        }
        Err(e) => {
            println!("Frontal face detection: DISABLED ({})", e);
            None
        }
    };
    
    let mut profile_face_cascade = match objdetect::CascadeClassifier::new(profile_face_cascade_path) {
        Ok(cascade) => {
            println!("Profile face detection: ENABLED");
            Some(cascade)
        }
        Err(e) => {
            println!("Profile face detection: DISABLED ({})", e);
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

    // Get frame size for tracker
    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let mut tracker = ObjectTracker::new(frame_width, frame_height);

    highgui::named_window(WINDOW, highgui::WINDOW_AUTOSIZE)?;

    let mut frame_count = 0u64;
    let mut last_key_check = Instant::now();
    let debounce_delay = Duration::from_millis(250);
    let mut frame = Mat::default();

    println!("Starting improved object detection. Press Q or ESC to quit.");
    println!("If PTZ is enabled: Arrow keys for pan/tilt, H for calibrate");

    loop {
        if !cap.read(&mut frame)? || frame.size()?.width == 0 {
            println!("End of stream or empty frame");
            break;
        }

        frame_count += 1;
        
        // Run detection on every frame (but tracker will handle smoothing)
        let mut all_detections = Vec::new();
        let mut all_confidences = Vec::new();
        
        if frame_count % 2 == 0 { // Run detection every other frame to balance CPU usage
            if let (Some(frontal_cascade), Some(profile_cascade)) = (&mut face_cascade, &mut profile_face_cascade) {
                let mut gray = Mat::default();
                imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
                
                let mut gray_eq = Mat::default();
                imgproc::equalize_hist(&gray, &mut gray_eq)?;
                
                // Detect faces with multiple classifiers
                let mut frontal_faces = core::Vector::<core::Rect>::new();
                let mut profile_faces = core::Vector::<core::Rect>::new();
                
                frontal_cascade.detect_multi_scale(
                    &gray_eq,
                    &mut frontal_faces,
                    1.05, // Lower scale factor = more precise but slower
                    4,    // Higher min neighbors = fewer false positives
                    objdetect::CASCADE_SCALE_IMAGE,
                    core::Size::new(40, 40), // Larger min size
                    core::Size::new(0, 0),
                ).ok();
                
                profile_cascade.detect_multi_scale(
                    &gray_eq,
                    &mut profile_faces,
                    1.05,
                    4,
                    objdetect::CASCADE_SCALE_IMAGE,
                    core::Size::new(40, 40),
                    core::Size::new(0, 0),
                ).ok();
                
                // Combine detections with confidence scores
                for face in &frontal_faces {
                    all_detections.push(*face);
                    all_confidences.push(0.8); // Higher confidence for frontal faces
                }
                
                for face in &profile_faces {
                    all_detections.push(*face);
                    all_confidences.push(0.6); // Lower confidence for profile faces
                }
                
                // Apply non-maximum suppression to remove overlapping detections
                let (filtered_detections, filtered_confidences) = 
                    non_maximum_suppression(&all_detections, &all_confidences, 0.3);
                
                all_detections = filtered_detections;
                all_confidences = filtered_confidences;
            }
        }
        
        // Update tracker with new detections
        tracker.update(all_detections, all_confidences);
        
        // Get tracked objects
        let tracked_objects = tracker.get_objects();
        
        // Draw tracked objects with history
        let mut display_frame = frame.clone();
        
        for obj in tracked_objects {
            // Determine color based on stability
            let color = if obj.stable {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green for stable
            } else {
                core::Scalar::new(255.0, 255.0, 0.0, 0.0) // Yellow for new/unstable
            };
            
            let bbox = obj.bbox;
            let rect = core::Rect::new(bbox.x, bbox.y, bbox.width, bbox.height);
            
            // Draw bounding box
            let thickness = if obj.stable { 3 } else { 2 };
            imgproc::rectangle(&mut display_frame, rect, color, thickness, imgproc::LINE_8, 0)?;
            
            // Draw tracking ID and confidence
            let label = format!("ID:{} ({:.1}%)", obj.id, obj.confidence * 100.0);
            let label_pos = core::Point::new(bbox.x, bbox.y - 5);
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
            
            // Draw center point and history trail
            if obj.history.len() > 1 {
                for i in 1..obj.history.len() {
                    let prev = obj.history[i-1];
                    let curr = obj.history[i];
                    
                    // Draw trail with fading color
                    let trail_color = core::Scalar::new(
                        0.0,
                        255.0 * (i as f32 / obj.history.len() as f32),
                        0.0,
                        0.0,
                    );
                    
                    imgproc::line(
                        &mut display_frame,
                        prev,
                        curr,
                        trail_color,
                        2,
                        imgproc::LINE_AA,
                        0,
                    )?;
                }
                
                // Draw current center
                let center = obj.center();
                imgproc::circle(
                    &mut display_frame,
                    center,
                    3,
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Red center point
                    2,
                    imgproc::LINE_AA,
                    0,
                )?;
            }
        }

        // Draw statistics
        let stats = format!(
            "Frame: {} | Tracked Objects: {}",
            frame_count,
            tracked_objects.len()
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
        
        // Draw tracking info
        let tracking_info = format!(
            "Stable: {} | New: {}",
            tracked_objects.iter().filter(|o| o.stable).count(),
            tracked_objects.iter().filter(|o| !o.stable).count(),
        );
        
        imgproc::put_text(
            &mut display_frame,
            &tracking_info,
            core::Point::new(10, 60),
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
            let key = highgui::wait_key(30)?;
            
            match key {
                113 | 27 => break, // 'q' or ESC
                104 | 72 if camera.is_some() => { // 'h' or 'H'
                    if let Some(cam) = &camera {
                        let mut cam_lock = cam.lock().await;
                        if let Err(e) = cam_lock.calibrate().await {
                            println!("Failed to calibrate: {}", e);
                        } else {
                            println!("Calibrating camera...");
                        }
                    }
                }
                81 | 82 if camera.is_some() => { // Left/Right arrow keys (Linux keycodes)
                    if let Some(cam) = &camera {
                        let mut cam_lock = cam.lock().await;
                        let _ = cam_lock.move_motor(-PAN_SPEED, 0).await;
                    }
                }
                83 | 84 if camera.is_some() => { // Up/Down arrow keys (Linux keycodes)
                    if let Some(cam) = &camera {
                        let mut cam_lock = cam.lock().await;
                        let _ = cam_lock.move_motor(0, TILT_SPEED).await;
                    }
                }
                _ => {}
            }
            
            last_key_check = now;
        }
    }

    highgui::destroy_window(WINDOW)?;
    println!("Application closed.");
    Ok(())
}

// Non-maximum suppression to remove overlapping detections
fn non_maximum_suppression(
    detections: &[core::Rect],
    confidences: &[f32],
    overlap_threshold: f32,
) -> (Vec<core::Rect>, Vec<f32>) {
    if detections.is_empty() {
        return (Vec::new(), Vec::new());
    }
    
    // Sort by confidence (highest first)
    let mut indices: Vec<usize> = (0..detections.len()).collect();
    indices.sort_by(|&a, &b| confidences[b].partial_cmp(&confidences[a]).unwrap());
    
    let mut picked = Vec::new();
    let mut suppressed = vec![false; detections.len()];
    
    for i in 0..indices.len() {
        if suppressed[indices[i]] {
            continue;
        }
        
        picked.push(indices[i]);
        
        for j in (i + 1)..indices.len() {
            if suppressed[indices[j]] {
                continue;
            }
            
            let iou = calculate_iou_simple(&detections[indices[i]], &detections[indices[j]]);
            if iou > overlap_threshold {
                suppressed[indices[j]] = true;
            }
        }
    }
    
    let filtered_detections: Vec<core::Rect> = picked.iter().map(|&idx| detections[idx]).collect();
    let filtered_confidences: Vec<f32> = picked.iter().map(|&idx| confidences[idx]).collect();
    
    (filtered_detections, filtered_confidences)
}

fn calculate_iou_simple(a: &core::Rect, b: &core::Rect) -> f32 {
    let intersection_x = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
    let intersection_y = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
    
    if intersection_x <= 0.0 || intersection_y <= 0.0 {
        return 0.0;
    }
    
    let intersection = (intersection_x as f32) * (intersection_y as f32);
    let area_a = (a.width as f32) * (a.height as f32);
    let area_b = (b.width as f32) * (b.height as f32);
    
    intersection / (area_a + area_b - intersection)
}