mod camera;

use anyhow::Result;
use camera::TapoCamera;
use opencv::{
    core, dnn, highgui, imgproc, prelude::*, videoio,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const CAM_IP: &str = "192.168.10.185";
const WINDOW: &str = "Tapo C200 - Object Detection (People/Animals)";
const PAN_SPEED: i32 = 40;
const TILT_SPEED: i32 = 40;

// MobileNet SSD model paths (these files need to be downloaded)
const MODEL_CONFIG: &str = "models/MobileNetSSD_deploy.prototxt";
const MODEL_WEIGHTS: &str = "models/MobileNetSSD_deploy.caffemodel";

// COCO class labels (what MobileNet SSD detects)
const CLASSES: [&str; 21] = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
];

// Colors for different classes
const CLASS_COLORS: [(f64, f64, f64); 21] = [
    (0.0, 0.0, 0.0),        // background - black
    (255.0, 0.0, 0.0),      // aeroplane - blue
    (0.0, 255.0, 0.0),      // bicycle - green
    (0.0, 0.0, 255.0),      // bird - red
    (255.0, 255.0, 0.0),    // boat - cyan
    (255.0, 0.0, 255.0),    // bottle - magenta
    (0.0, 255.0, 255.0),    // bus - yellow
    (128.0, 0.0, 0.0),      // car - dark blue
    (0.0, 128.0, 0.0),      // cat - dark green
    (0.0, 0.0, 128.0),      // chair - dark red
    (128.0, 128.0, 0.0),    // cow - dark cyan
    (128.0, 0.0, 128.0),    // diningtable - dark magenta
    (0.0, 128.0, 128.0),    // dog - dark yellow
    (255.0, 128.0, 0.0),    // horse - orange
    (128.0, 255.0, 0.0),    // motorbike - lime
    (255.0, 0.0, 128.0),    // person - pink (we care about this!)
    (0.0, 255.0, 128.0),    // pottedplant - spring green
    (128.0, 0.0, 255.0),    // sheep - purple
    (255.0, 128.0, 128.0),  // sofa - light pink
    (128.0, 255.0, 128.0),  // train - light green
    (128.0, 128.0, 255.0),  // tvmonitor - light blue
];

struct ObjectDetector {
    net: dnn::Net,
    input_size: core::Size,
    scale_factor: f64,
    mean: core::Scalar,
    swap_rb: bool,
    confidence_threshold: f32,
    nms_threshold: f32,
}

impl ObjectDetector {
    fn new(conf_thresh: f32, nms_thresh: f32) -> Result<Self> {
        // Check if model files exist
        if !std::path::Path::new(MODEL_CONFIG).exists() {
            anyhow::bail!("Model config file not found: {}. Please download it.", MODEL_CONFIG);
        }
        if !std::path::Path::new(MODEL_WEIGHTS).exists() {
            anyhow::bail!("Model weights file not found: {}. Please download it.", MODEL_WEIGHTS);
        }

        // Load the model
        let net = dnn::read_net_from_caffe(MODEL_CONFIG, MODEL_WEIGHTS)?;
        
        Ok(Self {
            net,
            input_size: core::Size::new(300, 300), // MobileNet SSD expects 300x300
            scale_factor: 1.0 / 127.5, // Scale from [0,255] to [-1,1]
            mean: core::Scalar::new(127.5, 127.5, 127.5, 0.0),
            swap_rb: true, // OpenCV loads as BGR, model expects RGB
            confidence_threshold: conf_thresh,
            nms_threshold: nms_thresh,
        })
    }

    fn detect(&mut self, frame: &Mat) -> Result<Vec<(core::Rect, &'static str, f32)>> {
        let mut blob = Mat::default();
        
        // Create blob from image
        dnn::blob_from_image(
            frame,
            &mut blob,
            self.scale_factor,
            self.input_size,
            &self.mean,
            self.swap_rb,
            false,
            core::CV_32F,
        )?;
        
        // Set input
        self.net.set_input(&blob, "", 1.0, core::Scalar::default())?;
        
        // Forward pass
        let mut output = Mat::default();
        self.net.forward(&mut output, &["detection_out".to_string()])?;
        
        // Parse detections
        let mut detections = Vec::new();
        
        // output shape: [1, 1, N, 7] where 7 = [batchId, classId, confidence, left, top, right, bottom]
        let num_detections = output.size(2)?;
        
        for i in 0..num_detections as usize {
            let confidence = *output.at_2d::<f32>(0, 0, i as i32, 2)?;
            
            if confidence > self.confidence_threshold {
                let class_id = (*output.at_2d::<f32>(0, 0, i as i32, 1)?) as usize;
                
                if class_id < CLASSES.len() {
                    let left = (*output.at_2d::<f32>(0, 0, i as i32, 3)?) * frame.cols() as f32;
                    let top = (*output.at_2d::<f32>(0, 0, i as i32, 4)?) * frame.rows() as f32;
                    let right = (*output.at_2d::<f32>(0, 0, i as i32, 5)?) * frame.cols() as f32;
                    let bottom = (*output.at_2d::<f32>(0, 0, i as i32, 6)?) * frame.rows() as f32;
                    
                    let bbox = core::Rect::new(
                        left as i32,
                        top as i32,
                        (right - left) as i32,
                        (bottom - top) as i32,
                    );
                    
                    detections.push((bbox, CLASSES[class_id], confidence));
                }
            }
        }
        
        // Apply Non-Maximum Suppression to remove overlapping boxes
        Ok(self.apply_nms(detections))
    }
    
    fn apply_nms(&self, mut detections: Vec<(core::Rect, &'static str, f32)>) -> Vec<(core::Rect, &'static str, f32)> {
        if detections.is_empty() {
            return detections;
        }
        
        // Group by class
        let mut by_class: std::collections::HashMap<&str, Vec<(core::Rect, f32)>> = std::collections::HashMap::new();
        
        for (bbox, class_name, confidence) in detections.drain(..) {
            by_class.entry(class_name)
                .or_insert_with(Vec::new)
                .push((bbox, confidence));
        }
        
        let mut result = Vec::new();
        
        for (class_name, mut class_detections) in by_class {
            // Sort by confidence
            class_detections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let mut kept = vec![true; class_detections.len()];
            
            for i in 0..class_detections.len() {
                if !kept[i] {
                    continue;
                }
                
                for j in (i + 1)..class_detections.len() {
                    if !kept[j] {
                        continue;
                    }
                    
                    let iou = self.calculate_iou(&class_detections[i].0, &class_detections[j].0);
                    if iou > self.nms_threshold {
                        kept[j] = false;
                    }
                }
                
                if kept[i] {
                    result.push((class_detections[i].0, class_name, class_detections[i].1));
                }
            }
        }
        
        result
    }
    
    fn calculate_iou(&self, a: &core::Rect, b: &core::Rect) -> f32 {
        let intersection_x = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
        let intersection_y = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
        
        if intersection_x <= 0 || intersection_y <= 0 {
            return 0.0;
        }
        
        let intersection = (intersection_x as f32) * (intersection_y as f32);
        let area_a = (a.width as f32) * (a.height as f32);
        let area_b = (b.width as f32) * (b.height as f32);
        
        intersection / (area_a + area_b - intersection)
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

    // Initialize object detector
    println!("Initializing MobileNet SSD object detector...");
    let mut detector = ObjectDetector::new(0.5, 0.4)?; // 50% confidence threshold, 40% NMS threshold
    println!("Object detector ready. Will detect: person, dog, cat, bird, horse, sheep, cow");

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

    println!("Starting object detection. Press Q or ESC to quit.");
    println!("If PTZ is enabled: Arrow keys for pan/tilt, H for calibrate");

    // Performance tracking
    let mut fps_counter = 0u32;
    let mut last_fps_time = Instant::now();
    let mut processing_times = Vec::new();
    let mut detection_times = Vec::new();

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
        
        // Run object detection every 5th frame for better performance
        let mut detections = Vec::new();
        if frame_count % 5 == 0 {
            let detection_start = Instant::now();
            
            match detector.detect(&frame) {
                Ok(dets) => {
                    detections = dets;
                    
                    // Track detection time
                    let detection_time = detection_start.elapsed();
                    detection_times.push(detection_time);
                    if detection_times.len() > 30 {
                        detection_times.remove(0);
                    }
                }
                Err(e) => {
                    eprintln!("Object detection error: {}", e);
                }
            }
        }
        
        // Draw detections
        for (bbox, class_name, confidence) in &detections {
            let class_id = CLASSES.iter().position(|&c| c == *class_name).unwrap_or(0);
            let color = core::Scalar::new(
                CLASS_COLORS[class_id].2, // B
                CLASS_COLORS[class_id].1, // G  
                CLASS_COLORS[class_id].0, // R
                0.0,
            );
            
            // Draw bounding box
            imgproc::rectangle(&mut display_frame, *bbox, color, 2, imgproc::LINE_8, 0)?;
            
            // Draw label with confidence
            let label = format!("{}: {:.1}%", class_name, confidence * 100.0);
            let label_pos = core::Point::new(bbox.x, bbox.y - 5);
            
            // Draw filled background for label
            let (label_width, label_height) = imgproc::get_text_size(
                &label,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &mut 0,
            )?;
            
            let label_bg = core::Rect::new(
                bbox.x,
                bbox.y - label_height - 5,
                label_width,
                label_height + 5,
            );
            
            imgproc::rectangle(&mut display_frame, label_bg, color, -1, imgproc::LINE_8, 0)?;
            
            // Draw label text
            imgproc::put_text(
                &mut display_frame,
                &label,
                label_pos,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                1,
                imgproc::LINE_AA,
                false,
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
            
            // Calculate average detection time
            let avg_detection_time: Duration = if !detection_times.is_empty() {
                detection_times.iter().sum::<Duration>() / detection_times.len() as u32
            } else {
                Duration::from_millis(0)
            };
            
            // Draw performance info
            let avg_process_time: Duration = processing_times.iter().sum::<Duration>() / processing_times.len() as u32;
            let stats = format!(
                "FPS: {:.1} | Objects: {} | Process: {:.1}ms | Detect: {:.1}ms",
                fps,
                detections.len(),
                avg_process_time.as_secs_f32() * 1000.0,
                avg_detection_time.as_secs_f32() * 1000.0
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

        // Draw detection classes found
        let unique_classes: std::collections::HashSet<&str> = detections.iter().map(|(_, c, _)| *c).collect();
        if !unique_classes.is_empty() {
            let classes_str = format!("Detected: {}", unique_classes.iter().fold(String::new(), |acc, &c| {
                if acc.is_empty() { c.to_string() } else { format!("{}, {}", acc, c) }
            }));
            
            imgproc::put_text(
                &mut display_frame,
                &classes_str,
                core::Point::new(10, 90),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                core::Scalar::new(255.0, 255.0, 0.0, 0.0), // Yellow for detected classes
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }

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