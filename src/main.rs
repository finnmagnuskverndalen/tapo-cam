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
const WINDOW: &str = "Tapo C200 - Detection";
const PAN_SPEED: i32 = 40;
const TILT_SPEED: i32 = 40;

const KEY_LEFT: i32 = 65361;
const KEY_UP: i32 = 65362;
const KEY_RIGHT: i32 = 65363;
const KEY_DOWN: i32 = 65364;

// How often to run each detector (in frames)
const FACE_EVERY_N: u64 = 3;
const PERSON_EVERY_N: u64 = 6;

// Scale factor for downscaling before detection (faster Haar + HOG)
const DETECT_SCALE: f64 = 0.5;

#[derive(Clone)]
struct Detection {
    rect: core::Rect,
    label: &'static str,
    color: core::Scalar,
}

struct Detector {
    face_frontal: objdetect::CascadeClassifier,
    face_profile: Option<objdetect::CascadeClassifier>,
    hog: objdetect::HOGDescriptor,
}

impl Detector {
    fn new() -> Result<Self> {
        let base = "/usr/share/opencv4/haarcascades/";

        let face_frontal = objdetect::CascadeClassifier::new(
            &format!("{}haarcascade_frontalface_default.xml", base),
        )?;

        // Profile cascade detects sideways-facing people
        let face_profile = match objdetect::CascadeClassifier::new(
            &format!("{}haarcascade_profileface.xml", base),
        ) {
            Ok(c) => { println!("Profile face cascade: loaded"); Some(c) }
            Err(_) => { println!("Profile face cascade: not found (frontal only)"); None }
        };

        // HOG person detector — built-in, no model files needed, far better than Haar body cascades
        let mut hog = objdetect::HOGDescriptor::default()?;
        let people_svm = objdetect::HOGDescriptor::get_default_people_detector()?;
        hog.set_svm_detector_vec(people_svm);
        println!("HOG person detector: ready");

        Ok(Self { face_frontal, face_profile, hog })
    }

    fn detect_faces(&mut self, frame: &Mat) -> Result<Vec<Detection>> {
        let mut small = Mat::default();
        imgproc::resize(frame, &mut small, core::Size::default(),
            DETECT_SCALE, DETECT_SCALE, imgproc::INTER_LINEAR)?;

        let mut gray = Mat::default();
        imgproc::cvt_color(&small, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        let mut gray_eq = Mat::default();
        imgproc::equalize_hist(&gray, &mut gray_eq)?;

        let min_sz = core::Size::new(15, 15); // 30px in original
        let inv = 1.0 / DETECT_SCALE;
        let mut detections = Vec::new();

        // Frontal faces
        let mut frontal = core::Vector::<core::Rect>::new();
        self.face_frontal.detect_multi_scale(
            &gray_eq, &mut frontal,
            1.1, 4, 0, min_sz, core::Size::new(0, 0),
        )?;
        for r in frontal.iter() {
            detections.push(Detection {
                rect: scale_rect(r, inv),
                label: "Face",
                color: core::Scalar::new(0.0, 230.0, 0.0, 0.0), // Green
            });
        }

        // Profile faces — skip if heavily overlaps with an already-found frontal face
        if let Some(profile) = &mut self.face_profile {
            let mut profiles = core::Vector::<core::Rect>::new();
            profile.detect_multi_scale(
                &gray_eq, &mut profiles,
                1.1, 4, 0, min_sz, core::Size::new(0, 0),
            )?;
            for p in profiles.iter() {
                let scaled = scale_rect(p, inv);
                if !detections.iter().any(|d| iou(&d.rect, &scaled) > 0.3) {
                    detections.push(Detection {
                        rect: scaled,
                        label: "Face",
                        color: core::Scalar::new(50.0, 200.0, 0.0, 0.0), // Slightly different green
                    });
                }
            }
        }

        Ok(detections)
    }

    fn detect_persons(&mut self, frame: &Mat) -> Result<Vec<Detection>> {
        // HOG works on BGR image directly; scale slightly smaller than face detection for speed
        let hog_scale = 0.4f64;
        let mut small = Mat::default();
        imgproc::resize(frame, &mut small, core::Size::default(),
            hog_scale, hog_scale, imgproc::INTER_LINEAR)?;

        let mut found = core::Vector::<core::Rect>::new();
        self.hog.detect_multi_scale(
            &small,
            &mut found,
            0.0,                       // hit_threshold: 0 = use trained default
            core::Size::new(8, 8),     // win_stride: smaller = more accurate but slower
            core::Size::new(0, 0),     // padding
            1.05,                      // scale: smaller = more levels = more accurate but slower
            2.0,                       // group_threshold: higher = fewer false positives
            false,                     // use_meanshift_grouping
        )?;

        let inv = 1.0 / hog_scale;
        Ok(found.iter().map(|r| Detection {
            rect: scale_rect(r, inv),
            label: "Person",
            color: core::Scalar::new(230.0, 180.0, 0.0, 0.0), // Cyan
        }).collect())
    }
}

fn scale_rect(r: core::Rect, inv: f64) -> core::Rect {
    core::Rect::new(
        (r.x as f64 * inv) as i32,
        (r.y as f64 * inv) as i32,
        (r.width as f64 * inv) as i32,
        (r.height as f64 * inv) as i32,
    )
}

fn iou(a: &core::Rect, b: &core::Rect) -> f32 {
    let ix = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
    let iy = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
    if ix <= 0 || iy <= 0 {
        return 0.0;
    }
    let inter = (ix * iy) as f32;
    let union = (a.width * a.height + b.width * b.height) as f32 - inter;
    inter / union
}

fn put_text_outlined(frame: &mut Mat, text: &str, pos: core::Point, scale: f64, color: core::Scalar) -> opencv::Result<()> {
    imgproc::put_text(frame, text, pos, imgproc::FONT_HERSHEY_SIMPLEX, scale,
        core::Scalar::new(0.0, 0.0, 0.0, 0.0), 3, imgproc::LINE_AA, false)?;
    imgproc::put_text(frame, text, pos, imgproc::FONT_HERSHEY_SIMPLEX, scale,
        color, 1, imgproc::LINE_AA, false)?;
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

    let camera = match TapoCamera::connect(CAM_IP, &email, &password).await {
        Ok(cam) => {
            let info = cam.get_device_info().await?;
            println!("Connected: {}", info["device_info"]["basic_info"]["device_alias"]
                .as_str().unwrap_or("C200"));
            println!("PTZ controls: ENABLED");
            Some(Arc::new(Mutex::new(cam)))
        }
        Err(e) => {
            println!("Camera auth failed: {}. PTZ disabled, streaming only.", e);
            None
        }
    };

    println!("Initializing detectors...");
    let mut detector = Detector::new()?;
    println!("Detectors ready");

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
        anyhow::bail!("Failed to open RTSP stream. Check credentials and network.");
    }
    cap.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let stream_fps = cap.get(videoio::CAP_PROP_FPS)?;
    println!("Stream: {}x{} @ {:.1} FPS", frame_width, frame_height, stream_fps);

    highgui::named_window(WINDOW, highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut frame_count = 0u64;
    let mut last_faces: Vec<Detection> = Vec::new();
    let mut last_persons: Vec<Detection> = Vec::new();

    let mut ptz_debounce = Instant::now() - Duration::from_secs(1);
    let ptz_debounce_ms = Duration::from_millis(200);

    let mut fps_counter = 0u32;
    let mut last_fps_time = Instant::now();
    let mut display_fps = 0.0f32;

    println!("Running. Q or ESC to quit.");
    println!("Detection: Green=Face, Cyan=Person");

    loop {
        if !cap.read(&mut frame)? || frame.size()?.width == 0 {
            println!("Stream ended or empty frame");
            break;
        }

        frame_count += 1;
        fps_counter += 1;

        let now = Instant::now();
        let elapsed = now.duration_since(last_fps_time).as_secs_f32();
        if elapsed >= 1.0 {
            display_fps = fps_counter as f32 / elapsed;
            fps_counter = 0;
            last_fps_time = now;
        }

        // Face detection: runs every FACE_EVERY_N frames
        if frame_count % FACE_EVERY_N == 0 {
            match detector.detect_faces(&frame) {
                Ok(faces) => last_faces = faces,
                Err(e) => eprintln!("Face detection error: {}", e),
            }
        }

        // Person detection: runs every PERSON_EVERY_N frames (heavier than face detection)
        if frame_count % PERSON_EVERY_N == 0 {
            match detector.detect_persons(&frame) {
                Ok(persons) => last_persons = persons,
                Err(e) => eprintln!("Person detection error: {}", e),
            }
        }

        let mut display = frame.clone();

        // Draw person boxes first (behind face boxes)
        for det in &last_persons {
            imgproc::rectangle(&mut display, det.rect, det.color, 2, imgproc::LINE_8, 0)?;
            put_text_outlined(&mut display, det.label,
                core::Point::new(det.rect.x, det.rect.y - 5),
                0.5, det.color)?;
        }

        // Draw face boxes on top
        for (i, det) in last_faces.iter().enumerate() {
            imgproc::rectangle(&mut display, det.rect, det.color, 2, imgproc::LINE_8, 0)?;
            put_text_outlined(&mut display,
                &format!("{} {}", det.label, i + 1),
                core::Point::new(det.rect.x, det.rect.y - 5),
                0.5, det.color)?;
        }

        // HUD — always visible every frame
        put_text_outlined(
            &mut display,
            &format!("FPS: {:.1}  Faces: {}  Persons: {}  Frame: {}",
                display_fps, last_faces.len(), last_persons.len(), frame_count),
            core::Point::new(10, 30),
            0.6,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        )?;
        put_text_outlined(
            &mut display,
            if camera.is_some() { "Q/ESC: Quit | H: Home | Arrows: Pan/Tilt" }
                               else { "Q/ESC: Quit | PTZ: Disabled" },
            core::Point::new(10, frame_height - 20),
            0.5,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        )?;

        highgui::imshow(WINDOW, &display)?;

        // Must be called every frame for the GUI to process events
        let key = highgui::wait_key(1)?;
        match key {
            113 | 27 => break, // q or ESC
            _ => {}
        }

        // PTZ commands with debounce
        if Instant::now().duration_since(ptz_debounce) > ptz_debounce_ms {
            if let Some(cam) = &camera {
                let handled = match key {
                    104 | 72 => {
                        cam.lock().await.calibrate().await.ok();
                        println!("Calibrating...");
                        true
                    }
                    KEY_LEFT  => { cam.lock().await.move_motor(-PAN_SPEED, 0).await.ok(); true }
                    KEY_RIGHT => { cam.lock().await.move_motor(PAN_SPEED, 0).await.ok(); true }
                    KEY_UP    => { cam.lock().await.move_motor(0, TILT_SPEED).await.ok(); true }
                    KEY_DOWN  => { cam.lock().await.move_motor(0, -TILT_SPEED).await.ok(); true }
                    _ => false,
                };
                if handled {
                    ptz_debounce = Instant::now();
                }
            }
        }
    }

    highgui::destroy_window(WINDOW)?;
    println!("Closed after {} frames.", frame_count);
    Ok(())
}
