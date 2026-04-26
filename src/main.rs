mod camera;

use anyhow::Result;
use camera::TapoCamera;
use opencv::{highgui, imgproc, prelude::*, videoio};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const CAM_IP: &str = "192.168.10.185";
// RTSP_URL will be constructed from environment variables
const WINDOW: &str = "Tapo C200 - Remote Control";
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

    let camera = TapoCamera::connect(CAM_IP, &email, &password).await?;
    let info = camera.get_device_info().await?;
    println!("Connected: {}", info["device_info"]["basic_info"]["device_alias"]
        .as_str()
        .unwrap_or("C200"));

    let camera = Arc::new(Mutex::new(camera));

    // Construct RTSP URL from credentials
    let rtsp_url = format!("rtsp://{}:{}@{}:554/stream1", email, password, CAM_IP);
    println!("Opening RTSP stream: rtsp://{}:****@{}:554/stream1", email, CAM_IP);
    
    let mut cap = videoio::VideoCapture::from_file(&rtsp_url, videoio::CAP_FFMPEG)?;
    if !cap.is_opened()? {
        anyhow::bail!("Failed to open RTSP stream. Check credentials and network connection.");
    }

    highgui::named_window(WINDOW, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(WINDOW, 1280, 720)?;

    println!("Controls: Arrow keys = pan/tilt | H = home | Q/ESC = quit");

    let mut frame = Mat::default();
    let mut last_move = Instant::now();
    let debounce = Duration::from_millis(250);

    loop {
        cap.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        imgproc::put_text(
            &mut frame,
            "Arrows: pan/tilt | H: home | Q: quit",
            opencv::core::Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        highgui::imshow(WINDOW, &frame)?;

        let key = highgui::wait_key(1)?;
        let now = Instant::now();

        if now.duration_since(last_move) >= debounce && key > 0 {
            let cam = camera.clone();
            let moved = match key {
                65361 => { // Left — pan left
                    tokio::spawn(async move {
                        let c = cam.lock().await;
                        let _ = c.move_motor(-PAN_SPEED, 0).await;
                    });
                    true
                }
                65363 => { // Right — pan right
                    tokio::spawn(async move {
                        let c = cam.lock().await;
                        let _ = c.move_motor(PAN_SPEED, 0).await;
                    });
                    true
                }
                65362 => { // Up — tilt up
                    tokio::spawn(async move {
                        let c = cam.lock().await;
                        let _ = c.move_motor(0, TILT_SPEED).await;
                    });
                    true
                }
                65364 => { // Down — tilt down
                    tokio::spawn(async move {
                        let c = cam.lock().await;
                        let _ = c.move_motor(0, -TILT_SPEED).await;
                    });
                    true
                }
                104 | 72 => { // H/h — home
                    tokio::spawn(async move {
                        let c = cam.lock().await;
                        let _ = c.calibrate().await;
                    });
                    true
                }
                _ => false,
            };
            if moved {
                last_move = now;
            }
        }

        if key == 113 || key == 27 {
            println!("Exiting.");
            break;
        }
    }

    Ok(())
}
