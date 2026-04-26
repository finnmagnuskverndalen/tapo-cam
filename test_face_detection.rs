use opencv::{core, highgui, imgproc, objdetect, prelude::*};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test face cascade loading
    let cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    
    println!("Testing face cascade loading from: {}", cascade_path);
    
    match objdetect::CascadeClassifier::new(cascade_path) {
        Ok(mut cascade) => {
            println!("✅ Face cascade loaded successfully!");
            
            // Test if cascade is empty
            if cascade.empty()? {
                println!("⚠️ Cascade classifier is empty (no data loaded)");
            } else {
                println!("✅ Cascade classifier has data");
            }
            
            // Try to load a test image if available
            let test_image = "/usr/share/opencv4/samples/data/lena.jpg";
            if std::path::Path::new(test_image).exists() {
                println!("\nTesting face detection on sample image...");
                
                let mut img = opencv::imgcodecs::imread(test_image, opencv::imgcodecs::IMREAD_COLOR)?;
                let mut gray = Mat::default();
                imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
                
                let mut faces = core::Vector::<core::Rect>::new();
                cascade.detect_multi_scale(
                    &gray,
                    &mut faces,
                    1.1,
                    3,
                    objdetect::CASCADE_SCALE_IMAGE,
                    core::Size::new(30, 30),
                    core::Size::new(0, 0),
                )?;
                
                println!("✅ Detected {} faces in sample image", faces.len());
                
                // Draw faces
                for face in &faces {
                    let rect = core::Rect::new(face.x, face.y, face.width, face.height);
                    imgproc::rectangle(&mut img, rect, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
                }
                
                highgui::named_window("Face Detection Test", highgui::WINDOW_AUTOSIZE)?;
                highgui::imshow("Face Detection Test", &img)?;
                println!("Press any key to close window...");
                highgui::wait_key(0)?;
                highgui::destroy_window("Face Detection Test")?;
            } else {
                println!("ℹ️ Sample image not found at: {}", test_image);
                println!("Try: sudo apt-get install opencv-data");
            }
        }
        Err(e) => {
            println!("❌ Failed to load face cascade: {}", e);
            println!("\nPossible solutions:");
            println!("1. Install opencv-data: sudo apt-get install opencv-data");
            println!("2. Check if cascade files exist: ls /usr/share/opencv4/haarcascades/");
            println!("3. Try different path: /usr/share/opencv/haarcascades/");
            return Err(e.into());
        }
    }
    
    // Test other cascade files
    println!("\n--- Testing other cascade files ---");
    
    let cascades = [
        ("Full body", "haarcascade_fullbody.xml"),
        ("Upper body", "haarcascade_upperbody.xml"),
        ("Cat face", "haarcascade_frontalcatface.xml"),
    ];
    
    for (name, file) in &cascades {
        let path = format!("/usr/share/opencv4/haarcascades/{}", file);
        match objdetect::CascadeClassifier::new(&path) {
            Ok(cascade) => {
                if !cascade.empty().unwrap_or(true) {
                    println!("✅ {} cascade loaded: {}", name, file);
                } else {
                    println!("⚠️ {} cascade empty: {}", name, file);
                }
            }
            Err(_) => {
                println!("❌ {} cascade not found: {}", name, file);
            }
        }
    }
    
    println!("\n✅ All tests completed!");
    Ok(())
}