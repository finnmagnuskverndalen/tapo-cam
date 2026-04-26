use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct Rect {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
}

#[derive(Debug, Clone)]
struct TrackedObject {
    id: u32,
    bbox: Rect,
    confidence: f32,
    last_seen: Instant,
    history: Vec<(i32, i32)>,
    stable: bool,
}

impl TrackedObject {
    fn new(id: u32, bbox: Rect, confidence: f32) -> Self {
        let center = (bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
        Self {
            id,
            bbox,
            confidence,
            last_seen: Instant::now(),
            history: vec![center],
            stable: false,
        }
    }

    fn update(&mut self, new_bbox: Rect, confidence: f32) {
        let new_center = (new_bbox.x + new_bbox.width / 2, new_bbox.y + new_bbox.height / 2);
        
        // Smooth the bbox position using moving average
        self.history.push(new_center);
        if self.history.len() > 10 {
            self.history.remove(0);
        }
        
        // Calculate smoothed center
        let sum_x: i32 = self.history.iter().map(|p| p.0).sum();
        let sum_y: i32 = self.history.iter().map(|p| p.1).sum();
        let avg_x = sum_x / self.history.len() as i32;
        let avg_y = sum_y / self.history.len() as i32;
        
        // Update bbox with smoothed position
        let half_width = new_bbox.width / 2;
        let half_height = new_bbox.height / 2;
        self.bbox = Rect::new(avg_x - half_width, avg_y - half_height, new_bbox.width, new_bbox.height);
        
        self.confidence = (self.confidence * 0.7) + (confidence * 0.3);
        self.last_seen = Instant::now();
        
        if self.history.len() >= 5 {
            self.stable = true;
        }
    }

    fn should_remove(&self) -> bool {
        let since_seen = Instant::now().duration_since(self.last_seen);
        since_seen > Duration::from_secs(2)
    }
}

impl Rect {
    fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self { x, y, width, height }
    }
}

fn calculate_iou(a: &Rect, b: &Rect) -> f32 {
    let intersection_x = (a.x + a.width).min(b.x + b.width) - a.x.max(b.x);
    let intersection_y = (a.y + a.height).min(b.y + b.height) - a.y.max(b.y);
    
    if intersection_x <= 0 || intersection_y <= 0 {
        return 0.0;
    }
    
    let intersection = intersection_x * intersection_y;
    let area_a = a.width * a.height;
    let area_b = b.width * b.height;
    
    intersection as f32 / (area_a + area_b - intersection) as f32
}

fn main() {
    println!("Testing tracking logic...");
    
    // Create initial object
    let mut obj = TrackedObject::new(
        1,
        Rect::new(100, 100, 50, 50),
        0.8
    );
    
    println!("Initial object: {:?}", obj);
    
    // Simulate updates with slight movement
    let movements = [
        (Rect::new(102, 102, 50, 50), 0.85),
        (Rect::new(105, 105, 50, 50), 0.82),
        (Rect::new(108, 108, 50, 50), 0.79),
        (Rect::new(110, 110, 50, 50), 0.83),
    ];
    
    for (bbox, confidence) in movements.iter() {
        std::thread::sleep(std::time::Duration::from_millis(100));
        obj.update(bbox.clone(), *confidence);
        println!("Updated object: ID={}, Center=({},{}), Confidence={:.1}%, Stable={}", 
            obj.id,
            obj.bbox.x + obj.bbox.width / 2,
            obj.bbox.y + obj.bbox.height / 2,
            obj.confidence * 100.0,
            obj.stable
        );
    }
    
    // Test IoU calculation
    let rect1 = Rect::new(0, 0, 100, 100);
    let rect2 = Rect::new(50, 50, 100, 100);
    let iou = calculate_iou(&rect1, &rect2);
    println!("\nIoU test: rect1 and rect2 overlap with IoU = {:.2}", iou);
    
    // Test non-overlapping
    let rect3 = Rect::new(0, 0, 50, 50);
    let rect4 = Rect::new(100, 100, 50, 50);
    let iou2 = calculate_iou(&rect3, &rect4);
    println!("IoU test: rect3 and rect4 don't overlap, IoU = {:.2}", iou2);
    
    println!("\nTracking test complete!");
}