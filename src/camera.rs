use anyhow::{bail, Result};
use reqwest::Client;
use serde_json::{json, Value};
use sha2::Sha256;
use sha2::Digest as _;

pub struct TapoCamera {
    client: Client,
    base_url: String,
    stok: String,
}

impl TapoCamera {
    pub async fn connect(ip: &str, email: &str, password: &str) -> Result<Self> {
        let client = Client::builder()
            .danger_accept_invalid_certs(true)
            .timeout(std::time::Duration::from_secs(5))
            .build()?;

        let base_url = format!("https://{ip}");
        let stok = Self::login(&client, &base_url, email, password).await?;
        Ok(Self { client, base_url, stok })
    }

    async fn login(client: &Client, base_url: &str, email: &str, password: &str) -> Result<String> {
        // Pre-compute hash variants used by different firmware versions
        let md5_upper  = format!("{:x}", md5::compute(password.as_bytes())).to_uppercase();
        let md5_lower  = format!("{:x}", md5::compute(password.as_bytes()));
        let sha256_lower = format!("{:x}", Sha256::digest(password.as_bytes()));
        let sha256_upper = sha256_lower.to_uppercase();

        // Ordered from most likely (pytapo / C200 community findings) to least likely.
        // Tapo cameras differ by firmware: some want "admin", some want the cloud email;
        // most C200 units want MD5-uppercase with username "admin".
        let candidates: &[(&str, &str, bool)] = &[
            ("admin", &md5_upper,   true),   // pytapo default — works on most C200 firmware
            (email,   &md5_upper,   true),   // same hash, cloud email as username
            ("admin", &sha256_upper, true),  // SHA256 uppercase, admin
            (email,   &sha256_lower, true),  // SHA256 lowercase, cloud email
            (email,   &md5_lower,   true),   // MD5 lowercase, cloud email
            ("admin", password,     false),  // plain password, admin (some older units)
            (email,   password,     false),  // plain password, cloud email
        ];

        for (username, pw, hashed) in candidates {
            let mut params = json!({ "username": username, "password": pw });
            if *hashed {
                params["hashed"] = json!(true);
            }
            let resp: Value = match client
                .post(base_url)
                .json(&json!({ "method": "login", "params": params }))
                .send()
                .await
            {
                Ok(r) => match r.json().await {
                    Ok(v) => v,
                    Err(e) => { eprintln!("Login parse error: {e}"); continue; }
                },
                Err(e) => { eprintln!("Login request error: {e}"); continue; }
            };

            if resp["error_code"].as_i64().unwrap_or(-1) == 0 {
                let stok = resp["result"]["stok"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("No stok in login response"))?
                    .to_string();
                println!("Auth OK  (username={username}, hashed={hashed})");
                return Ok(stok);
            }
            eprintln!("Auth fail (username={username}, hashed={hashed}): code {}",
                resp["error_code"]);
        }

        bail!(
            "All login attempts failed.\n\
             Tip: in the Tapo app go to Camera Settings → Advanced → Camera Account \
             and check/set the local username and password, then update .env."
        )
    }

    fn api_url(&self) -> String {
        format!("{}/stok={}/ds", self.base_url, self.stok)
    }

    async fn request(&self, body: Value) -> Result<Value> {
        let resp: Value = self
            .client
            .post(self.api_url())
            .json(&body)
            .send()
            .await?
            .json()
            .await?;
        if resp["error_code"].as_i64().unwrap_or(-1) != 0 {
            bail!("Camera API error: {}", resp["error_code"]);
        }
        Ok(resp)
    }

    pub async fn get_device_info(&self) -> Result<Value> {
        self.request(json!({
            "method": "get",
            "device_info": {"name": ["basic_info"]}
        })).await
    }

    /// Pan/tilt: x_speed and y_speed are in -100..100.
    /// Positive x = pan right, positive y = tilt up.
    pub async fn move_motor(&self, x_speed: i32, y_speed: i32) -> Result<()> {
        self.request(json!({
            "method": "do",
            "motor": {"move": {"x_speed": x_speed, "y_speed": y_speed}}
        })).await?;
        Ok(())
    }

    pub async fn calibrate(&self) -> Result<()> {
        self.request(json!({
            "method": "do",
            "motor": {"manual_cali": {}}
        })).await?;
        Ok(())
    }
}
