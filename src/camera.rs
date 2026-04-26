use anyhow::{bail, Result};
use md5::Md5;
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
        // Camera uses a self-signed cert — accept it for local connections
        let client = Client::builder()
            .danger_accept_invalid_certs(true)
            .build()?;

        let base_url = format!("https://{ip}");
        let stok = Self::login(&client, &base_url, email, password).await?;

        Ok(Self { client, base_url, stok })
    }

    async fn login(client: &Client, base_url: &str, email: &str, password: &str) -> Result<String> {
        // Try three known variants; cameras differ by firmware version
        let candidates: &[(&str, bool)] = &[
            (&format!("{:x}", Sha256::digest(password.as_bytes())), true),  // newer firmware
            (&format!("{:x}", Md5::digest(password.as_bytes())),    true),  // older firmware
            (password,                                               false), // no hashing
        ];

        for (pw, hashed) in candidates {
            let mut params = json!({
                "username": email,
                "password": pw,
            });
            if *hashed {
                params["hashed"] = json!(true);
            }
            let body = json!({ "method": "login", "params": params });
            let resp: Value = client.post(base_url).json(&body).send().await?.json().await?;

            if resp["error_code"].as_i64().unwrap_or(-1) == 0 {
                let stok = resp["result"]["stok"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("No stok in login response"))?
                    .to_string();
                return Ok(stok);
            }
            eprintln!("Auth attempt failed (hashed={hashed}): {}", resp["error_code"]);
        }

        bail!("All login attempts failed — check your Tapo email and password")
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
        Ok(resp)
    }

    pub async fn get_device_info(&self) -> Result<Value> {
        self.request(json!({
            "method": "get",
            "device_info": {"name": ["basic_info"]}
        }))
        .await
    }

    /// Pan/tilt: x_speed and y_speed are -100..100
    pub async fn move_motor(&self, x_speed: i32, y_speed: i32) -> Result<()> {
        self.request(json!({
            "method": "do",
            "motor": {"move": {"x_speed": x_speed, "y_speed": y_speed}}
        }))
        .await?;
        Ok(())
    }

    /// Return to home/calibrated position
    pub async fn calibrate(&self) -> Result<()> {
        self.request(json!({
            "method": "do",
            "motor": {"manual_cali": {}}
        }))
        .await?;
        Ok(())
    }
}
