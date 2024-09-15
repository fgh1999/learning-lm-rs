use reqwest::Client;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct GenerationError {
    err_msg: String,
}

pub struct EndpointClient {
    client: Client,
    base_url: String,
    sess_id: String,
}

impl EndpointClient {
    pub fn new(base_url: &str, sess_id: &str) -> Self {
        EndpointClient {
            client: Client::new(),
            base_url: base_url.to_string(),
            sess_id: sess_id.to_string(),
        }
    }

    pub async fn create_session(&self) -> Result<(), reqwest::Error> {
        let url = format!("{}/createSession", self.base_url);
        self.client
            .post(&url)
            .body(self.sess_id.clone())
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    pub async fn remove_session(&self) -> Result<(), reqwest::Error> {
        let url = format!("{}/removeSession/{}", self.base_url, self.sess_id);
        self.client.delete(&url).send().await?.error_for_status()?;
        Ok(())
    }

    pub async fn generate(&self, token_ids: Vec<u32>) -> Result<Vec<u32>, reqwest::Error> {
        let url = format!("{}/generate/{}", self.base_url, self.sess_id);
        let token_ids_json = serde_json::to_string(&token_ids).unwrap();

        let generated_token_ids = self
            .client
            .put(&url)
            .body(token_ids_json)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        Ok(serde_json::from_slice(&generated_token_ids).unwrap())
    }

    // pub async fn revert(&self, ith_gen: usize) -> Result<Vec<u32>, reqwest::Error> {
    //     let url = format!("{}/revert/{}", self.base_url, self.sess_id);

    //     let reverted_token_ids = self
    //         .client
    //         .put(&url)
    //         .body(ith_gen.to_string())
    //         .send()
    //         .await?
    //         .error_for_status()?
    //         .bytes()
    //         .await?;
    //     Ok(serde_json::from_slice(&reverted_token_ids).unwrap())
    // }
}
