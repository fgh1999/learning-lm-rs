use clap::Parser;
use log::info;

use lm_infer_core::{
    model::{Llama, ModelResource},
    service::ChatService,
    session::{LmSession, TokenGeneration},
};
use std::sync::Arc;
#[macro_use]
extern crate rocket;
use rocket::{
    http::Status,
    response::{content, status},
    State,
};

type ModelParamType = f32;

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The path to the model directory
    #[arg(short, long)]
    model_dir: String,

    #[arg(long, default_value = "128")]
    max_seq_len: usize,

    #[arg(short = 'p', long, default_value = "0.55")]
    top_p: f32,
    #[arg(short = 'k', long, default_value = "35")]
    top_k: u32,
    #[arg(short, long, default_value = "0.65")]
    temperature: f32,
    #[arg(short, long)]
    repetition_penalty: Option<f32>,
}

#[post("/createSession", data = "<sess_id>")]
fn create_session(
    sess_id: &str,
    chat_service: &State<ChatService<LmSession<ModelParamType, u32, Llama<ModelParamType>>>>,
    llama: &State<Arc<Llama<ModelParamType>>>,
) -> (Status, ()) {
    if chat_service.with_session(sess_id, |_| {}).is_some() {
        return (Status::Conflict, ());
    }
    chat_service.create_session(sess_id.to_owned(), LmSession::new(llama.inner().clone()));
    info!("Session created: {}", sess_id);
    (Status::Created, ())
}

#[delete("/removeSession/<sess_id>")]
fn remove_session(
    sess_id: &str,
    chat_service: &State<ChatService<LmSession<ModelParamType, u32, Llama<ModelParamType>>>>,
) -> (Status, ()) {
    if chat_service.remove_session(sess_id).is_some() {
        info!("Session removed: {}", sess_id);
        (Status::Ok, ())
    } else {
        (Status::NotFound, ())
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct GenerationError {
    err_msg: String,
}

#[put("/generate/<sess_id>", data = "<token_ids>")]
fn generate(
    sess_id: &str,
    token_ids: &str,
    chat_service: &State<ChatService<LmSession<ModelParamType, u32, Llama<ModelParamType>>>>,
    args: &State<Args>,
) -> status::Custom<content::RawJson<String>> {
    let Ok(token_ids) = serde_json::from_str::<Vec<u32>>(token_ids) else {
        return status::Custom(
            Status::BadRequest,
            content::RawJson(
                serde_json::to_string(&GenerationError {
                    err_msg: "Invalid token_ids".to_owned(),
                })
                .unwrap(),
            ),
        );
    };
    let generated_token_ids = chat_service.with_session_mut(sess_id, |sess| {
        let token_ids = sess.generate(
            &token_ids,
            args.max_seq_len,
            args.top_p,
            args.top_k,
            args.temperature,
            args.repetition_penalty,
        );
        info!("Generated tokens for session {}", sess_id);
        token_ids
    });
    match generated_token_ids {
        Some(generated_token_ids) => status::Custom(
            Status::Ok,
            content::RawJson(serde_json::to_string(&generated_token_ids).unwrap()),
        ),
        None => status::Custom(
            Status::NotFound,
            content::RawJson(
                serde_json::to_string(&GenerationError {
                    err_msg: "Session not found".to_owned(),
                })
                .unwrap(),
            ),
        ),
    }
}

#[launch]
fn rocket() -> _ {
    env_logger::init();

    rocket::build()
        .manage(Args::parse())
        .manage(ChatService::<
            LmSession<ModelParamType, u32, Llama<ModelParamType>>,
        >::default())
        .manage({
            let args = Args::parse();
            let model_dir = std::path::Path::new(&args.model_dir);
            assert!(
                model_dir.exists(),
                "Model directory does not exist: {:?}",
                model_dir
            );

            let llama = {
                let resources = ModelResource {
                    config: Some(std::fs::read(model_dir.join("config.json")).unwrap()),
                    model_data: Some(std::fs::read(model_dir.join("model.safetensors")).unwrap()),
                    ..Default::default()
                };
                Arc::new(Llama::<ModelParamType>::from_safetensors(&resources))
            };
            info!("Model loaded from {:?}", model_dir);
            llama
        })
        .mount("/", routes![create_session, remove_session, generate])
}
