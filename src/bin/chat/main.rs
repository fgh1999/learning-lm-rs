mod message;
use log::info;
use message::ChatMessage;
mod command;

use lm_infer::{
    model::Llama,
    service::{ChatService, TemplateName},
    session::{Generation, LmSession},
};
use minijinja::{context as jcontext, Environment as JinJaEnv};
use std::{path::PathBuf, sync::Arc};
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    env_logger::init();

    let llama = Arc::new(Llama::<f32>::from_safetensors(&model_dir));
    info!("Model loaded from {:?}", model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    info!("Tokenizer loaded from {:?}", model_dir);

    let chat_template = {
        let rdr = std::fs::File::open(model_dir.join("tokenizer_config.json")).unwrap();
        let tokenizer_config: serde_json::Value = serde_json::from_reader(rdr).unwrap();
        tokenizer_config["chat_template"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let mut jenv = JinJaEnv::new();
    jenv.add_template(TemplateName::Chat.into(), &chat_template)
        .unwrap();
    let chat_svc: ChatService<LmSession<f32, u32, Llama<f32>>> = ChatService::new(tokenizer, jenv);
    info!("Chat service initialized");

    let user_id = "user0".to_string();
    chat_svc.create_session(user_id.clone(), LmSession::new(llama));
    // todo!("merge chat_history into session");
    let mut chat_history: Vec<ChatMessage> = Vec::new();
    chat_history.push(ChatMessage::from_system("You are an assistant."));
    chat_history.push(ChatMessage::from_user("Is apple red?"));

    let chat_template = chat_svc
        .jenv()
        .get_template(TemplateName::Chat.into())
        .unwrap();
    let prompt = chat_template
        .render(jcontext!(add_generation_prompt => true, messages => chat_history))
        .unwrap();
    chat_svc.with_session_mut(&user_id, |sess| {
        print!("PROMPT: {}", prompt);
        let input_ids = {
            let binding = chat_svc.tokenizer().encode(prompt.as_str(), true).unwrap();
            binding.get_ids().to_owned()
        };
        let output_ids = sess.generate(&input_ids, 500, 0.55, 35, 0.65, Some(1.176));
        println!(
            "ANSWER: {}",
            chat_svc.tokenizer().decode(&output_ids, true).unwrap()
        );
    });

    #[cfg(feature = "perf")]
    chat_svc.with_session(&user_id, |sess| {
        sess.print_perf_info();
    });
}
