mod message;
use log::info;
use message::ChatMessage;
mod command;
use clap::Parser;

use lm_infer::{
    model::Llama,
    service::{ChatService, TemplateName},
    session::{Generation, LmSession},
};
use minijinja::{context as jcontext, Environment as JinjaEnv};
use std::sync::Arc;
use tokenizers::Tokenizer;

type ModelParamType = f32;
const DEFAULT_MODEL_DIR: &str = "models/chat";

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The path to the model directory
    #[arg(short, long, default_value = DEFAULT_MODEL_DIR)]
    model_dir: String,
}

fn main() {
    let args = Args::parse();
    let model_dir = std::env::current_dir().unwrap().join(args.model_dir);
    env_logger::init();

    let llama = Arc::new(Llama::<ModelParamType>::from_safetensors(&model_dir));
    info!("Model loaded from {:?}", model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    info!("Tokenizer loaded from {:?}", model_dir);

    let eos_token = {
        let rdr = std::fs::File::open(model_dir.join("generation_config.json")).unwrap();
        let generation_config: serde_json::Value = serde_json::from_reader(rdr).unwrap();
        let token_id = generation_config["eos_token_id"].as_u64().unwrap();
        tokenizer.decode(&[token_id as u32], false).unwrap()
    };

    let chat_template = {
        let rdr = std::fs::File::open(model_dir.join("tokenizer_config.json")).unwrap();
        let tokenizer_config: serde_json::Value = serde_json::from_reader(rdr).unwrap();
        tokenizer_config["chat_template"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let mut jenv = JinjaEnv::new();
    jenv.add_template(TemplateName::Chat.into(), &chat_template)
        .unwrap();
    let chat_svc: ChatService<LmSession<ModelParamType, u32, Llama<ModelParamType>>> =
        ChatService::new(tokenizer, jenv);
    info!("Chat service initialized");

    let user_id = "user0".to_string();
    chat_svc.create_session(user_id.clone(), LmSession::new(llama));
    let mut chat_history: Vec<ChatMessage> = vec![
        ChatMessage::from_system("You are an assistant."),
        ChatMessage::from_user("Is apple red?"),
    ];

    let chat_template = chat_svc
        .jenv()
        .get_template(TemplateName::Chat.into())
        .unwrap();
    let prompt = {
        let prompt = chat_template
            .render(jcontext!(add_generation_prompt => true, messages => chat_history, eos_token))
            .unwrap();
        prompt
            .chars()
            .fold((String::new(), ' '), |(mut s, last), c| {
                if c.is_whitespace() && last.is_whitespace() {
                    (s, c)
                } else {
                    s.push(c);
                    (s, c)
                }
            })
            .0
    };
    chat_svc.with_session_mut(&user_id, |sess| {
        print!("PROMPT: {}", prompt);
        let input_ids = {
            let binding = chat_svc.tokenizer().encode(prompt.as_str(), true).unwrap();
            binding.get_ids().to_owned()
        };
        let output_ids = sess.generate(&input_ids, 128, 0.55, 35, 0.65, Some(1.176));
        let answer = chat_svc.tokenizer().decode(&output_ids, true).unwrap();
        chat_history.push(ChatMessage::from_assistant(&answer));
        println!("ANSWER: {}", answer);
    });

    #[cfg(feature = "perf")]
    chat_svc.with_session(&user_id, |sess| {
        sess.print_perf_info();
    });
}
