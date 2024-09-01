mod message;
use clap::Parser;
use log::info;
use message::ChatMessage;

use lm_infer_core::{
    model::Llama,
    service::ChatService,
    session::{LmSession, TokenGeneration},
};
use minijinja::{context as jcontext, Environment as JinjaEnv};
use std::sync::Arc;
use tokenizers::Tokenizer;

type ModelParamType = f32;

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The path to the model directory
    #[arg(short, long)]
    model_dir: String,
}

#[derive(Debug)]
pub enum TemplateName {
    Chat,
}
impl From<TemplateName> for &str {
    fn from(val: TemplateName) -> Self {
        match val {
            TemplateName::Chat => "chat",
        }
    }
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
        ChatService::default();
    info!("Chat service initialized");

    let user_id = "user0".to_string();
    chat_svc.create_session(user_id.clone(), LmSession::new(llama));
    let mut chat_history: Vec<ChatMessage> = vec![
        ChatMessage::from_system("You are an assistant."),
        ChatMessage::from_user("Is apple red?"),
    ];

    let chat_template = jenv.get_template(TemplateName::Chat.into()).unwrap();

    fn compact_prompt(prompt: &str) -> String {
        prompt
            .chars()
            .fold((String::new(), ' '), |(mut s, last), c| {
                if c.is_whitespace() && c == last {
                    (s, c)
                } else {
                    s.push(c);
                    (s, c)
                }
            })
            .0
    }
    let prompt_from_history =
        |chat_history: &[ChatMessage], add_generation_prompt: bool| -> String {
            let prompt = chat_template
                .render(jcontext!(add_generation_prompt, messages => chat_history, eos_token))
                .unwrap();
            compact_prompt(&prompt)
        };

    let prompt = prompt_from_history(chat_history.as_slice(), true);
    let answer = chat_svc
        .with_session_mut(&user_id, |sess| {
            print!("PROMPT: {}", prompt);
            let input_ids = {
                let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
                binding.get_ids().to_owned()
            };
            let output_ids = sess.generate(&input_ids, 128, 0.55, 35, 0.65, Some(1.176));
            tokenizer.decode(&output_ids, true).unwrap()
        })
        .expect("Session not found");
    chat_history.push(ChatMessage::from_assistant(&answer));
    println!("ANSWER: {}\n", answer);

    chat_history.push(ChatMessage::from_user("How many legs does a rabbit have?"));
    let prompt = prompt_from_history(chat_history.as_slice(), true);
    let answer = chat_svc
        .with_session_mut(&user_id, |sess| {
            print!("PROMPT: {}", prompt);
            let input_ids = {
                let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
                binding.get_ids().to_owned()
            };
            let output_ids = sess.generate(&input_ids, 128, 0.55, 35, 0.65, Some(1.176));
            tokenizer.decode(&output_ids, true).unwrap()
        })
        .expect("Session not found");
    chat_history.push(ChatMessage::from_assistant(&answer));
    println!("ANSWER: {}\n", answer);

    // revert the previous question & answer
    chat_history.pop(); // pop the answer
    chat_history.pop(); // pop the question
    let prompt_after_revert = prompt_from_history(chat_history.as_slice(), false);
    println!("correct PROMPT after [revertion]: {}", prompt_after_revert);
    // let input_ids = {
    //     let binding = chat_svc.tokenizer().encode(prompt_after_revert.as_str(), true).unwrap();
    //     binding.get_ids().to_owned()
    // };
    chat_svc
        .with_session_mut(&user_id, |sess| {
            let token_ids = sess.revert_to_before(1).collect::<Vec<_>>();
            let tokens = tokenizer.decode(&token_ids, false).unwrap();
            println!("true PROMPT after [revertion]: {}", tokens);
        })
        .expect("Session not found");
    println!("[Reverted] to the previous question.\n");

    // generate again with another question
    chat_history.push(ChatMessage::from_user("What is the color of the sky?"));
    let prompt = prompt_from_history(chat_history.as_slice(), true);
    let answer = chat_svc
        .with_session_mut(&user_id, |sess| {
            print!("PROMPT: {}", prompt);
            let input_ids = {
                let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
                binding.get_ids().to_owned()
            };
            let output_ids = sess.generate(&input_ids, 128, 0.55, 35, 0.65, Some(1.176));
            tokenizer.decode(&output_ids, true).unwrap()
        })
        .expect("Session not found");
    chat_history.push(ChatMessage::from_assistant(&answer));
    println!("ANSWER: {}\n", answer);

    #[cfg(feature = "perf")]
    chat_svc.with_session(&user_id, |sess| {
        sess.print_perf_info();
    });
}
