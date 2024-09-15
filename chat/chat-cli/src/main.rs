mod chat_client;
mod console;
use console::Console;

use clap::Parser;
use lm_infer_core::message::{ChatMessage, ChatRole};
use log::info;

use anyhow::Result;
use minijinja::{context as jcontext, Environment as JinjaEnv};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// The path to the model directory
    #[arg(short, long)]
    model_dir: String,

    #[arg(short, long, default_value = "test_session_id")]
    id: String,

    #[arg(short, long, default_value = "http://localhost:8000")]
    service_url: String,
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

#[tokio::main]
async fn main() -> Result<()> {
    let Args {
        model_dir,
        id: session_id,
        service_url,
    } = Args::parse();
    let model_dir = std::env::current_dir().unwrap().join(model_dir);
    env_logger::init();

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

    let mut chat_history: Vec<ChatMessage> =
        vec![ChatMessage::from_system("You are an assistant.")];

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

    let client = chat_client::EndpointClient::new(&service_url, &session_id);
    client.create_session().await?;
    info!("Session created");

    loop {
        let input = Console::from(&chat_history).user_input();
        // command mode
        if input.trim().to_lowercase().starts_with(':') {
            let input = input.trim().to_lowercase();
            let input = input[1..].trim();
            match input {
                // loops until user types "exit" command
                "exit" => {
                    client.remove_session().await?;
                    return Ok(());
                }
                "history" => Console::from(&chat_history).print_all_chat_history(),
                input if input.starts_with("revert_to") => {
                    let Ok(target_ith) = input["revert_to".len()..].trim().parse::<usize>() else {
                        Console::system_println("Invalid ith: should be of u64");
                        continue;
                    };
                    let cur_conversation_num = chat_history
                        .iter()
                        .filter(|msg| msg.role() == &ChatRole::Assistant)
                        .count();
                    if target_ith >= cur_conversation_num {
                        Console::system_println("Invalid ith: out of range");
                        continue;
                    }

                    // Revertions are only taken offline
                    // let _ = client.revert(target_ith).await?;
                    chat_history.truncate(target_ith * 2 + 1);
                    Console::system_println(&format!(
                        "Reverted to the state before {}th generation",
                        target_ith
                    ));
                }
                // TODO: `help` cmd
                _ => {
                    Console::system_println(&format!("Unknown command: {}", input));
                }
            }
            continue;
        }

        // chat mode
        chat_history.push(ChatMessage::from_user(&input));
        let token_ids = {
            let prompt = prompt_from_history(chat_history.as_slice(), true);
            let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
            binding.get_ids().to_owned()
        };
        let answer = {
            let output_ids = client.generate(token_ids).await?;
            tokenizer.decode(&output_ids, true).unwrap()
        };
        chat_history.push(ChatMessage::from_assistant(&answer));
        Console::assistant_println(&answer);
    }
}
