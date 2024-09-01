mod chat_client;
mod message;

use std::io::Write;

use clap::Parser;
use log::info;
use message::ChatMessage;

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

    // until user types "exit"
    loop {
        print!("You: ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();
        if input.to_lowercase() == "exit" {
            break;
        }

        chat_history.push(ChatMessage::from_user(input));
        let prompt = prompt_from_history(chat_history.as_slice(), true);
        let token_ids = {
            let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
            binding.get_ids().to_owned()
        };
        let output_ids = client.generate(token_ids).await?;
        let answer = tokenizer.decode(&output_ids, true).unwrap();
        chat_history.push(ChatMessage::from_assistant(&answer));
        println!("Assistant: {}", answer);
    }

    Ok(())
}
