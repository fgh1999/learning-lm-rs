use lm_infer_core::message::{ChatMessage, ChatRole};
use std::io::Write;

pub struct Console<'h> {
    chat_history: &'h [ChatMessage],
}

impl<'h, 'a: 'h> From<&'a [ChatMessage]> for Console<'h> {
    fn from(chat_history: &'a [ChatMessage]) -> Self {
        Self { chat_history }
    }
}
impl<'h, 'a: 'h> From<&'a Vec<ChatMessage>> for Console<'h> {
    fn from(chat_history: &'a Vec<ChatMessage>) -> Self {
        chat_history.as_slice().into()
    }
}

impl<'h> Console<'h> {
    pub fn user_input(&self) -> String {
        let ith = self
            .chat_history
            .iter()
            .filter(|msg| msg.role() == &ChatRole::User)
            .count();
        print!("You({ith})> ");
        std::io::stdout().flush().unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        input.trim().to_string()
    }
    pub fn user_println(msg: &str) {
        println!("You: {}", msg);
    }

    pub fn system_println(msg: &str) {
        println!("System: {}", msg);
    }

    pub fn assistant_println(msg: &str) {
        println!("Assistant: {}", msg);
    }

    pub fn print_all_chat_history(&self) {
        Console::system_println("Chat history:");
        for msg in self.chat_history.iter() {
            match msg.role() {
                ChatRole::System => continue,
                ChatRole::User => Self::user_println(msg.content()),
                ChatRole::Assistant => Self::assistant_println(msg.content()),
            }
        }
        Console::system_println("Chat history ends.");
    }
}
