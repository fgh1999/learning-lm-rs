use serde::Serialize;

#[allow(unused)]
#[derive(Debug)]
enum ChatRole {
    System,
    User,
    Assistant,
}
impl Serialize for ChatRole {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let role_str = match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        serializer.serialize_str(role_str)
    }
}
#[derive(Debug, Serialize)]
pub struct ChatMessage {
    role: ChatRole,
    content: String,
}

impl ChatMessage {
    pub fn from_system(content: &str) -> Self {
        ChatMessage {
            role: ChatRole::System,
            content: content.to_string(),
        }
    }
    pub fn from_user(content: &str) -> Self {
        ChatMessage {
            role: ChatRole::User,
            content: content.to_string(),
        }
    }
    pub fn from_assistant(content: &str) -> Self {
        ChatMessage {
            role: ChatRole::Assistant,
            content: content.to_string(),
        }
    }
}
