use serde::Deserialize;

#[derive(Debug)]
pub enum CliCommand {
    Identity,
    Chat,
    History,
    Help,
    Exit,
}

#[derive(Debug, Deserialize)]
pub enum CliChatCommand {
    Exit,
}
