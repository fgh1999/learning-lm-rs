use dashmap::DashMap;
use getset::Getters;
use log::info;
use minijinja::Environment;
use tokenizers::Tokenizer;

use crate::session::Generation;

pub type Identity = String;

#[derive(Getters)]
pub struct ChatService<'source, G: Generation> {
    sessions: DashMap<Identity, G>,
    #[getset(get = "pub")]
    tokenizer: Tokenizer,
    #[getset(get = "pub")]
    jenv: Environment<'source>,
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

impl<'source, G: Generation> ChatService<'source, G> {
    pub fn new(tokenizer: Tokenizer, jenv: Environment<'source>) -> Self {
        Self {
            sessions: DashMap::new(),
            tokenizer,
            jenv,
        }
    }

    pub fn create_session(&self, identity: Identity, generation: G) {
        let identity_clone = identity.clone();
        self.sessions.insert(identity, generation);
        info!("Session created: {}", identity_clone);
    }

    pub fn with_session<T>(&self, identity: &Identity, f: impl FnOnce(&G) -> T) -> Option<T> {
        self.sessions.get(identity).map(|sess| f(sess.value()))
    }

    pub fn with_session_mut<T>(
        &self,
        identity: &Identity,
        f: impl FnOnce(&mut G) -> T,
    ) -> Option<T> {
        self.sessions
            .get_mut(identity)
            .map(|mut session| f(session.value_mut()))
    }

    pub fn remove_session(&self, identity: &Identity) -> Option<G> {
        let removed = self.sessions.remove(identity).map(|(_, sess)| sess);
        info!("Session removed: {}", identity);
        removed
    }
}
