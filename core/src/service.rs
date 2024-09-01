use dashmap::DashMap;
use getset::Getters;
use log::info;

use crate::session::TokenGeneration;

pub type Identity = String;

#[derive(Getters)]
pub struct ChatService<G: TokenGeneration> {
    sessions: DashMap<Identity, G>,
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

impl<'source, G: TokenGeneration> ChatService<G> {
    pub fn new() -> Self {
        Self {
            sessions: DashMap::new(),
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
