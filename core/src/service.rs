use dashmap::DashMap;
use getset::Getters;
use log::info;

use crate::session::TokenGeneration;

#[derive(Getters)]
pub struct ChatService<G: TokenGeneration> {
    sessions: DashMap<String, G>,
}

impl<G: TokenGeneration> Default for ChatService<G> {
    fn default() -> Self {
        ChatService {
            sessions: DashMap::new(),
        }
    }
}

impl<G: TokenGeneration> ChatService<G> {
    pub fn create_session(&self, identity: String, generation: G) {
        let identity_clone = identity.clone();
        self.sessions.insert(identity, generation);
        info!("Session created: {}", identity_clone);
    }

    pub fn with_session<T>(&self, identity: &str, f: impl FnOnce(&G) -> T) -> Option<T> {
        self.sessions.get(identity).map(|sess| f(sess.value()))
    }

    pub fn with_session_mut<T>(&self, identity: &str, f: impl FnOnce(&mut G) -> T) -> Option<T> {
        self.sessions
            .get_mut(identity)
            .map(|mut session| f(session.value_mut()))
    }

    pub fn remove_session(&self, identity: &str) -> Option<G> {
        let removed = self.sessions.remove(identity).map(|(_, sess)| sess);
        info!("Session removed: {}", identity);
        removed
    }
}
