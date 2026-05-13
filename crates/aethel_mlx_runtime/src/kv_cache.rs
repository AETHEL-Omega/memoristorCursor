use memristor::services::vchip_api::{OmegaVChip, OmegaVChipError};
use std::collections::HashMap;

pub type SessionId = u64;

#[derive(Debug)]
pub struct MemristorKvCacheManager {
    vchip: OmegaVChip,
    /// Last compressed tensor per session (v1 scaffold: lossless passthrough from chip inference).
    sessions: HashMap<SessionId, Vec<f32>>,
    session_budgets: HashMap<SessionId, usize>,
}

#[derive(Debug)]
pub enum CompressKvError {
    Chip(OmegaVChipError),
}

impl std::fmt::Display for CompressKvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressKvError::Chip(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CompressKvError {}

impl From<OmegaVChipError> for CompressKvError {
    fn from(e: OmegaVChipError) -> Self {
        CompressKvError::Chip(e)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum DecompressKvError {
    SessionNotFound(SessionId),
}

impl std::fmt::Display for DecompressKvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecompressKvError::SessionNotFound(id) => {
                write!(f, "no compressed KV for session {id}")
            }
        }
    }
}

impl std::error::Error for DecompressKvError {}

impl MemristorKvCacheManager {
    pub fn new(size: usize) -> Self {
        Self {
            vchip: OmegaVChip::new(size, 0.001, 0.01),
            sessions: HashMap::new(),
            session_budgets: HashMap::new(),
        }
    }

    pub fn compress_kv(
        &mut self,
        session_id: SessionId,
        kv_cache: &[f32],
    ) -> Result<(), CompressKvError> {
        let compressed = self.vchip.infer(kv_cache)?;
        let bytes = compressed.len() * core::mem::size_of::<f32>();
        self.session_budgets.insert(session_id, bytes);
        self.sessions.insert(session_id, compressed);
        Ok(())
    }

    pub fn decompress_kv(&self, session_id: SessionId) -> Result<Vec<f32>, DecompressKvError> {
        self.sessions
            .get(&session_id)
            .cloned()
            .ok_or(DecompressKvError::SessionNotFound(session_id))
    }

    pub fn session_budget(&self, session_id: SessionId) -> Option<usize> {
        self.session_budgets.get(&session_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_updates_session_budget() {
        let mut manager = MemristorKvCacheManager::new(4);
        let payload = vec![0.25_f32; 4];
        manager.compress_kv(7, &payload).expect("compression should pass");
        assert_eq!(manager.session_budget(7), Some(16));
    }

    #[test]
    fn decompress_round_trip_size_match() {
        let mut manager = MemristorKvCacheManager::new(4);
        let payload = vec![0.1_f32, 0.2, 0.3, 0.4];
        manager.compress_kv(1, &payload).unwrap();
        let back = manager.decompress_kv(1).unwrap();
        assert_eq!(back.len(), payload.len());
    }

    #[test]
    fn decompress_unknown_session() {
        let manager = MemristorKvCacheManager::new(4);
        assert_eq!(
            manager.decompress_kv(99).unwrap_err(),
            DecompressKvError::SessionNotFound(99)
        );
    }

    #[test]
    fn compress_wrong_size_is_error() {
        let mut manager = MemristorKvCacheManager::new(4);
        let payload = vec![0.25_f32; 2];
        assert!(manager.compress_kv(1, &payload).is_err());
        assert_eq!(manager.session_budget(1), None);
        assert_eq!(
            manager.decompress_kv(1).unwrap_err(),
            DecompressKvError::SessionNotFound(1)
        );
    }
}
