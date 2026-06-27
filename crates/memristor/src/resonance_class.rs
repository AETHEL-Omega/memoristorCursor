//! Resonance classes for memristive relations
//!
//! Defines different types of resonance patterns that can occur
//! between signals and relations in the MEMORISSTORE system.

use serde::{Deserialize, Serialize};

/// Resonance classes for different types of semantic and temporal patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResonanceClass {
    // Semantic resonance patterns
    Semantic,
    Conceptual,
    Associative,

    // Temporal resonance patterns
    Temporal,
    Sequential,
    Causal,

    // Spatial resonance patterns
    Spatial,
    Hierarchical,
    Network,

    // Emotional resonance patterns
    Emotional,
    Affective,
    Motivational,
}

impl ResonanceClass {
    /// Get the default resonance class
    pub fn default() -> Self {
        ResonanceClass::Semantic
    }

    /// Check if this is a semantic resonance class
    pub fn is_semantic(&self) -> bool {
        matches!(
            self,
            ResonanceClass::Semantic | ResonanceClass::Conceptual | ResonanceClass::Associative
        )
    }

    /// Check if this is a temporal resonance class
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            ResonanceClass::Temporal | ResonanceClass::Sequential | ResonanceClass::Causal
        )
    }

    /// Check if this is a spatial resonance class
    pub fn is_spatial(&self) -> bool {
        matches!(
            self,
            ResonanceClass::Spatial | ResonanceClass::Hierarchical | ResonanceClass::Network
        )
    }

    /// Check if this is an emotional resonance class
    pub fn is_emotional(&self) -> bool {
        matches!(
            self,
            ResonanceClass::Emotional | ResonanceClass::Affective | ResonanceClass::Motivational
        )
    }
}

impl Default for ResonanceClass {
    fn default() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let rc = ResonanceClass::default();
        assert_eq!(rc, ResonanceClass::Semantic);
    }

    #[test]
    fn test_is_semantic() {
        assert!(ResonanceClass::Semantic.is_semantic());
        assert!(ResonanceClass::Conceptual.is_semantic());
        assert!(ResonanceClass::Associative.is_semantic());
        assert!(!ResonanceClass::Temporal.is_semantic());
    }

    #[test]
    fn test_is_temporal() {
        assert!(ResonanceClass::Temporal.is_temporal());
        assert!(ResonanceClass::Sequential.is_temporal());
        assert!(ResonanceClass::Causal.is_temporal());
        assert!(!ResonanceClass::Semantic.is_temporal());
    }

    #[test]
    fn test_is_spatial() {
        assert!(ResonanceClass::Spatial.is_spatial());
        assert!(ResonanceClass::Hierarchical.is_spatial());
        assert!(ResonanceClass::Network.is_spatial());
        assert!(!ResonanceClass::Semantic.is_spatial());
    }

    #[test]
    fn test_is_emotional() {
        assert!(ResonanceClass::Emotional.is_emotional());
        assert!(ResonanceClass::Affective.is_emotional());
        assert!(ResonanceClass::Motivational.is_emotional());
        assert!(!ResonanceClass::Semantic.is_emotional());
    }
}
