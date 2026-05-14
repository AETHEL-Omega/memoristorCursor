//! Use case tests for ResonanceClass
//!
//! Tests all use cases specified in the feature specification.

use crate::resonance_class::ResonanceClass;

#[cfg(test)]
mod use_case_tests {
    use super::*;

    /// Use Case 1: Default ResonanceClass
    /// Beschreibung: Eine ResonanceClass wird mit Default-Werten erstellt.
    #[test]
    fn use_case_1_default_resonance_class() {
        let rc = ResonanceClass::default();
        assert_eq!(rc, ResonanceClass::Semantic);
    }

    /// Use Case 2: Semantic Resonance Class
    /// Beschreibung: Eine semantische ResonanceClass wird erstellt und geprüft.
    #[test]
    fn use_case_2_semantic_resonance_class() {
        let rc = ResonanceClass::Semantic;
        assert!(rc.is_semantic());
        assert!(!rc.is_temporal());
        assert!(!rc.is_spatial());
        assert!(!rc.is_emotional());
    }

    /// Use Case 3: Temporal Resonance Class
    /// Beschreibung: Eine temporale ResonanceClass wird erstellt und geprüft.
    #[test]
    fn use_case_3_temporal_resonance_class() {
        let rc = ResonanceClass::Temporal;
        assert!(!rc.is_semantic());
        assert!(rc.is_temporal());
        assert!(!rc.is_spatial());
        assert!(!rc.is_emotional());
    }

    /// Use Case 4: Spatial Resonance Class
    /// Beschreibung: Eine räumliche ResonanceClass wird erstellt und geprüft.
    #[test]
    fn use_case_4_spatial_resonance_class() {
        let rc = ResonanceClass::Spatial;
        assert!(!rc.is_semantic());
        assert!(!rc.is_temporal());
        assert!(rc.is_spatial());
        assert!(!rc.is_emotional());
    }

    /// Use Case 5: Emotional Resonance Class
    /// Beschreibung: Eine emotionale ResonanceClass wird erstellt und geprüft.
    #[test]
    fn use_case_5_emotional_resonance_class() {
        let rc = ResonanceClass::Emotional;
        assert!(!rc.is_semantic());
        assert!(!rc.is_temporal());
        assert!(!rc.is_spatial());
        assert!(rc.is_emotional());
    }

    /// Use Case 6: All Semantic Variants
    /// Beschreibung: Alle semantischen Varianten werden geprüft.
    #[test]
    fn use_case_6_all_semantic_variants() {
        assert!(ResonanceClass::Semantic.is_semantic());
        assert!(ResonanceClass::Conceptual.is_semantic());
        assert!(ResonanceClass::Associative.is_semantic());
    }

    /// Use Case 7: All Temporal Variants
    /// Beschreibung: Alle temporalen Varianten werden geprüft.
    #[test]
    fn use_case_7_all_temporal_variants() {
        assert!(ResonanceClass::Temporal.is_temporal());
        assert!(ResonanceClass::Sequential.is_temporal());
        assert!(ResonanceClass::Causal.is_temporal());
    }

    /// Use Case 8: All Spatial Variants
    /// Beschreibung: Alle räumlichen Varianten werden geprüft.
    #[test]
    fn use_case_8_all_spatial_variants() {
        assert!(ResonanceClass::Spatial.is_spatial());
        assert!(ResonanceClass::Hierarchical.is_spatial());
        assert!(ResonanceClass::Network.is_spatial());
    }

    /// Use Case 9: All Emotional Variants
    /// Beschreibung: Alle emotionalen Varianten werden geprüft.
    #[test]
    fn use_case_9_all_emotional_variants() {
        assert!(ResonanceClass::Emotional.is_emotional());
        assert!(ResonanceClass::Affective.is_emotional());
        assert!(ResonanceClass::Motivational.is_emotional());
    }

    /// Use Case 10: Cross-Category Exclusivity
    /// Beschreibung: ResonanceClass ist nur in einer Kategorie.
    #[test]
    fn use_case_10_cross_category_exclusivity() {
        let rc = ResonanceClass::Semantic;
        assert!(rc.is_semantic());
        assert!(!rc.is_temporal());
        assert!(!rc.is_spatial());
        assert!(!rc.is_emotional());
    }
}
