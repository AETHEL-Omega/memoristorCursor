//! Use case tests for MemristiveRelation
//!
//! Tests all use cases specified in the feature specification.

use crate::memristive_relation::MemristiveRelation;
use crate::memristor::cell::MemristorCell;

#[cfg(test)]
mod use_case_tests {
    use super::*;

    /// Use Case 1: Erstellen einer MemristiveRelation aus MemristorCell
    #[test]
    fn use_case_1_from_memristor_cell() {
        let cell = MemristorCell::new(0.01, 0.0);
        let relation = MemristiveRelation::from_memristor_cell(&cell);

        // Physikalische Eigenschaften
        assert_eq!(relation.conductance, cell.conductance());
        assert_eq!(relation.activation_energy, cell.voltage_threshold());
        assert_eq!(relation.decay_rate, cell.decay_rate());
        assert_eq!(relation.coherence, cell.normalized_state());

        // Semantische Eigenschaften initialisiert
        assert_eq!(relation.phase, 0.0);
        assert_eq!(relation.frequency, 1.0);
        assert_eq!(relation.semantic_energy, 0.0);
        assert_eq!(relation.confidence, 1.0);

        // Historische Eigenschaften initialisiert
        assert_eq!(relation.historical_depth, 0);
        assert_eq!(relation.activation_count, 0);
    }

    /// Use Case 2: Manuelles Erstellen einer MemristiveRelation
    #[test]
    fn use_case_2_manual_creation() {
        let relation = MemristiveRelation {
            conductance: 5.0,
            activation_energy: 0.5,
            decay_rate: 0.002,
            coherence: 0.8,
            phase: 0.5,
            frequency: 2.0,
            provenance: [1; 32],
            historical_depth: 10,
            activation_count: 100,
            last_activation: 1000,
            resonance_class: crate::resonance_class::ResonanceClass::Temporal,
            semantic_energy: 0.5,
            confidence: 0.9,
            transformability: 0.7,
            plasticity: 0.2,
        };

        assert_eq!(relation.conductance, 5.0);
        assert_eq!(relation.activation_energy, 0.5);
        assert_eq!(relation.decay_rate, 0.002);
        assert_eq!(relation.coherence, 0.8);
    }

    /// Use Case 3: Aktivierung einer MemristiveRelation
    #[test]
    fn use_case_3_activation() {
        let mut relation = MemristiveRelation::new();
        let original_conductance = relation.conductance;
        let original_count = relation.activation_count;

        relation.activate();

        // conductance erhöht
        assert!(relation.conductance > original_conductance);
        // activation_count erhöht
        assert_eq!(relation.activation_count, original_count + 1);
        // last_activation aktualisiert
        assert!(relation.last_activation > 0);
    }

    /// Use Case 4: Zerfall einer MemristiveRelation
    #[test]
    fn use_case_4_decay() {
        let mut relation = MemristiveRelation::new();
        relation.conductance = 10.0;
        relation.coherence = 1.0;

        let original_conductance = relation.conductance;
        let original_coherence = relation.coherence;

        relation.decay(1.0);

        // conductance reduziert
        assert!(relation.conductance < original_conductance);
        // coherence reduziert
        assert!(relation.coherence < original_coherence);
    }

    /// Use Case 5: Resonanz-Berechnung zwischen Signal und MemristiveRelation
    #[test]
    fn use_case_5_resonance_calculation() {
        let mut relation = MemristiveRelation::new();
        relation.conductance = 1.0;
        relation.coherence = 1.0;
        relation.phase = 0.0;
        relation.frequency = 1.0;

        // Signal mit gleicher Phase und Frequenz
        let resonance = relation.compute_resonance(0.0, 1.0);

        // Resonanz sollte hoch sein
        assert!(resonance > 0.9);

        // Signal mit unterschiedlicher Phase
        let resonance_different_phase = relation.compute_resonance(3.14, 1.0);

        // Resonanz sollte niedriger sein
        assert!(resonance_different_phase < resonance);
    }

    /// Use Case 6: Serialisierung und Deserialisierung
    #[test]
    fn use_case_6_serialization() {
        let original = MemristiveRelation::new();
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: MemristiveRelation = serde_json::from_str(&serialized).unwrap();

        assert_eq!(original.conductance, deserialized.conductance);
        assert_eq!(original.activation_energy, deserialized.activation_energy);
        assert_eq!(original.decay_rate, deserialized.decay_rate);
    }
}
