//! Memristive relation for MEMORISSTORE
//!
//! Defines "remembering relationships" with physical, semantic, and historical properties.

use crate::memristor::cell::MemristorCell;
use crate::resonance_class::ResonanceClass;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Memristive relation with physical, semantic, and historical properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemristiveRelation {
    // Physical properties (from MemristorCell)
    pub conductance: f32,
    pub activation_energy: f32,
    pub decay_rate: f32,

    // Coherence properties
    pub coherence: f32,
    pub phase: f32,
    pub frequency: f32,

    // Historical properties
    pub provenance: [u8; 32],
    pub historical_depth: u64,
    pub activation_count: u64,
    pub last_activation: u64,

    // Semantic properties
    pub resonance_class: ResonanceClass,
    pub semantic_energy: f32,
    pub confidence: f32,

    // Transformation properties
    pub transformability: f32,
    pub plasticity: f32,
}

impl MemristiveRelation {
    /// Create a new memristive relation with default values
    pub fn new() -> Self {
        Self {
            conductance: 1.0,
            activation_energy: 0.0,
            decay_rate: 0.001,
            coherence: 1.0,
            phase: 0.0,
            frequency: 1.0,
            provenance: [0; 32],
            historical_depth: 0,
            activation_count: 0,
            last_activation: 0,
            resonance_class: ResonanceClass::default(),
            semantic_energy: 0.0,
            confidence: 1.0,
            transformability: 0.5,
            plasticity: 0.1,
        }
    }

    /// Create a memristive relation from a memristor cell
    pub fn from_memristor_cell(cell: &MemristorCell) -> Self {
        Self {
            conductance: cell.conductance(),
            activation_energy: cell.voltage_threshold(),
            decay_rate: cell.decay_rate(),
            coherence: cell.normalized_state(),
            phase: 0.0,
            frequency: 1.0,
            provenance: [0; 32],
            historical_depth: 0,
            activation_count: 0,
            last_activation: 0,
            resonance_class: ResonanceClass::default(),
            semantic_energy: 0.0,
            confidence: 1.0,
            transformability: 0.5,
            plasticity: 0.1,
        }
    }

    /// Activate the relation (increases conductance, updates counters)
    pub fn activate(&mut self) {
        self.conductance *= 1.0 + self.plasticity;
        self.activation_count += 1;
        self.last_activation = current_timestamp();
        self.phase += 0.1 * self.semantic_energy;
    }

    /// Decay the relation over time
    pub fn decay(&mut self, dt: f32) {
        let decay_factor = (-self.decay_rate * dt).exp();
        self.conductance *= decay_factor;
        self.coherence *= decay_factor;
    }

    /// Compute resonance with a signal
    pub fn compute_resonance(&self, signal_phase: f32, signal_frequency: f32) -> f32 {
        let phase_alignment = (signal_phase - self.phase).cos();
        let frequency_match = 1.0 - (signal_frequency - self.frequency).abs();
        self.conductance * self.coherence * phase_alignment * frequency_match
    }
}

impl Default for MemristiveRelation {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in seconds since UNIX epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_default_relation() {
        let relation = MemristiveRelation::new();
        assert_eq!(relation.conductance, 1.0);
        assert_eq!(relation.activation_energy, 0.0);
        assert_eq!(relation.decay_rate, 0.001);
        assert_eq!(relation.coherence, 1.0);
        assert_eq!(relation.phase, 0.0);
        assert_eq!(relation.frequency, 1.0);
        assert_eq!(relation.historical_depth, 0);
        assert_eq!(relation.activation_count, 0);
        assert_eq!(relation.semantic_energy, 0.0);
        assert_eq!(relation.confidence, 1.0);
    }

    #[test]
    fn test_from_memristor_cell() {
        let cell = MemristorCell::new(0.01, 0.0);
        let relation = MemristiveRelation::from_memristor_cell(&cell);
        
        assert_eq!(relation.conductance, cell.conductance());
        assert_eq!(relation.activation_energy, cell.voltage_threshold());
        assert_eq!(relation.decay_rate, cell.decay_rate());
        assert_eq!(relation.coherence, cell.normalized_state());
    }

    #[test]
    fn test_activate_increases_conductance() {
        let mut relation = MemristiveRelation::new();
        let original_conductance = relation.conductance;
        
        relation.activate();
        
        assert!(relation.conductance > original_conductance);
        assert_eq!(relation.activation_count, 1);
    }

    #[test]
    fn test_decay_reduces_conductance() {
        let mut relation = MemristiveRelation::new();
        relation.conductance = 10.0;
        let original_conductance = relation.conductance;
        
        relation.decay(1.0);
        
        assert!(relation.conductance < original_conductance);
    }

    #[test]
    fn test_resonance_in_range() {
        let relation = MemristiveRelation::new();
        let resonance = relation.compute_resonance(0.0, 1.0);
        
        assert!(resonance >= 0.0);
        assert!(resonance <= 1.0);
    }
}
