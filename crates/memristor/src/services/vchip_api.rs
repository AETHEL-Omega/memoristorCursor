use crate::memristor::crossbar::{Crossbar, CrossbarError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum OmegaVChipError {
    #[error("invalid input size: expected {expected}, got {got}")]
    InvalidInputSize { expected: usize, got: usize },

    #[error("crossbar error: {0}")]
    Crossbar(#[from] CrossbarError),
}

pub type OmegaVChipResult<T> = Result<T, OmegaVChipError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceMode {
    Digital,
    Analog,
    Hybrid,
}

#[derive(Debug)]
pub struct OmegaVChip {
    crossbar: Crossbar,
    size: usize,
}

impl OmegaVChip {
    pub fn new(size: usize, drift_factor: f32, noise_level: f32) -> Self {
        Self {
            crossbar: Crossbar::new(size, drift_factor, noise_level),
            size,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn crossbar_mut(&mut self) -> &mut Crossbar {
        &mut self.crossbar
    }

    /// Digital-ish path: same dimensions as analog, deterministic given fixed resistances.
    pub fn infer_digital(&self, input: &[f32]) -> OmegaVChipResult<Vec<f32>> {
        self.infer(input)
    }

    pub fn infer_analog(&self, input: &[f32]) -> OmegaVChipResult<Vec<f32>> {
        self.crossbar.forward(input).map_err(OmegaVChipError::from)
    }

    /// Fixed blend documented for Phase 3 hybrid routing.
    pub fn infer_hybrid(&self, input: &[f32], digital_weight: f32) -> OmegaVChipResult<Vec<f32>> {
        let dw = digital_weight.clamp(0.0, 1.0);
        let aw = 1.0 - dw;
        let digital = self.infer_digital(input)?;
        let analog = self.infer_analog(input)?;
        Ok(digital
            .into_iter()
            .zip(analog)
            .map(|(d, a)| dw * d + aw * a)
            .collect())
    }

    pub fn infer(&self, input: &[f32]) -> OmegaVChipResult<Vec<f32>> {
        if input.len() != self.size {
            return Err(OmegaVChipError::InvalidInputSize {
                expected: self.size,
                got: input.len(),
            });
        }
        self.crossbar.forward(input).map_err(OmegaVChipError::from)
    }

    /// Sequentially apply the same crossbar `depth` times (Windsurf Cascade prototype).
    /// `depth == 0` returns a copy of `input`.
    pub fn infer_cascade(&self, input: &[f32], depth: usize) -> OmegaVChipResult<Vec<f32>> {
        if input.len() != self.size {
            return Err(OmegaVChipError::InvalidInputSize {
                expected: self.size,
                got: input.len(),
            });
        }
        if depth == 0 {
            return Ok(input.to_vec());
        }
        let mut x = input.to_vec();
        for _ in 0..depth {
            x = self.infer(&x)?;
        }
        Ok(x)
    }
}

/// Multi-pass inference through one virtual crossbar (fractal decomposition hook).
pub trait WindsurfCascade {
    fn infer_cascade(&self, input: &[f32], depth: usize) -> OmegaVChipResult<Vec<f32>>;
}

impl WindsurfCascade for OmegaVChip {
    fn infer_cascade(&self, input: &[f32], depth: usize) -> OmegaVChipResult<Vec<f32>> {
        OmegaVChip::infer_cascade(self, input, depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_rejects_wrong_length() {
        let chip = OmegaVChip::new(4, 0.001, 0.0);
        assert_eq!(
            chip.infer(&[1.0, 2.0]).unwrap_err(),
            OmegaVChipError::InvalidInputSize {
                expected: 4,
                got: 2,
            }
        );
    }

    #[test]
    fn infer_hybrid_matches_endpoints() {
        let mut chip = OmegaVChip::new(2, 0.001, 0.0);
        chip.crossbar_mut().set_resistance(0, 0, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(0, 1, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(1, 0, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(1, 1, 1.0).unwrap();
        let input = [1.0_f32, 0.0];
        let d = chip.infer_digital(&input).unwrap();
        let h0 = chip.infer_hybrid(&input, 1.0).unwrap();
        let h1 = chip.infer_hybrid(&input, 0.0).unwrap();
        assert_eq!(d, h0);
        assert_eq!(d, h1);
    }

    #[test]
    fn infer_cascade_depth_one_matches_infer() {
        let mut chip = OmegaVChip::new(2, 0.001, 0.0);
        chip.crossbar_mut().set_resistance(0, 0, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(0, 1, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(1, 0, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(1, 1, 1.0).unwrap();
        let input = [2.0_f32, 1.0];
        let once = chip.infer(&input).unwrap();
        let cascade = chip.infer_cascade(&input, 1).unwrap();
        assert_eq!(once, cascade);
    }

    #[test]
    fn infer_cascade_depth_two_is_double_forward() {
        let mut chip = OmegaVChip::new(2, 0.001, 0.0);
        chip.crossbar_mut().set_resistance(0, 0, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(0, 1, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(1, 0, 1.0).unwrap();
        chip.crossbar_mut().set_resistance(1, 1, 1.0).unwrap();
        let input = [2.0_f32, 1.0];
        let first = chip.infer(&input).unwrap();
        let second = chip.infer(&first).unwrap();
        let cascade = chip.infer_cascade(&input, 2).unwrap();
        assert_eq!(second, cascade);
    }
}
