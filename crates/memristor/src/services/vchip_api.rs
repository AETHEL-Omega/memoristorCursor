use crate::memristor::crossbar::{Crossbar, CrossbarError};
use rand::Rng;
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

    /// Gecachte Leitwert‑Matrix des Chips (`G_ij=1/R_ij`) — direkt für Metal / NEON‑Pfade.
    pub fn conductance_matrix(&self) -> &[f32] {
        self.crossbar.conductance_matrix()
    }

    /// Epoch der Leitwert‑Matrix (für bedingten GPU‑Upload).
    pub fn conductance_epoch(&self) -> u64 {
        self.crossbar.conductance_epoch()
    }

    /// Alle Zellwiderstände setzen (ein [`Self::conductance_epoch`]-Bump). Siehe [`Crossbar::fill_resistance`].
    pub fn fill_resistance<F: FnMut(usize, usize) -> f32>(&mut self, f: F) {
        self.crossbar.fill_resistance(f);
    }

    /// Wie [`Self::fill_resistance`], Widerstandsberechnung parallel (Feature **`memristor-parallel`**).
    #[cfg(feature = "memristor-parallel")]
    pub fn fill_resistance_par<F>(&mut self, f: F)
    where
        F: Fn(usize, usize) -> f32 + Sync,
    {
        self.crossbar.fill_resistance_par(f);
    }

    /// Ein Zeitschritt **Programmierung** auf dem Crossbar: `V_{ij} = V_row[i] − V_col[j]` pro Kreuzung.
    pub fn pulse_programming(&mut self, row_v: &[f32], col_v: &[f32]) -> OmegaVChipResult<()> {
        self.crossbar.pulse_rows_cols(row_v, col_v)?;
        Ok(())
    }

    /// Wie [`Self::pulse_programming`], mit explizitem RNG (bitweise reproduzierbar).
    pub fn pulse_programming_with_rng<R: Rng + ?Sized>(
        &mut self,
        row_v: &[f32],
        col_v: &[f32],
        rng: &mut R,
    ) -> OmegaVChipResult<()> {
        self.crossbar.pulse_rows_cols_with_rng(row_v, col_v, rng)?;
        Ok(())
    }

    /// Wie [`Self::pulse_programming`], mit Sneak‑Kopplung (siehe [`Crossbar::junction_voltage`]).
    pub fn pulse_programming_sneak(
        &mut self,
        row_v: &[f32],
        col_v: &[f32],
        sneak_strength: f32,
    ) -> OmegaVChipResult<()> {
        self.crossbar
            .pulse_rows_cols_sneak(row_v, col_v, sneak_strength)?;
        Ok(())
    }

    pub fn pulse_programming_sneak_with_rng<R: Rng + ?Sized>(
        &mut self,
        row_v: &[f32],
        col_v: &[f32],
        sneak_strength: f32,
        rng: &mut R,
    ) -> OmegaVChipResult<()> {
        self.crossbar
            .pulse_rows_cols_sneak_with_rng(row_v, col_v, sneak_strength, rng)?;
        Ok(())
    }

    /// Eine Zelle **1/2‑halbselektiv** schreiben ([`Crossbar::pulse_single_cell_half_select`]).
    pub fn pulse_single_cell_select(
        &mut self,
        row: usize,
        col: usize,
        v_prog: f32,
    ) -> OmegaVChipResult<()> {
        self.crossbar
            .pulse_single_cell_half_select(row, col, v_prog)?;
        Ok(())
    }

    pub fn pulse_single_cell_select_with_rng<R: Rng + ?Sized>(
        &mut self,
        row: usize,
        col: usize,
        v_prog: f32,
        rng: &mut R,
    ) -> OmegaVChipResult<()> {
        self.crossbar
            .pulse_single_cell_half_select_with_rng(row, col, v_prog, rng)?;
        Ok(())
    }

    pub fn pulse_single_cell_select_sneak(
        &mut self,
        row: usize,
        col: usize,
        v_prog: f32,
        sneak_strength: f32,
    ) -> OmegaVChipResult<()> {
        self.crossbar
            .pulse_single_cell_half_select_sneak(row, col, v_prog, sneak_strength)?;
        Ok(())
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
        self.crossbar
            .forward_cascade(input, depth)
            .map_err(OmegaVChipError::from)
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
    fn fill_resistance_on_chip_updates_epoch_once() {
        let mut chip = OmegaVChip::new(3, 0.001, 0.0);
        let e0 = chip.conductance_epoch();
        chip.fill_resistance(|i, j| 1.0 + 0.1 * (i + j) as f32);
        assert!(chip.conductance_epoch() > e0);
        assert!((chip.crossbar_mut().resistance_at(1, 2).unwrap() - 1.3).abs() < 1e-4);
    }

    #[cfg(feature = "memristor-parallel")]
    #[test]
    fn fill_resistance_par_matches_fill_resistance() {
        let n = 32usize;
        let mut a = OmegaVChip::new(n, 0.001, 0.0);
        let mut b = OmegaVChip::new(n, 0.001, 0.0);
        a.fill_resistance(|i, j| 0.9 + 0.001 * ((i * n + j) as f32));
        b.fill_resistance_par(|i, j| 0.9 + 0.001 * ((i * n + j) as f32));
        assert_eq!(a.conductance_matrix(), b.conductance_matrix());
    }

    #[test]
    fn pulse_programming_routes_to_crossbar() {
        let mut chip = OmegaVChip::new(2, 0.04, 0.0);
        let before = chip.crossbar_mut().resistance_at(0, 0).unwrap();
        chip.pulse_programming(&[0.9, 0.9], &[0.0, 0.0]).unwrap();
        let after = chip.crossbar_mut().resistance_at(0, 0).unwrap();
        assert!(
            (after - before).abs() > 1e-5,
            "expected resistance drift, before={before} after={after}"
        );
        let input = [1.0_f32, 0.0];
        let _ = chip.infer_analog(&input).unwrap();
    }

    #[test]
    fn pulse_single_cell_select_and_sneak_smoke() {
        let mut chip = OmegaVChip::new(3, 0.02, 0.0);
        chip.pulse_single_cell_select(1, 2, 1.5).unwrap();
        chip.pulse_programming_sneak(&[1.0, 0.0, 0.0], &[0.0, 0.0, -0.5], 0.25)
            .unwrap();
        chip.pulse_single_cell_select_sneak(0, 0, 0.9, 0.1).unwrap();
        let _ = chip.infer_analog(&[0.1, 0.2, 0.3]).unwrap();
    }

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
