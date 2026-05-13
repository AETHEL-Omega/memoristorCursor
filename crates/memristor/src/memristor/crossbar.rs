use super::cell::MemristorCell;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum CrossbarError {
    #[error("invalid input length: expected {expected}, got {got}")]
    InvalidInputLen { expected: usize, got: usize },

    #[error("coordinates out of range: row={row}, col={col}, size={size}")]
    OutOfBounds { row: usize, col: usize, size: usize },
}

#[derive(Debug, Clone)]
pub struct Crossbar {
    size: usize,
    cells: Vec<MemristorCell>,
}

impl Crossbar {
    pub fn new(size: usize, drift_factor: f32, noise_level: f32) -> Self {
        Self {
            size,
            cells: vec![MemristorCell::new(drift_factor, noise_level); size * size],
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn resistance_at(&self, row: usize, col: usize) -> Result<f32, CrossbarError> {
        let idx = self.index_checked(row, col)?;
        Ok(self.cells[idx].resistance())
    }

    fn index_checked(&self, row: usize, col: usize) -> Result<usize, CrossbarError> {
        if row >= self.size || col >= self.size {
            return Err(CrossbarError::OutOfBounds {
                row,
                col,
                size: self.size,
            });
        }
        Ok(row * self.size + col)
    }

    pub fn set_resistance(
        &mut self,
        row: usize,
        col: usize,
        resistance: f32,
    ) -> Result<(), CrossbarError> {
        let idx = self.index_checked(row, col)?;
        self.cells[idx].set_resistance(resistance);
        Ok(())
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, CrossbarError> {
        if input.len() != self.size {
            return Err(CrossbarError::InvalidInputLen {
                expected: self.size,
                got: input.len(),
            });
        }

        let mut output = vec![0.0; self.size];
        for row in 0..self.size {
            for (col, value) in input.iter().enumerate() {
                let idx = row * self.size + col;
                let resistance = self.cells[idx].resistance();
                output[row] += *value / resistance;
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_2x2_uniform_resistance() {
        let mut bar = Crossbar::new(2, 0.001, 0.0);
        bar.set_resistance(0, 0, 1.0).unwrap();
        bar.set_resistance(0, 1, 1.0).unwrap();
        bar.set_resistance(1, 0, 1.0).unwrap();
        bar.set_resistance(1, 1, 1.0).unwrap();
        let out = bar.forward(&[1.0, 2.0]).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn forward_rejects_bad_length() {
        let bar = Crossbar::new(3, 0.001, 0.0);
        assert_eq!(
            bar.forward(&[1.0, 2.0]).unwrap_err(),
            CrossbarError::InvalidInputLen {
                expected: 3,
                got: 2,
            }
        );
    }

    #[test]
    fn set_resistance_oob() {
        let mut bar = Crossbar::new(2, 0.001, 0.0);
        assert!(matches!(
            bar.set_resistance(2, 0, 1.0),
            Err(CrossbarError::OutOfBounds { .. })
        ));
    }
}
