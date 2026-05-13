use rand::Rng;

#[derive(Debug, Clone)]
pub struct MemristorCell {
    resistance: f32,
    drift_factor: f32,
    noise_level: f32,
}

impl MemristorCell {
    pub fn new(drift_factor: f32, noise_level: f32) -> Self {
        Self {
            resistance: 1.0,
            drift_factor: drift_factor.clamp(0.0, 0.1),
            noise_level: noise_level.clamp(0.0, 0.1),
        }
    }

    pub fn resistance(&self) -> f32 {
        self.resistance
    }

    pub fn set_resistance(&mut self, resistance: f32) {
        self.resistance = resistance.clamp(0.1, 10.0);
    }

    pub fn apply_pulse(&mut self, voltage: f32) {
        let mut rng = rand::thread_rng();
        self.apply_pulse_with_rng(voltage, &mut rng);
    }

    /// Deterministic variant for tests or seeded simulations.
    pub fn apply_pulse_with_rng<R: Rng + ?Sized>(&mut self, voltage: f32, rng: &mut R) {
        let noise = self.noise_level * rng.gen::<f32>();
        let delta = (voltage + noise).signum() * self.drift_factor.max(0.0001);
        self.set_resistance(self.resistance + delta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn set_resistance_clamps_low() {
        let mut cell = MemristorCell::new(0.01, 0.0);
        cell.set_resistance(0.01);
        assert!((cell.resistance() - 0.1).abs() < 1e-5);
    }

    #[test]
    fn set_resistance_clamps_high() {
        let mut cell = MemristorCell::new(0.01, 0.0);
        cell.set_resistance(100.0);
        assert!((cell.resistance() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn apply_pulse_with_rng_is_deterministic() {
        let mut a = MemristorCell::new(0.01, 0.05);
        let mut b = MemristorCell::new(0.01, 0.05);
        let seed = [7u8; 32];
        let mut rng_a = StdRng::from_seed(seed);
        let mut rng_b = StdRng::from_seed(seed);
        a.apply_pulse_with_rng(0.5, &mut rng_a);
        b.apply_pulse_with_rng(0.5, &mut rng_b);
        assert_eq!(a.resistance(), b.resistance());
    }

    #[test]
    fn apply_pulse_keeps_within_bounds_after_many_steps() {
        let mut cell = MemristorCell::new(0.05, 0.05);
        let mut rng = StdRng::from_seed([42u8; 32]);
        for _ in 0..500 {
            let v = rng.gen::<f32>() * 2.0 - 1.0;
            cell.apply_pulse_with_rng(v, &mut rng);
            assert!(
                (0.1..=10.0).contains(&cell.resistance()),
                "got {}",
                cell.resistance()
            );
        }
    }
}
