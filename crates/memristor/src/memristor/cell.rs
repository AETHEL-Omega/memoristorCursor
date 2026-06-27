use rand::Rng;

/// Einzelzelle für ein **vereinfachtes bipolares Memristor-/ReRAM-Modell** (Idee wie HP/Strukov, stark
/// reduziert für Simulation).
///
/// Statt den Widerstand direkt um eine Konstante zu schubsen, führen wir eine **normierte innere
/// Variable** `w ∈ [0,1]` (z. B. Filamentlänge / Dotiergrad). Der Widerstand folgt
///
/// `R(w) = R_on + (1 - w) · (R_off - R_on)`  →  `w = 1`: niedrigster Widerstand (LRS), `w = 0`: höchster (HRS).
///
/// Impulse verschieben `w` proportional zur **Polarität und effektiven Spannung** und nutzen ein
/// **Joglekar-Fenster** `F(w) = 1 - (2w-1)^{2p}`, damit die Drift an den Rändern ausläuft (kein
/// „Überschießen“ jenseits der physikalischen Grenzen).
#[derive(Debug, Clone)]
pub struct MemristorCell {
    /// Normierter Zustand ∈ [0, 1].
    w: f32,
    r_on: f32,
    r_off: f32,
    /// Effektive Empfindlichkeit pro Impuls (Skalierung der Schritte in `w`; entspricht in etwa
    /// dem früheren `drift_factor`, ist aber in **Zustandsraum** definiert).
    alpha: f32,
    /// Rauschen auf die Spannung (wie bisher: skalierte [0,1)-Stichprobe).
    noise_level: f32,
    /// Mindest-|V| (nach Rauschaddition), ab dem geschaltet wird — 0 ⇒ wie früher jeder nicht-triviale Impuls wirkt.
    v_th: f32,
    /// Referenzspannung für die Skalierung `|V|/v_ref` (typisch 1 V simuliert).
    v_ref: f32,
    /// Obere Kappe für `|V|` in der dw-Formel (numerische Sicherheit).
    v_clip: f32,
    /// Fenster-Ordnung `p` in `1 - (2w-1)^{2p}`.
    window_p: u32,
}

impl MemristorCell {
    const DEFAULT_R_ON: f32 = 0.1;
    const DEFAULT_R_OFF: f32 = 10.0;

    /// Erzeugt eine Zelle mit Standard-Rändern (`R_on=0.1`, `R_off=10`) und Startwiderstand **1.0 Ω**
    /// (wie zuvor implizit durch `resistance: 1.0`).
    ///
    /// `drift_factor` wird als **`alpha`** (Schrittweite in `w`, skaliert mit Spannung/Fenster) genutzt.
    pub fn new(drift_factor: f32, noise_level: f32) -> Self {
        let r_on = Self::DEFAULT_R_ON;
        let r_off = Self::DEFAULT_R_OFF;
        let initial_r = 1.0_f32;
        let w = Self::w_from_resistance(initial_r, r_on, r_off);
        Self {
            w,
            r_on,
            r_off,
            alpha: drift_factor.max(0.0),
            noise_level: noise_level.clamp(0.0, 0.1),
            v_th: 0.0,
            v_ref: 1.0,
            v_clip: 10.0,
            window_p: 1,
        }
    }

    /// Wie [`Self::new`], plus explizite **Schwellspannung** (nach Rauschen): kleinere |V| ändern den Zustand nicht.
    pub fn with_voltage_threshold(drift_factor: f32, noise_level: f32, v_th: f32) -> Self {
        let mut c = Self::new(drift_factor, noise_level);
        c.v_th = v_th.max(0.0);
        c
    }

    /// Explizite LRS/HRS-Grenzen (Ω); `R_off > R_on > 0`. Erhält den aktuellen **physikalischen** Widerstand,
    /// sofern er in die neuen Grenzen passt — sonst auf den nächsten Rand geklemmt.
    pub fn with_resistance_bounds(mut self, r_on: f32, r_off: f32) -> Self {
        let current = self.r();
        let r_on = r_on.max(1e-6);
        let r_off = r_off.max(r_on * (1.0 + 1e-3));
        self.r_on = r_on;
        self.r_off = r_off;
        let r_clamped = current.clamp(r_on, r_off);
        self.w = Self::w_from_resistance(r_clamped, r_on, r_off);
        self
    }

    fn w_from_resistance(r: f32, r_on: f32, r_off: f32) -> f32 {
        let span = (r_off - r_on).max(1e-9);
        let r = r.clamp(r_on, r_off);
        1.0 - (r - r_on) / span
    }

    /// Aktueller Widerstand R(w) ∈ [R_on, R_off].
    pub fn resistance(&self) -> f32 {
        self.r()
    }

    /// Normierter innerer Zustand `w ∈ [0,1]` (LRS bei `w→1`, HRS bei `w→0`).
    #[inline]
    pub fn normalized_state(&self) -> f32 {
        self.w
    }

    /// Leitwert `G = 1/R` (für Crossbars, die in G rechnen wollen).
    #[inline]
    pub fn conductance(&self) -> f32 {
        1.0 / self.r().max(1e-9)
    }

    /// Voltage threshold for switching
    #[inline]
    pub fn voltage_threshold(&self) -> f32 {
        self.v_th
    }

    /// Decay rate / sensitivity factor
    #[inline]
    pub fn decay_rate(&self) -> f32 {
        self.alpha
    }

    #[inline]
    fn r(&self) -> f32 {
        self.r_on + (1.0 - self.w) * (self.r_off - self.r_on)
    }

    pub fn set_resistance(&mut self, resistance: f32) {
        let r = resistance.clamp(self.r_on, self.r_off);
        self.w = Self::w_from_resistance(r, self.r_on, self.r_off);
    }

    pub fn apply_pulse(&mut self, voltage: f32) {
        let mut rng = rand::thread_rng();
        self.apply_pulse_with_rng(voltage, &mut rng);
    }

    /// Physikalisch inspirierter Impuls: `Δw ∝ sign(V_eff) · (|V_eff|-V_th)_+ · F(w) · |V_eff|`, mit Cap und Fenster.
    pub fn apply_pulse_with_rng<R: Rng + ?Sized>(&mut self, voltage: f32, rng: &mut R) {
        let noise = self.noise_level * rng.gen::<f32>();
        let v = voltage + noise;
        let mag = (v.abs() - self.v_th).max(0.0);
        if mag <= 0.0 {
            return;
        }
        let sign = v.signum();
        if sign == 0.0 {
            return;
        }
        let v_eff_mag = mag.min(self.v_clip);
        let g_w = joglekar_window(self.w, self.window_p);
        let span = (self.r_off - self.r_on).max(1e-6);
        let a = self.alpha.max(1e-4); // wie früher: auch bei `alpha=0` ein minimaler Schritt möglich
        let dw = sign * a * g_w * (v_eff_mag / self.v_ref.max(1e-9)) / span;
        self.w = (self.w + dw).clamp(0.0, 1.0);
    }
}

/// Joglekar et al.: `F(w) = 1 - (2w-1)^{2p}`, null am Rand, maximal bei w=0.5.
#[inline]
fn joglekar_window(w: f32, p: u32) -> f32 {
    let w = w.clamp(0.0, 1.0);
    let exp = 2 * p.max(1) as i32;
    let d = 2.0 * w - 1.0;
    (1.0 - d.powi(exp)).max(0.0)
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
        assert!((cell.resistance() - MemristorCell::DEFAULT_R_ON).abs() < 1e-5);
    }

    #[test]
    fn set_resistance_clamps_high() {
        let mut cell = MemristorCell::new(0.01, 0.0);
        cell.set_resistance(100.0);
        assert!((cell.resistance() - MemristorCell::DEFAULT_R_OFF).abs() < 1e-4);
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
                (MemristorCell::DEFAULT_R_ON..=MemristorCell::DEFAULT_R_OFF)
                    .contains(&cell.resistance()),
                "got {}",
                cell.resistance()
            );
        }
    }

    #[test]
    fn voltage_threshold_blocks_small_excursions() {
        let r_start = 1.0_f32;
        let mut gated = MemristorCell::with_voltage_threshold(0.1, 0.0, 0.5);
        assert!((gated.resistance() - r_start).abs() < 1e-3);
        gated.apply_pulse_with_rng(0.2, &mut StdRng::from_seed([9u8; 32]));
        assert!(
            (gated.resistance() - r_start).abs() < 1e-3,
            "gated moved to {}",
            gated.resistance()
        );

        let mut free = MemristorCell::new(0.1, 0.0);
        assert!((free.resistance() - r_start).abs() < 1e-3);
        free.apply_pulse_with_rng(0.2, &mut StdRng::from_seed([9u8; 32]));
        assert!(
            (free.resistance() - r_start).abs() > 1e-5,
            "expected movement, got {}",
            free.resistance()
        );
    }

    #[test]
    fn joglekar_reduces_motion_near_rails() {
        let mut hi = MemristorCell::new(0.05, 0.0);
        hi.set_resistance(MemristorCell::DEFAULT_R_OFF * 0.999);
        let r0 = hi.resistance();
        hi.apply_pulse_with_rng(0.3, &mut StdRng::from_seed([1u8; 32]));
        let dr_hi = (hi.resistance() - r0).abs();

        let mut mid = MemristorCell::new(0.05, 0.0);
        mid.set_resistance(1.0);
        let r1 = mid.resistance();
        mid.apply_pulse_with_rng(0.3, &mut StdRng::from_seed([1u8; 32]));
        let dr_mid = (mid.resistance() - r1).abs();

        assert!(dr_mid > dr_hi * 1.5, "edge dr {dr_hi} mid dr {dr_mid}");
    }
}
