use super::cell::MemristorCell;
use rand::Rng;
use thiserror::Error;

#[cfg(feature = "memristor-parallel")]
use rayon::prelude::*;

/// Ab dieser Zeilenzahl nutzt `forward` optional Zeilen-Parallelismus (Rayon). Auf **macOS** etwas früher
/// als auf anderen Zielen — Apple Silicon kann viele NEON‑Zeilen effizient parallelisieren.
#[cfg(all(feature = "memristor-parallel", target_os = "macos"))]
const PARALLEL_ROW_THRESHOLD: usize = 36;
#[cfg(all(feature = "memristor-parallel", not(target_os = "macos")))]
const PARALLEL_ROW_THRESHOLD: usize = 64;

/// Fehler beim Crossbar‑Zugriff (Lesen/Programmieren).
#[derive(Error, Debug, PartialEq, Eq)]
pub enum CrossbarError {
    #[error("invalid input length: expected {expected}, got {got}")]
    InvalidInputLen { expected: usize, got: usize },

    #[error("coordinates out of range: row={row}, col={col}, size={size}")]
    OutOfBounds { row: usize, col: usize, size: usize },
}

/// Memristor‑**Crossbar** (VMM‑Näherung für `forward`, getrennte Programmier‑APIs).
///
/// Programmierung: pro Zelle `V_ij = V_row[i] − V_col[j]`; optional **Sneak** ([`Crossbar::junction_voltage`])
/// und **1/2‑Halbselektion** ([`Crossbar::pulse_single_cell_half_select`]).
///
/// Für **M‑Serie / GPU**: die **Leitwert‑Matrix** `G_ij=1/R_ij` liegt gecacht in [`Self::conductance_matrix`]
/// (wird bei `set_resistance` / Pulsen inkrementell aktualisiert) — kein erneutes Sammeln aus Zellen pro `forward`.
#[derive(Debug, Clone)]
pub struct Crossbar {
    size: usize,
    cells: Vec<MemristorCell>,
    /// Zeilenmajor `G[row*n+col]`, synchron zu `cells`.
    conductance: Vec<f32>,
    /// Inkrement bei jeder Änderung an `conductance` — für GPU‑Upload‑Skipping ([`Self::conductance_epoch`]).
    g_epoch: u64,
}

impl Crossbar {
    pub fn new(size: usize, drift_factor: f32, noise_level: f32) -> Self {
        let cells = vec![MemristorCell::new(drift_factor, noise_level); size * size];
        let conductance = cells.iter().map(|c| c.conductance()).collect();
        Self {
            size,
            cells,
            conductance,
            g_epoch: 1,
        }
    }

    /// Monoton steigend, sobald `G` sich ändert (Programmierung / `set_resistance`). An [`memristor_metal::MetalRunner`] übergeben.
    pub fn conductance_epoch(&self) -> u64 {
        self.g_epoch
    }

    /// Gecachte Leitwert‑Matrix `G` (Länge `n²`, zeilenmajor). Direkt an [`memristor_metal::MetalRunner`] übergeben.
    pub fn conductance_matrix(&self) -> &[f32] {
        &self.conductance
    }

    /// Alle `G_ij` aus den Zellen neu berechnen (nach externen Eingriffen — normalerweise nicht nötig).
    pub fn refresh_conductance_matrix(&mut self) {
        for (c, g) in self.cells.iter().zip(self.conductance.iter_mut()) {
            *g = c.conductance();
        }
        self.g_epoch = self.g_epoch.wrapping_add(1);
    }

    #[inline]
    fn sync_conductance_at(&mut self, idx: usize) {
        self.conductance[idx] = self.cells[idx].conductance();
        self.g_epoch = self.g_epoch.wrapping_add(1);
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn resistance_at(&self, row: usize, col: usize) -> Result<f32, CrossbarError> {
        let idx = self.index_checked(row, col)?;
        Ok(self.cells[idx].resistance())
    }

    /// Leitwert `G = 1/R` (u. a. für GPU‑/NEON‑Pfade mit Multiplikation statt Division).
    pub fn conductance_at(&self, row: usize, col: usize) -> Result<f32, CrossbarError> {
        let idx = self.index_checked(row, col)?;
        Ok(self.cells[idx].conductance())
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
        self.sync_conductance_at(idx);
        Ok(())
    }

    /// Setzt alle Zellwiderstände aus `f(row, col)` in **einem** Durchlauf (ein [`Self::conductance_epoch`]-Bump).
    pub fn fill_resistance<F: FnMut(usize, usize) -> f32>(&mut self, mut f: F) {
        let n = self.size;
        for row in 0..n {
            for col in 0..n {
                let idx = row * n + col;
                let r = f(row, col);
                self.cells[idx].set_resistance(r);
                self.conductance[idx] = self.cells[idx].conductance();
            }
        }
        self.g_epoch = self.g_epoch.wrapping_add(1);
    }

    /// Wie [`Self::fill_resistance`], berechnet Widerstände parallel (Feature **`memristor-parallel`**).
    #[cfg(feature = "memristor-parallel")]
    pub fn fill_resistance_par<F>(&mut self, f: F)
    where
        F: Fn(usize, usize) -> f32 + Sync,
    {
        let n = self.size;
        let resistances: Vec<f32> = (0..n * n)
            .into_par_iter()
            .map(|idx| {
                let row = idx / n;
                let col = idx % n;
                f(row, col)
            })
            .collect();
        self.apply_resistance_bulk(&resistances);
    }

    fn apply_resistance_bulk(&mut self, resistances: &[f32]) {
        debug_assert_eq!(resistances.len(), self.size * self.size);
        for (idx, &r) in resistances.iter().enumerate() {
            self.cells[idx].set_resistance(r);
            self.conductance[idx] = self.cells[idx].conductance();
        }
        self.g_epoch = self.g_epoch.wrapping_add(1);
    }

    /// Ein Zeitschritt **Programmierung** im Sinne eines 1T1R-Crossbars: An der Kreuzung `(i,j)` liegt
    /// die Differenzspannung **`V_row[i] − V_col[j]`** über der Zelle (Bipolarität wie bei Zeilen‑ /
    /// Spalten‑Treibern; kein sneak‑/Halbleiter‑Modell).
    ///
    /// Ruft pro Zelle [`MemristorCell::apply_pulse`] auf (thread‑RNG pro Zelle).
    pub fn pulse_rows_cols(&mut self, row_v: &[f32], col_v: &[f32]) -> Result<(), CrossbarError> {
        self.validate_pair(row_v, col_v)?;
        let size = self.size;
        for i in 0..size {
            for j in 0..size {
                let idx = i * size + j;
                let v = row_v[i] - col_v[j];
                self.cells[idx].apply_pulse(v);
                self.sync_conductance_at(idx);
            }
        }
        Ok(())
    }

    /// Wie [`Self::pulse_rows_cols`], aber ein gemeinsamer RNG für **reproduzierbare** Simulationen.
    pub fn pulse_rows_cols_with_rng<R: Rng + ?Sized>(
        &mut self,
        row_v: &[f32],
        col_v: &[f32],
        rng: &mut R,
    ) -> Result<(), CrossbarError> {
        self.validate_pair(row_v, col_v)?;
        let size = self.size;
        for i in 0..size {
            for j in 0..size {
                let idx = i * size + j;
                let v = row_v[i] - col_v[j];
                self.cells[idx].apply_pulse_with_rng(v, rng);
                self.sync_conductance_at(idx);
            }
        }
        Ok(())
    }

    /// Direkt ein einzelnes Matrixelement mit Spannung `voltage` treiben (ohne Zeilen-/Spalten‑Differenz).
    pub fn pulse_at(&mut self, row: usize, col: usize, voltage: f32) -> Result<(), CrossbarError> {
        let idx = self.index_checked(row, col)?;
        self.cells[idx].apply_pulse(voltage);
        self.sync_conductance_at(idx);
        Ok(())
    }

    fn validate_pair(&self, row_v: &[f32], col_v: &[f32]) -> Result<(), CrossbarError> {
        let n = self.size;
        if row_v.len() != n {
            return Err(CrossbarError::InvalidInputLen {
                expected: n,
                got: row_v.len(),
            });
        }
        if col_v.len() != n {
            return Err(CrossbarError::InvalidInputLen {
                expected: n,
                got: col_v.len(),
            });
        }
        Ok(())
    }

    /// Spannung am Knoten **`(i,j)`** zur Vorschau / Diagnose: ideale Differenz `V_row[i]−V_col[j]` plus
    /// optionaler Sneak‑Term (gleiche Formel wie [`Self::pulse_rows_cols_sneak`]).
    pub fn junction_voltage(
        row_v: &[f32],
        col_v: &[f32],
        i: usize,
        j: usize,
        sneak_strength: f32,
    ) -> Option<f32> {
        let n = row_v.len();
        if n == 0 || col_v.len() != n || i >= n || j >= n {
            return None;
        }
        let v_ideal = row_v[i] - col_v[j];
        if sneak_strength <= 0.0 {
            return Some(v_ideal);
        }
        let nf = n as f32;
        let mean_abs_row = row_v.iter().map(|x| x.abs()).sum::<f32>() / nf;
        let mean_abs_col = col_v.iter().map(|x| x.abs()).sum::<f32>() / nf;
        Some(v_ideal + sneak_strength * mean_abs_row * mean_abs_col)
    }

    /// Wie [`Self::pulse_rows_cols`], mit **Sneak‑Kopplung** (siehe [`Self::junction_voltage`]).
    pub fn pulse_rows_cols_sneak(
        &mut self,
        row_v: &[f32],
        col_v: &[f32],
        sneak_strength: f32,
    ) -> Result<(), CrossbarError> {
        self.validate_pair(row_v, col_v)?;
        let size = self.size;
        for i in 0..size {
            for j in 0..size {
                let idx = i * size + j;
                let v = Self::junction_voltage(row_v, col_v, i, j, sneak_strength)
                    .expect("lengths validated");
                self.cells[idx].apply_pulse(v);
                self.sync_conductance_at(idx);
            }
        }
        Ok(())
    }

    /// Wie [`Self::pulse_rows_cols_sneak`], mit gemeinsamem RNG.
    pub fn pulse_rows_cols_sneak_with_rng<R: Rng + ?Sized>(
        &mut self,
        row_v: &[f32],
        col_v: &[f32],
        sneak_strength: f32,
        rng: &mut R,
    ) -> Result<(), CrossbarError> {
        self.validate_pair(row_v, col_v)?;
        let size = self.size;
        for i in 0..size {
            for j in 0..size {
                let idx = i * size + j;
                let v = Self::junction_voltage(row_v, col_v, i, j, sneak_strength)
                    .expect("lengths validated");
                self.cells[idx].apply_pulse_with_rng(v, rng);
                self.sync_conductance_at(idx);
            }
        }
        Ok(())
    }

    /// **Halbselektion (1/2‑Schema)** für eine Zelle `(row_sel, col_sel)`:  
    /// `V_row[row_sel]=V_prog/2`, `V_col[col_sel]=−V_prog/2`, alle anderen Zeilen/Spalten `0`.  
    /// Dann: Zielzelle sieht **`V_prog`**, Halbselektierte auf derselben Zeile/Spalte **`V_prog/2`**, der Rest **`0`**.
    pub fn pulse_single_cell_half_select(
        &mut self,
        row_sel: usize,
        col_sel: usize,
        v_prog: f32,
    ) -> Result<(), CrossbarError> {
        self.index_checked(row_sel, col_sel)?;
        let n = self.size;
        let mut row_v = vec![0.0_f32; n];
        let mut col_v = vec![0.0_f32; n];
        row_v[row_sel] = v_prog * 0.5;
        col_v[col_sel] = -v_prog * 0.5;
        self.pulse_rows_cols(&row_v, &col_v)
    }

    /// Wie [`Self::pulse_single_cell_half_select`], mit gemeinsamem RNG.
    pub fn pulse_single_cell_half_select_with_rng<R: Rng + ?Sized>(
        &mut self,
        row_sel: usize,
        col_sel: usize,
        v_prog: f32,
        rng: &mut R,
    ) -> Result<(), CrossbarError> {
        self.index_checked(row_sel, col_sel)?;
        let n = self.size;
        let mut row_v = vec![0.0_f32; n];
        let mut col_v = vec![0.0_f32; n];
        row_v[row_sel] = v_prog * 0.5;
        col_v[col_sel] = -v_prog * 0.5;
        self.pulse_rows_cols_with_rng(&row_v, &col_v, rng)
    }

    /// 1/2‑Selektion wie [`Self::pulse_single_cell_half_select`], aber mit **Sneak** auf dem gleichen Treibernuster.
    pub fn pulse_single_cell_half_select_sneak(
        &mut self,
        row_sel: usize,
        col_sel: usize,
        v_prog: f32,
        sneak_strength: f32,
    ) -> Result<(), CrossbarError> {
        self.index_checked(row_sel, col_sel)?;
        let n = self.size;
        let mut row_v = vec![0.0_f32; n];
        let mut col_v = vec![0.0_f32; n];
        row_v[row_sel] = v_prog * 0.5;
        col_v[col_sel] = -v_prog * 0.5;
        self.pulse_rows_cols_sneak(&row_v, &col_v, sneak_strength)
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, CrossbarError> {
        if input.len() != self.size {
            return Err(CrossbarError::InvalidInputLen {
                expected: self.size,
                got: input.len(),
            });
        }

        #[cfg(feature = "memristor-parallel")]
        if self.size >= PARALLEL_ROW_THRESHOLD {
            return self.forward_parallel(input);
        }

        Ok(self.forward_sequential(input))
    }

    /// Wie [`Self::forward`], schreibt in `output` (**Länge `n`**) ohne neue `Vec`‑Allocation.
    pub fn forward_into(&self, input: &[f32], output: &mut [f32]) -> Result<(), CrossbarError> {
        if input.len() != self.size {
            return Err(CrossbarError::InvalidInputLen {
                expected: self.size,
                got: input.len(),
            });
        }
        if output.len() != self.size {
            return Err(CrossbarError::InvalidInputLen {
                expected: self.size,
                got: output.len(),
            });
        }
        #[cfg(feature = "memristor-parallel")]
        if self.size >= PARALLEL_ROW_THRESHOLD {
            let par = self.forward_parallel(input)?;
            output.copy_from_slice(&par);
            return Ok(());
        }
        self.forward_sequential_into(input, output);
        Ok(())
    }

    /// `depth` mal \(y\leftarrow Gy\) auf der CPU (gecachtes `G`). `depth == 0` → Kopie von `input`.
    pub fn forward_cascade(&self, input: &[f32], depth: usize) -> Result<Vec<f32>, CrossbarError> {
        if input.len() != self.size {
            return Err(CrossbarError::InvalidInputLen {
                expected: self.size,
                got: input.len(),
            });
        }
        if depth == 0 {
            return Ok(input.to_vec());
        }
        let mut x = input.to_vec();
        for _ in 0..depth {
            x = self.forward_sequential(&x);
        }
        Ok(x)
    }

    /// Wie [`Self::forward_cascade`], nutzt `output` als letzten Schritt‑Puffer wenn `depth > 0`.
    pub fn forward_cascade_into(
        &self,
        input: &[f32],
        depth: usize,
        output: &mut [f32],
    ) -> Result<(), CrossbarError> {
        if depth == 0 {
            if input.len() != self.size || output.len() != self.size {
                return Err(CrossbarError::InvalidInputLen {
                    expected: self.size,
                    got: input.len().max(output.len()),
                });
            }
            output.copy_from_slice(input);
            return Ok(());
        }
        if input.len() != self.size || output.len() != self.size {
            return Err(CrossbarError::InvalidInputLen {
                expected: self.size,
                got: input.len(),
            });
        }
        let n = self.size;
        let mut a = input.to_vec();
        let mut b = vec![0.0_f32; n];
        for pass in 0..depth {
            if pass + 1 == depth {
                self.forward_sequential_into(&a, output);
            } else {
                self.forward_sequential_into(&a, &mut b);
                std::mem::swap(&mut a, &mut b);
            }
        }
        Ok(())
    }

    pub(crate) fn forward_sequential(&self, input: &[f32]) -> Vec<f32> {
        let n = self.size;
        let mut output = vec![0.0; n];
        self.forward_sequential_into(input, &mut output);
        output
    }

    pub(crate) fn forward_sequential_into(&self, input: &[f32], output: &mut [f32]) {
        let n = self.size;
        let g = &self.conductance;
        #[cfg(target_arch = "aarch64")]
        {
            for row in 0..n {
                let row_g = &g[row * n..row * n + n];
                output[row] = super::forward_aarch64::dot_input_times_conductance(input, row_g);
            }
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for row in 0..n {
                let mut sum = 0.0_f32;
                let base = row * n;
                for col in 0..n {
                    sum += input[col] * g[base + col];
                }
                output[row] = sum;
            }
        }
    }

    #[cfg(feature = "memristor-parallel")]
    pub(crate) fn forward_parallel(&self, input: &[f32]) -> Result<Vec<f32>, CrossbarError> {
        let size = self.size;
        let g = &self.conductance;
        #[cfg(target_arch = "aarch64")]
        {
            let out: Vec<f32> = (0..size)
                .into_par_iter()
                .map(|row| {
                    let row_g = &g[row * size..row * size + size];
                    super::forward_aarch64::dot_input_times_conductance(input, row_g)
                })
                .collect();
            Ok(out)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let out: Vec<f32> = (0..size)
                .into_par_iter()
                .map(|row| {
                    let base = row * size;
                    let mut sum = 0.0_f32;
                    for col in 0..size {
                        sum += input[col] * g[base + col];
                    }
                    sum
                })
                .collect();
            Ok(out)
        }
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

    #[test]
    fn pulse_rows_cols_rejects_bad_length() {
        let mut bar = Crossbar::new(2, 0.01, 0.0);
        let e = bar.pulse_rows_cols(&[0.0], &[0.0, 0.0]).unwrap_err();
        assert_eq!(
            e,
            CrossbarError::InvalidInputLen {
                expected: 2,
                got: 1,
            }
        );
    }

    #[test]
    fn pulse_rows_cols_zeros_no_change_without_noise() {
        let mut bar = Crossbar::new(2, 0.05, 0.0);
        bar.set_resistance(0, 0, 2.0).unwrap();
        bar.set_resistance(0, 1, 3.0).unwrap();
        bar.set_resistance(1, 0, 4.0).unwrap();
        bar.set_resistance(1, 1, 5.0).unwrap();
        let mut before = Vec::with_capacity(4);
        for i in 0..2 {
            for j in 0..2 {
                before.push(bar.resistance_at(i, j).unwrap());
            }
        }
        bar.pulse_rows_cols(&[0.0, 0.0], &[0.0, 0.0]).unwrap();
        let mut after = Vec::with_capacity(4);
        for i in 0..2 {
            for j in 0..2 {
                after.push(bar.resistance_at(i, j).unwrap());
            }
        }
        assert_eq!(before, after);
    }

    #[test]
    fn pulse_uniform_field_moves_all_cells_same() {
        let mut bar = Crossbar::new(2, 0.02, 0.0);
        for _ in 0..30 {
            bar.pulse_rows_cols(&[1.0, 1.0], &[0.0, 0.0]).unwrap();
        }
        let r00 = bar.resistance_at(0, 0).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (bar.resistance_at(i, j).unwrap() - r00).abs() < 1e-4,
                    "i={i} j={j}"
                );
            }
        }
    }

    #[test]
    fn pulse_rows_cols_with_rng_is_deterministic() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut a = Crossbar::new(2, 0.01, 0.05);
        let mut b = Crossbar::new(2, 0.01, 0.05);
        let seed = [3u8; 32];
        let mut ra = StdRng::from_seed(seed);
        let mut rb = StdRng::from_seed(seed);
        let rv = [0.45_f32, -0.12];
        let cv = [0.11, 0.28];
        a.pulse_rows_cols_with_rng(&rv, &cv, &mut ra).unwrap();
        b.pulse_rows_cols_with_rng(&rv, &cv, &mut rb).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let da = a.resistance_at(i, j).unwrap();
                let db = b.resistance_at(i, j).unwrap();
                assert!((da - db).abs() < 1e-5, "i={i} j={j} da={da} db={db}");
            }
        }
    }

    #[test]
    fn half_select_junction_voltages_2x2() {
        let v_prog = 2.0_f32;
        let row = [v_prog * 0.5, 0.0];
        let col = [0.0, -v_prog * 0.5];
        let eps = 1e-5;
        assert!((Crossbar::junction_voltage(&row, &col, 0, 1, 0.0).unwrap() - v_prog).abs() < eps);
        assert!(
            (Crossbar::junction_voltage(&row, &col, 0, 0, 0.0).unwrap() - v_prog * 0.5).abs() < eps
        );
        assert!(
            (Crossbar::junction_voltage(&row, &col, 1, 1, 0.0).unwrap() - v_prog * 0.5).abs() < eps
        );
        assert!(
            Crossbar::junction_voltage(&row, &col, 1, 0, 0.0)
                .unwrap()
                .abs()
                < eps
        );
    }

    #[test]
    fn sneak_adds_mean_row_times_mean_col() {
        let row = [1.0_f32, 0.0];
        let col = [0.0_f32, -1.0];
        let ideal = Crossbar::junction_voltage(&row, &col, 0, 1, 0.0).unwrap();
        let with_sneak = Crossbar::junction_voltage(&row, &col, 0, 1, 1.0).unwrap();
        assert!((ideal - 2.0).abs() < 1e-5);
        // mean(|row|)=0.5, mean(|col|)=0.5 → Produkt 0.25
        assert!((with_sneak - ideal - 0.25).abs() < 1e-4);
    }

    #[test]
    fn pulse_sneak_differs_from_ideal_on_active_lines() {
        let mut a = Crossbar::new(2, 0.03, 0.0);
        let mut b = Crossbar::new(2, 0.03, 0.0);
        let row = [1.0_f32, 0.0];
        let col = [0.0_f32, -1.0];
        for _ in 0..40 {
            a.pulse_rows_cols(&row, &col).unwrap();
            b.pulse_rows_cols_sneak(&row, &col, 1.0).unwrap();
        }
        assert!(
            (a.resistance_at(0, 1).unwrap() - b.resistance_at(0, 1).unwrap()).abs() > 1e-4,
            "sneak should perturb switching at least at target cell"
        );
    }

    #[test]
    fn fill_resistance_bumps_epoch_once() {
        let mut bar = Crossbar::new(4, 0.001, 0.0);
        let e0 = bar.conductance_epoch();
        bar.fill_resistance(|_, _| 1.25);
        assert!(bar.conductance_epoch() > e0);
        let e1 = bar.conductance_epoch();
        for i in 0..4 {
            for j in 0..4 {
                assert!((bar.resistance_at(i, j).unwrap() - 1.25).abs() < 1e-5);
            }
        }
        bar.fill_resistance(|i, j| 1.0 + 0.01 * (i + j) as f32);
        assert!(bar.conductance_epoch() > e1);
    }

    #[test]
    fn conductance_matrix_tracks_resistance_and_pulse() {
        let mut bar = Crossbar::new(2, 0.05, 0.0);
        bar.set_resistance(0, 0, 2.0).unwrap();
        let g0 = bar.conductance_matrix()[0];
        assert!((g0 - 0.5).abs() < 1e-5);
        let e0 = bar.conductance_epoch();
        bar.pulse_at(0, 0, 0.8).unwrap();
        let g1 = bar.conductance_matrix()[0];
        assert!(g1 != g0);
        assert!(bar.conductance_epoch() > e0);
        assert!((bar.conductance_at(0, 0).unwrap() - g1).abs() < 1e-6);
    }

    #[test]
    fn forward_cascade_depth_two_matches_double_forward() {
        let mut bar = Crossbar::new(4, 0.001, 0.0);
        for i in 0..4 {
            for j in 0..4 {
                bar.set_resistance(i, j, 1.0).unwrap();
            }
        }
        let input = [1.0_f32, 0.5, 0.25, 0.125];
        let once = bar.forward(&input).unwrap();
        let twice = bar.forward(&once).unwrap();
        let cascade = bar.forward_cascade(&input, 2).unwrap();
        assert_eq!(twice, cascade);
    }

    #[cfg(feature = "memristor-parallel")]
    #[test]
    fn forward_parallel_matches_sequential_large() {
        let size = 128_usize;
        let mut bar = Crossbar::new(size, 0.001, 0.0);
        for i in 0..size {
            for j in 0..size {
                bar.set_resistance(i, j, 1.0).unwrap();
            }
        }
        let input: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let seq = bar.forward_sequential(&input);
        let par = bar.forward_parallel(&input).unwrap();
        assert_eq!(seq.len(), par.len());
        for (a, b) in seq.iter().zip(par.iter()) {
            assert!((a - b).abs() < 1e-5, "seq {a} par {b}");
        }
    }
}
