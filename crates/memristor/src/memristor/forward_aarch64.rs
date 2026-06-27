//! Zeilensumme `sum_j input[j] * G_ij` mit **NEON** (`vfmaq` f32x4), für **Apple Silicon (M‑Serie)**.
//! Dabei ist `G_ij = 1/R_ij` (**Leitwert**); mathematisch gleich `input[j]/R_ij`, aber ohne teure SIMD‑Divisionen.

use core::arch::aarch64::*;

/// Skalar‑Fallback ohne NEON (identische Arithmetik).
#[cfg(test)]
pub(crate) fn dot_input_times_conductance_scalar(input: &[f32], conductance: &[f32]) -> f32 {
    assert_eq!(input.len(), conductance.len());
    input.iter().zip(conductance).map(|(&x, &g)| x * g).sum()
}

/// `input.len() == conductance.len()`; Leitwerte `G = 1/R`. Bei `n == 0` → `0.0`.
#[inline]
pub fn dot_input_times_conductance(input: &[f32], conductance: &[f32]) -> f32 {
    assert_eq!(input.len(), conductance.len());
    let n = input.len();
    if n == 0 {
        return 0.0;
    }
    unsafe { dot_input_times_conductance_impl(input.as_ptr(), conductance.as_ptr(), n) }
}

#[target_feature(enable = "neon")]
unsafe fn dot_input_times_conductance_impl(
    input: *const f32,
    conductance: *const f32,
    n: usize,
) -> f32 {
    let mut j = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    while j + 16 <= n {
        let vi0 = vld1q_f32(input.add(j));
        let vg0 = vld1q_f32(conductance.add(j));
        acc0 = vfmaq_f32(acc0, vi0, vg0);
        let vi1 = vld1q_f32(input.add(j + 4));
        let vg1 = vld1q_f32(conductance.add(j + 4));
        acc1 = vfmaq_f32(acc1, vi1, vg1);
        let vi2 = vld1q_f32(input.add(j + 8));
        let vg2 = vld1q_f32(conductance.add(j + 8));
        acc2 = vfmaq_f32(acc2, vi2, vg2);
        let vi3 = vld1q_f32(input.add(j + 12));
        let vg3 = vld1q_f32(conductance.add(j + 12));
        acc3 = vfmaq_f32(acc3, vi3, vg3);
        j += 16;
    }
    let mut acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    while j + 8 <= n {
        let vi0 = vld1q_f32(input.add(j));
        let vg0 = vld1q_f32(conductance.add(j));
        let vi1 = vld1q_f32(input.add(j + 4));
        let vg1 = vld1q_f32(conductance.add(j + 4));
        acc = vfmaq_f32(acc, vi0, vg0);
        acc = vfmaq_f32(acc, vi1, vg1);
        j += 8;
    }
    while j + 4 <= n {
        let vi = vld1q_f32(input.add(j));
        let vg = vld1q_f32(conductance.add(j));
        acc = vfmaq_f32(acc, vi, vg);
        j += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while j < n {
        sum += *input.add(j) * *conductance.add(j);
        j += 1;
    }
    sum
}

/// `input.len() == r_row.len()`; bei `n == 0` → `0.0`.
///
/// SIMD‑Division „`input/R`“. Für wiederholte Zugriffe ist [`dot_input_times_conductance`] mit vorbereitetem **`1/R`** schneller.
#[inline]
#[allow(dead_code)]
pub fn dot_input_over_r(input: &[f32], r_row: &[f32]) -> f32 {
    assert_eq!(input.len(), r_row.len());
    let n = input.len();
    if n == 0 {
        return 0.0;
    }
    unsafe { dot_input_over_r_impl(input.as_ptr(), r_row.as_ptr(), n) }
}

#[target_feature(enable = "neon")]
unsafe fn dot_input_over_r_impl(input: *const f32, r_row: *const f32, n: usize) -> f32 {
    let mut j = 0usize;
    let mut acc = vdupq_n_f32(0.0);
    while j + 4 <= n {
        let vi = vld1q_f32(input.add(j));
        let vr = vld1q_f32(r_row.add(j));
        let vq = vdivq_f32(vi, vr);
        acc = vaddq_f32(acc, vq);
        j += 4;
    }
    let mut sum = vaddvq_f32(acc);
    while j < n {
        sum += *input.add(j) / *r_row.add(j);
        j += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    fn scalar_over_r(input: &[f32], r: &[f32]) -> f32 {
        input.iter().zip(r).map(|(&a, &b)| a / b).sum()
    }

    #[test]
    fn neon_matches_scalar_small_and_tail() {
        let input: Vec<f32> = (0..17).map(|i| (i as f32) * 0.03 - 0.2).collect();
        let r: Vec<f32> = (0..17).map(|i| 0.85 + (i as f32) * 0.011).collect();
        let a = super::dot_input_over_r(&input, &r);
        let b = scalar_over_r(&input, &r);
        assert!((a - b).abs() < 1e-4, "a={a} b={b}");
    }

    #[test]
    fn conductance_fma_matches_scalar() {
        let input: Vec<f32> = (0..19).map(|i| (i as f32) * 0.02 - 0.15).collect();
        let g: Vec<f32> = (0..19).map(|i| 1.0 / (0.9 + (i as f32) * 0.013)).collect();
        let a = super::dot_input_times_conductance(&input, &g);
        let b = super::dot_input_times_conductance_scalar(&input, &g);
        assert!((a - b).abs() < 1e-4, "a={a} b={b}");
    }
}
