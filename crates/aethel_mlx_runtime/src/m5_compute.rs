//! Apple‑Silicon‑Heuristik: wann **Metal‑GPU** vs **NEON‑CPU** für Crossbar‑Matvec.
//!
//! Schwellen orientieren sich an Release‑Benchmarks (`compare_cpu_metal`) mit simdgroup‑Kernel
//! ab [`SIMD_KERNEL_MIN_N`] und Input‑Tiling ab [`TILED_KERNEL_MIN_N`] in `memristor_metal`.

#[cfg(all(feature = "memristor-metal", target_os = "macos"))]
pub use memristor_metal::{
    SIMD_KERNEL_MIN_N, TILED_CASCADE_MIN_N, TILED_KERNEL_MIN_N, TILED_KERNEL_TILE,
};

#[cfg(not(all(feature = "memristor-metal", target_os = "macos")))]
pub const SIMD_KERNEL_MIN_N: usize = 512;
#[cfg(not(all(feature = "memristor-metal", target_os = "macos")))]
pub const TILED_KERNEL_MIN_N: usize = 4096;
#[cfg(not(all(feature = "memristor-metal", target_os = "macos")))]
pub const TILED_CASCADE_MIN_N: usize = 4096;
#[cfg(not(all(feature = "memristor-metal", target_os = "macos")))]
pub const TILED_KERNEL_TILE: usize = 256;

/// Einzelnes `forward`: Metal simdgroup ab 2048 (Executor‑Routing; NEON oft noch schneller).
pub const METAL_FORWARD_MIN_N: usize = 2048;
/// Cascade depth ≥ [`METAL_CASCADE_MIN_DEPTH`]: GPU ab 2048 (tiled ab 4096 im Kernel).
pub const METAL_CASCADE_MIN_N: usize = 2048;
pub const METAL_CASCADE_MIN_DEPTH: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatvecDevice {
    CpuNeon,
    #[cfg(feature = "memristor-metal")]
    MetalGpu,
}

/// Wählt das Matvec‑Gerät für festes `n` und optionalen Cascade‑`depth`.
pub fn choose_matvec_device(n: usize, cascade_depth: Option<usize>) -> MatvecDevice {
    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    {
        if let Some(d) = cascade_depth {
            if d >= METAL_CASCADE_MIN_DEPTH && n >= METAL_CASCADE_MIN_N {
                return MatvecDevice::MetalGpu;
            }
        }
        if cascade_depth.is_none() && n >= METAL_FORWARD_MIN_N {
            return MatvecDevice::MetalGpu;
        }
    }
    let _ = (n, cascade_depth);
    MatvecDevice::CpuNeon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_n_stays_cpu() {
        assert_eq!(choose_matvec_device(64, None), MatvecDevice::CpuNeon);
    }

    #[test]
    fn forward_512_stays_cpu() {
        assert_eq!(choose_matvec_device(512, None), MatvecDevice::CpuNeon);
    }

    #[test]
    fn forward_1024_stays_cpu() {
        assert_eq!(choose_matvec_device(1024, None), MatvecDevice::CpuNeon);
    }

    #[test]
    fn forward_2048_prefers_metal_when_feature_on() {
        let d = choose_matvec_device(2048, None);
        #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
        assert_eq!(d, MatvecDevice::MetalGpu);
        #[cfg(not(all(feature = "memristor-metal", target_os = "macos")))]
        assert_eq!(d, MatvecDevice::CpuNeon);
    }

    #[test]
    fn cascade_depth_two_at_1024_stays_cpu() {
        assert_eq!(choose_matvec_device(1024, Some(2)), MatvecDevice::CpuNeon);
    }

    #[test]
    fn cascade_depth_two_at_2048_uses_metal() {
        let d = choose_matvec_device(2048, Some(2));
        #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
        assert_eq!(d, MatvecDevice::MetalGpu);
        #[cfg(not(all(feature = "memristor-metal", target_os = "macos")))]
        assert_eq!(d, MatvecDevice::CpuNeon);
    }

    #[test]
    fn cascade_depth_two_at_4096() {
        let d = choose_matvec_device(4096, Some(2));
        #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
        assert_eq!(d, MatvecDevice::MetalGpu);
        #[cfg(not(all(feature = "memristor-metal", target_os = "macos")))]
        assert_eq!(d, MatvecDevice::CpuNeon);
    }

    #[test]
    fn cascade_depth_two_at_256_stays_cpu() {
        let d = choose_matvec_device(256, Some(2));
        assert_eq!(d, MatvecDevice::CpuNeon);
    }

    #[test]
    fn shallow_cascade_small_n_cpu() {
        assert_eq!(choose_matvec_device(128, Some(1)), MatvecDevice::CpuNeon);
    }
}
