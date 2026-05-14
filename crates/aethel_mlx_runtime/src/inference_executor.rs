//! Maps [`RouteDecision`](crate::route_engine::RouteDecision) to executable paths.
//! `Mlx` / `CoreMl` are explicit stubs until real backends are wired.
//!
//! Mit Feature **`memristor-metal`**: große Crossbars / tiefe Cascades laufen über
//! [`crate::m5_compute::choose_matvec_device`] auf der **Metal‑GPU** (macOS).

use crate::m5_compute::{choose_matvec_device, MatvecDevice};
use crate::route_engine::{RouteDecision, RouteEngine, RuntimeState, TaskReq};
use memristor::services::vchip_api::OmegaVChipError;

#[derive(Debug)]
pub enum ExecuteError {
    Memristor(OmegaVChipError),
    #[cfg(feature = "memristor-metal")]
    Metal(String),
}

impl std::fmt::Display for ExecuteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecuteError::Memristor(e) => write!(f, "{e}"),
            #[cfg(feature = "memristor-metal")]
            ExecuteError::Metal(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for ExecuteError {}

impl From<OmegaVChipError> for ExecuteError {
    fn from(e: OmegaVChipError) -> Self {
        ExecuteError::Memristor(e)
    }
}

#[cfg(feature = "memristor-metal")]
impl From<memristor_metal::MetalForwardError> for ExecuteError {
    fn from(e: memristor_metal::MetalForwardError) -> Self {
        ExecuteError::Metal(e.to_string())
    }
}

/// Result of [`InferenceExecutor::execute_detailed`]: routing choice plus optional M5 matvec backend.
#[derive(Debug, Clone)]
pub struct ExecuteOutcome {
    pub output: Vec<f32>,
    pub route: RouteDecision,
    /// `Some` when the memristor / cascade path ran; `None` for MLX / CoreML / hybrid stubs.
    pub matvec_device: Option<MatvecDevice>,
    /// Metal‑Kernelstufe, wenn [`MatvecDevice::MetalGpu`] genutzt wurde.
    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    pub metal_kernel: Option<memristor_metal::MetalKernelTier>,
}

pub struct InferenceExecutor {
    route_engine: RouteEngine,
    #[cfg(feature = "memristor-metal")]
    metal_runner: Option<memristor_metal::MetalRunner>,
}

impl InferenceExecutor {
    pub fn new(route_engine: RouteEngine) -> Self {
        Self {
            route_engine,
            #[cfg(feature = "memristor-metal")]
            metal_runner: None,
        }
    }

    pub fn route_engine_mut(&mut self) -> &mut RouteEngine {
        &mut self.route_engine
    }

    pub fn route_engine(&self) -> &RouteEngine {
        &self.route_engine
    }

    /// Run routing + execution. `Mlx` / `CoreMl` passthrough stubs return a copy of input.
    pub fn execute(&mut self, req: &TaskReq, state: &RuntimeState) -> Result<Vec<f32>, ExecuteError> {
        Ok(self.execute_detailed(req, state)?.output)
    }

    /// Like [`execute`](Self::execute), but records [`RouteDecision`] and chosen [`MatvecDevice`].
    pub fn execute_detailed(
        &mut self,
        req: &TaskReq,
        state: &RuntimeState,
    ) -> Result<ExecuteOutcome, ExecuteError> {
        let route = self.route_engine.choose_route(req, state);
        let (output, matvec_device) = match route {
            RouteDecision::Mlx => (crate::mlx_backend::execute_mlx(&req.input), None),
            RouteDecision::CoreMl => (req.input.clone(), None),
            RouteDecision::Hybrid { digital_weight } => (
                self.route_engine
                    .execute_hybrid(&req.input, digital_weight)?,
                None,
            ),
            RouteDecision::Memristor => {
                let (out, dev) = self.execute_matvec(&req.input, 0, req.matvec_repeats)?;
                (out, Some(dev))
            }
            RouteDecision::WindsurfCascade { depth } => {
                let (out, dev) = self.execute_matvec(&req.input, depth, req.matvec_repeats)?;
                (out, Some(dev))
            }
        };
        #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
        let metal_kernel = Self::metal_kernel_tier_for_outcome(
            self.route_engine.vchip_len(),
            &route,
            matvec_device,
        );
        Ok(ExecuteOutcome {
            output,
            route,
            matvec_device,
            #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
            metal_kernel,
        })
    }

    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    fn metal_kernel_tier_for_outcome(
        n: usize,
        route: &RouteDecision,
        device: Option<MatvecDevice>,
    ) -> Option<memristor_metal::MetalKernelTier> {
        if device != Some(MatvecDevice::MetalGpu) {
            return None;
        }
        let cascade = matches!(route, RouteDecision::WindsurfCascade { .. });
        Some(memristor_metal::metal_kernel_tier(n, cascade))
    }

    fn execute_matvec(
        &mut self,
        input: &[f32],
        depth: usize,
        repeats: usize,
    ) -> Result<(Vec<f32>, MatvecDevice), ExecuteError> {
        let repeats = repeats.max(1);
        let n = self.route_engine.vchip_len();
        let cascade_opt = if depth > 0 { Some(depth) } else { None };
        let device = choose_matvec_device(n, cascade_opt);
        let output = match device {
            MatvecDevice::CpuNeon => {
                let mut out = if depth == 0 {
                    self.route_engine.execute_memristor(input)?
                } else {
                    self.route_engine.execute_cascade(input, depth)?
                };
                for _ in 1..repeats {
                    out = if depth == 0 {
                        self.route_engine.execute_memristor(input)?
                    } else {
                        self.route_engine.execute_cascade(input, depth)?
                    };
                }
                out
            }
            #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
            MatvecDevice::MetalGpu => self.execute_matvec_metal(input, depth, repeats)?,
        };
        Ok((output, device))
    }

    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    fn execute_matvec_metal(
        &mut self,
        input: &[f32],
        depth: usize,
        repeats: usize,
    ) -> Result<Vec<f32>, ExecuteError> {
        use memristor_metal::MetalRunner;

        let n = self.route_engine.vchip_len();
        let g = self.route_engine.chip().conductance_matrix();
        let epoch = self.route_engine.chip().conductance_epoch();
        let runner = self
            .metal_runner
            .get_or_insert_with(|| MetalRunner::new().expect("MetalRunner::new"));
        let mut out = vec![0.0_f32; n];
        if depth == 0 {
            runner.forward_repeated_into(n, g, epoch, input, repeats, &mut out)?;
        } else {
            runner.forward_cascade_repeated_into(n, g, epoch, input, depth, repeats, &mut out)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_engine::{MemoryGovernor, TaskType};
    use memristor::services::vchip_api::InferenceMode;

    fn uniform_chip(engine: &mut RouteEngine, _n: usize) {
        engine.chip_mut().fill_resistance(|_, _| 1.0);
    }

    fn matvec_close(got: f32, expected: f32) -> bool {
        let diff = (got - expected).abs();
        let scale = got.abs().max(expected.abs()).max(1.0);
        diff <= 1e-5 * scale + 2.0
    }

    #[test]
    fn mlx_stub_passthrough() {
        let engine = RouteEngine::new(2, 0.001, 0.01)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Digital));
        let mut ex = InferenceExecutor::new(engine);
        let req = TaskReq::new(TaskType::ShortGeneration, vec![9.0, -1.5], 1);
        let out = ex.execute(&req, &RuntimeState::default()).unwrap();
        assert_eq!(out, req.input);
    }

    #[test]
    fn memristor_route_executes_chip() {
        let mut engine = RouteEngine::new(2, 0.001, 0.0)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
        uniform_chip(&mut engine, 2);
        let mut ex = InferenceExecutor::new(engine);
        let req = TaskReq::new(TaskType::Embed, vec![2.0, 1.0], 2);
        let out = ex.execute(&req, &RuntimeState::default()).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-4);
        assert!((out[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn cascade_route_runs_multipass_chip() {
        let mut engine = RouteEngine::new(2, 0.001, 0.0)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Digital));
        uniform_chip(&mut engine, 2);
        let mut ex = InferenceExecutor::new(engine);
        let req = TaskReq::new(TaskType::ShortGeneration, vec![2.0, 1.0], 9).with_cascade_depth(2);
        let out = ex.execute(&req, &RuntimeState::default()).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 6.0).abs() < 1e-3, "got {}", out[0]);
        assert!((out[1] - 6.0).abs() < 1e-3, "got {}", out[1]);
    }

    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    #[test]
    fn large_forward_matvec_repeats_matches_cpu() {
        let n = 2048usize;
        let repeats = 4usize;
        let mut engine = RouteEngine::new(n, 0.001, 0.0)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
        uniform_chip(&mut engine, n);
        let mut ex = InferenceExecutor::new(engine);
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
        let req = TaskReq::new(TaskType::Embed, input.clone(), 1).with_matvec_repeats(repeats);
        let outcome = ex.execute_detailed(&req, &RuntimeState::default()).unwrap();
        assert_eq!(outcome.matvec_device, Some(MatvecDevice::MetalGpu));
        let cpu_ref = ex.route_engine().execute_memristor(&input).unwrap();
        for i in 0..n {
            assert!(
                matvec_close(outcome.output[i], cpu_ref[i]),
                "i={} metal {} cpu {}",
                i,
                outcome.output[i],
                cpu_ref[i]
            );
        }
    }

    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    #[test]
    fn large_cascade_matvec_repeats_matches_cpu() {
        let n = 2048usize;
        let repeats = 3usize;
        let mut engine = RouteEngine::new(n, 0.001, 0.0);
        uniform_chip(&mut engine, n);
        let mut ex = InferenceExecutor::new(engine);
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
        let req = TaskReq::new(TaskType::Embed, input.clone(), 1)
            .with_cascade_depth(2)
            .with_matvec_repeats(repeats);
        let outcome = ex.execute_detailed(&req, &RuntimeState::default()).unwrap();
        assert_eq!(outcome.matvec_device, Some(MatvecDevice::MetalGpu));
        let cpu_ref = ex.route_engine().execute_cascade(&input, 2).unwrap();
        for i in 0..n {
            assert!(
                matvec_close(outcome.output[i], cpu_ref[i]),
                "i={} metal {} cpu {}",
                i,
                outcome.output[i],
                cpu_ref[i]
            );
        }
    }

    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    #[test]
    fn large_forward_uses_metal_simd_at_2048() {
        let n = 2048usize;
        let mut engine = RouteEngine::new(n, 0.001, 0.0)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
        uniform_chip(&mut engine, n);
        let mut ex = InferenceExecutor::new(engine);
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
        let req = TaskReq::new(TaskType::Embed, input.clone(), 1);
        let outcome = ex.execute_detailed(&req, &RuntimeState::default()).unwrap();
        assert_eq!(outcome.matvec_device, Some(MatvecDevice::MetalGpu));
        assert_eq!(
            outcome.metal_kernel,
            Some(memristor_metal::MetalKernelTier::SimdgroupBatch)
        );
        let cpu_ref = ex.route_engine().execute_memristor(&input).unwrap();
        for i in 0..n {
            assert!(
                matvec_close(outcome.output[i], cpu_ref[i]),
                "i={} metal {} cpu {}",
                i,
                outcome.output[i],
                cpu_ref[i]
            );
        }
    }

    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    #[test]
    fn large_cascade_uses_metal_when_enabled() {
        let n = 2048usize;
        let mut engine = RouteEngine::new(n, 0.001, 0.0);
        uniform_chip(&mut engine, n);
        let mut ex = InferenceExecutor::new(engine);
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
        let req = TaskReq::new(TaskType::Embed, input.clone(), 1).with_cascade_depth(2);
        let outcome = ex.execute_detailed(&req, &RuntimeState::default()).unwrap();
        assert_eq!(outcome.matvec_device, Some(MatvecDevice::MetalGpu));
        assert_eq!(
            outcome.metal_kernel,
            Some(memristor_metal::MetalKernelTier::Simdgroup)
        );
        let cpu_ref = ex
            .route_engine()
            .execute_cascade(&input, 2)
            .unwrap();
        assert_eq!(outcome.output.len(), n);
        for i in 0..n {
            assert!(
                matvec_close(outcome.output[i], cpu_ref[i]),
                "i={} metal {} cpu {}",
                i,
                outcome.output[i],
                cpu_ref[i]
            );
        }
    }

    #[cfg(all(feature = "memristor-metal", target_os = "macos"))]
    #[test]
    fn programming_pulse_rematches_cpu_after_g_epoch_bump() {
        let n = 2048usize;
        let mut engine = RouteEngine::new(n, 0.001, 0.0)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
        uniform_chip(&mut engine, n);
        let mut ex = InferenceExecutor::new(engine);
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let req = TaskReq::new(TaskType::Embed, input.clone(), 2).with_cascade_depth(2);
        let before = ex.execute_detailed(&req, &RuntimeState::default()).unwrap();
        assert_eq!(before.matvec_device, Some(MatvecDevice::MetalGpu));
        ex.route_engine_mut()
            .chip_mut()
            .pulse_programming(&vec![0.5; n], &vec![0.0; n])
            .unwrap();
        let after = ex.execute_detailed(&req, &RuntimeState::default()).unwrap();
        let cpu_ref = ex.route_engine().execute_cascade(&input, 2).unwrap();
        assert_eq!(after.output.len(), n);
        for i in 0..n {
            assert!(
                matvec_close(after.output[i], cpu_ref[i]),
                "i={} gpu {} cpu {}",
                i,
                after.output[i],
                cpu_ref[i]
            );
        }
        assert_ne!(before.output, after.output);
    }

    #[test]
    fn small_memristor_reports_cpu_neon_device() {
        let mut engine = RouteEngine::new(2, 0.001, 0.0)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
        uniform_chip(&mut engine, 2);
        let mut ex = InferenceExecutor::new(engine);
        let req = TaskReq::new(TaskType::Embed, vec![2.0, 1.0], 2);
        let outcome = ex.execute_detailed(&req, &RuntimeState::default()).unwrap();
        assert_eq!(outcome.matvec_device, Some(MatvecDevice::CpuNeon));
    }
}
