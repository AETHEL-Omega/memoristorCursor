//! Route decision scaffold for Digital / Memristor / Hybrid execution.
//! Full hardware governor integration is Phase 3.

use memristor::services::vchip_api::{InferenceMode, OmegaVChip};
use std::fmt;

/// Stub task categories for routing prototypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskType {
    Embed,
    Classify,
    Rerank,
    ShortGeneration,
    LongGeneration,
}

/// Where to execute the next step.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteDecision {
    Mlx,
    CoreMl,
    Memristor,
    Hybrid { digital_weight: f32 },
    /// Multi-pass virtual crossbar (Windsurf Cascade prototype).
    WindsurfCascade { depth: usize },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimeState {}

/// Memory / thermal stub: replace with real signals in Phase 3.
#[derive(Debug, Clone)]
pub struct MemoryGovernor {
    forced_mode: Option<InferenceMode>,
}

impl MemoryGovernor {
    pub fn new() -> Self {
        Self { forced_mode: None }
    }

    pub fn with_inference_mode(mode: InferenceMode) -> Self {
        Self {
            forced_mode: Some(mode),
        }
    }

    pub fn inference_mode(&self) -> InferenceMode {
        self.forced_mode.unwrap_or(InferenceMode::Digital)
    }
}

impl Default for MemoryGovernor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TaskReq {
    pub task_type: TaskType,
    pub input: Vec<f32>,
    pub session_id: u64,
    /// When `Some(d)` with `d > 0` for [`TaskType::Embed`] or [`TaskType::ShortGeneration`],
    /// routes to [`RouteDecision::WindsurfCascade`].
    pub cascade_depth: Option<usize>,
}

impl TaskReq {
    pub fn new(task_type: TaskType, input: Vec<f32>, session_id: u64) -> Self {
        Self {
            task_type,
            input,
            session_id,
            cascade_depth: None,
        }
    }

    /// Chaining helper for cascade routing experiments.
    pub fn with_cascade_depth(mut self, depth: usize) -> Self {
        self.cascade_depth = Some(depth);
        self
    }
}

impl fmt::Debug for TaskReq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskReq")
            .field("task_type", &self.task_type)
            .field("input_len", &self.input.len())
            .field("session_id", &self.session_id)
            .field("cascade_depth", &self.cascade_depth)
            .finish()
    }
}

pub struct RouteEngine {
    governor: MemoryGovernor,
    vchip: OmegaVChip,
}

impl RouteEngine {
    pub fn new(crossbar_side: usize, drift: f32, noise: f32) -> Self {
        Self {
            governor: MemoryGovernor::new(),
            vchip: OmegaVChip::new(crossbar_side, drift, noise),
        }
    }

    pub fn with_governor(mut self, governor: MemoryGovernor) -> Self {
        self.governor = governor;
        self
    }

    pub fn vchip_len(&self) -> usize {
        self.vchip.len()
    }

    /// Documented default blend for hybrid (digital vs memristor analog path).
    pub const DEFAULT_HYBRID_DIGITAL_WEIGHT: f32 = 0.7;

    pub fn choose_route(&self, req: &TaskReq, state: &RuntimeState) -> RouteDecision {
        let _ = state;
        if let Some(depth) = req.cascade_depth {
            if depth > 0
                && matches!(
                    req.task_type,
                    TaskType::Embed | TaskType::ShortGeneration
                )
            {
                return RouteDecision::WindsurfCascade { depth };
            }
        }
        let mode = self.governor.inference_mode();

        match req.task_type {
            TaskType::LongGeneration => RouteDecision::Mlx,
            TaskType::Embed | TaskType::Classify | TaskType::Rerank => match mode {
                InferenceMode::Digital => RouteDecision::CoreMl,
                InferenceMode::Analog => RouteDecision::Memristor,
                InferenceMode::Hybrid => RouteDecision::Hybrid {
                    digital_weight: Self::DEFAULT_HYBRID_DIGITAL_WEIGHT,
                },
            },
            TaskType::ShortGeneration => match mode {
                InferenceMode::Digital => RouteDecision::Mlx,
                InferenceMode::Analog => RouteDecision::Memristor,
                InferenceMode::Hybrid => RouteDecision::Hybrid {
                    digital_weight: Self::DEFAULT_HYBRID_DIGITAL_WEIGHT,
                },
            },
        }
    }

    pub fn chip_mut(&mut self) -> &mut OmegaVChip {
        &mut self.vchip
    }

    pub fn execute_memristor(&self, input: &[f32]) -> Result<Vec<f32>, memristor::services::vchip_api::OmegaVChipError> {
        self.vchip.infer_analog(input)
    }

    pub fn execute_hybrid(
        &self,
        input: &[f32],
        digital_weight: f32,
    ) -> Result<Vec<f32>, memristor::services::vchip_api::OmegaVChipError> {
        self.vchip.infer_hybrid(input, digital_weight)
    }

    pub fn execute_cascade(
        &self,
        input: &[f32],
        depth: usize,
    ) -> Result<Vec<f32>, memristor::services::vchip_api::OmegaVChipError> {
        self.vchip.infer_cascade(input, depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_generation_always_mlx() {
        let engine = RouteEngine::new(8, 0.001, 0.01)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
        let req = TaskReq {
            task_type: TaskType::LongGeneration,
            input: vec![0.1; 8],
            session_id: 1,
            cascade_depth: None,
        };
        assert_eq!(
            engine.choose_route(&req, &RuntimeState::default()),
            RouteDecision::Mlx
        );
    }

    #[test]
    fn embed_analog_targets_memristor() {
        let engine = RouteEngine::new(3, 0.001, 0.01)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
        let req = TaskReq {
            task_type: TaskType::Embed,
            input: vec![0.0; 3],
            session_id: 2,
            cascade_depth: None,
        };
        assert_eq!(
            engine.choose_route(&req, &RuntimeState::default()),
            RouteDecision::Memristor
        );
    }

    #[test]
    fn execute_memristor_smoke() {
        let mut engine = RouteEngine::new(2, 0.001, 0.0);
        engine
            .chip_mut()
            .crossbar_mut()
            .set_resistance(0, 0, 1.0)
            .unwrap();
        engine
            .chip_mut()
            .crossbar_mut()
            .set_resistance(0, 1, 1.0)
            .unwrap();
        engine
            .chip_mut()
            .crossbar_mut()
            .set_resistance(1, 0, 1.0)
            .unwrap();
        engine
            .chip_mut()
            .crossbar_mut()
            .set_resistance(1, 1, 1.0)
            .unwrap();
        let out = engine.execute_memristor(&[2.0, 1.0]).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-4);
        assert!((out[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn embed_cascade_depth_overrides_governor() {
        let engine = RouteEngine::new(3, 0.001, 0.01)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
        let req = TaskReq::new(TaskType::Embed, vec![0.0; 3], 3).with_cascade_depth(2);
        assert_eq!(
            engine.choose_route(&req, &RuntimeState::default()),
            RouteDecision::WindsurfCascade { depth: 2 }
        );
    }

    #[test]
    fn long_generation_ignores_cascade_pref() {
        let engine = RouteEngine::new(4, 0.001, 0.01);
        let req = TaskReq::new(TaskType::LongGeneration, vec![0.0; 4], 1).with_cascade_depth(3);
        assert_eq!(
            engine.choose_route(&req, &RuntimeState::default()),
            RouteDecision::Mlx
        );
    }
}
