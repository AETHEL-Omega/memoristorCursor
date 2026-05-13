//! Maps [`RouteDecision`](crate::route_engine::RouteDecision) to executable paths.
//! `Mlx` / `CoreMl` are explicit stubs until real backends are wired.

use crate::route_engine::{RouteDecision, RouteEngine, RuntimeState, TaskReq};
use memristor::services::vchip_api::OmegaVChipError;

#[derive(Debug)]
pub enum ExecuteError {
    Memristor(OmegaVChipError),
}

impl std::fmt::Display for ExecuteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecuteError::Memristor(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ExecuteError {}

impl From<OmegaVChipError> for ExecuteError {
    fn from(e: OmegaVChipError) -> Self {
        ExecuteError::Memristor(e)
    }
}

pub struct InferenceExecutor {
    route_engine: RouteEngine,
}

impl InferenceExecutor {
    pub fn new(route_engine: RouteEngine) -> Self {
        Self { route_engine }
    }

    pub fn route_engine_mut(&mut self) -> &mut RouteEngine {
        &mut self.route_engine
    }

    pub fn route_engine(&self) -> &RouteEngine {
        &self.route_engine
    }

    /// Run routing + execution. `Mlx` / `CoreMl` passthrough stubs return a copy of input.
    pub fn execute(&self, req: &TaskReq, state: &RuntimeState) -> Result<Vec<f32>, ExecuteError> {
        match self.route_engine.choose_route(req, state) {
            RouteDecision::Mlx | RouteDecision::CoreMl => Ok(req.input.clone()),
            RouteDecision::Memristor => Ok(self.route_engine.execute_memristor(&req.input)?),
            RouteDecision::Hybrid { digital_weight } => {
                Ok(self.route_engine.execute_hybrid(&req.input, digital_weight)?)
            }
            RouteDecision::WindsurfCascade { depth } => {
                Ok(self.route_engine.execute_cascade(&req.input, depth)?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route_engine::{MemoryGovernor, TaskType};
    use memristor::services::vchip_api::InferenceMode;

    #[test]
    fn mlx_stub_passthrough() {
        let engine = RouteEngine::new(2, 0.001, 0.01)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Digital));
        let ex = InferenceExecutor::new(engine);
        let req = TaskReq::new(TaskType::ShortGeneration, vec![9.0, -1.5], 1);
        let out = ex.execute(&req, &RuntimeState::default()).unwrap();
        assert_eq!(out, req.input);
    }

    #[test]
    fn memristor_route_executes_chip() {
        let mut engine = RouteEngine::new(2, 0.001, 0.0)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Analog));
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
        let ex = InferenceExecutor::new(engine);
        let req = TaskReq::new(TaskType::Embed, vec![2.0, 1.0], 2);
        let out = ex.execute(&req, &RuntimeState::default()).unwrap();
        assert!((out[0] - 3.0).abs() < 1e-4);
        assert!((out[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn cascade_route_runs_multipass_chip() {
        let mut engine = RouteEngine::new(2, 0.001, 0.0)
            .with_governor(MemoryGovernor::with_inference_mode(InferenceMode::Digital));
        for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            engine.chip_mut().crossbar_mut().set_resistance(r, c, 1.0).unwrap();
        }
        let ex = InferenceExecutor::new(engine);
        let req = TaskReq::new(TaskType::ShortGeneration, vec![2.0, 1.0], 9).with_cascade_depth(2);
        let out = ex.execute(&req, &RuntimeState::default()).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 6.0).abs() < 1e-3, "got {}", out[0]);
        assert!((out[1] - 6.0).abs() < 1e-3, "got {}", out[1]);
    }
}
