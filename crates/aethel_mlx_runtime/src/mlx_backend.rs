//! Pluggable MLX / digital backend for [`RouteDecision::Mlx`] (browser wires Candle).

use std::sync::{Mutex, OnceLock};

type MlxHandler = Box<dyn Fn(&[f32]) -> Result<Vec<f32>, String> + Send + Sync>;

static MLX_HANDLER: OnceLock<Mutex<Option<MlxHandler>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<MlxHandler>> {
    MLX_HANDLER.get_or_init(|| Mutex::new(None))
}

/// Register MLX executor (e.g. Candle Metal matvec from aethelBrowser).
pub fn set_mlx_handler(handler: MlxHandler) {
    if let Ok(mut guard) = slot().lock() {
        *guard = Some(handler);
    }
}

pub fn clear_mlx_handler() {
    if let Ok(mut guard) = slot().lock() {
        *guard = None;
    }
}

pub fn has_mlx_handler() -> bool {
    slot()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|_| true))
        .unwrap_or(false)
}

/// Run registered MLX backend or passthrough input when unset.
pub fn execute_mlx(input: &[f32]) -> Vec<f32> {
    if let Ok(guard) = slot().lock() {
        if let Some(handler) = guard.as_ref() {
            if let Ok(out) = handler(input) {
                if out.len() == input.len() {
                    return out;
                }
            }
        }
    }
    input.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registered_handler_runs() {
        set_mlx_handler(Box::new(|input| {
            Ok(input.iter().map(|x| x * 2.0).collect())
        }));
        let out = execute_mlx(&[1.0, 2.0]);
        assert_eq!(out, vec![2.0, 4.0]);
        clear_mlx_handler();
    }

    #[test]
    fn passthrough_when_unset() {
        clear_mlx_handler();
        let out = execute_mlx(&[3.0, 4.0]);
        assert_eq!(out, vec![3.0, 4.0]);
    }
}
