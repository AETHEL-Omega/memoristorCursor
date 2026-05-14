//! Pluggable CoreML / digital-embed backend for [`RouteDecision::CoreMl`].

use std::sync::{Mutex, OnceLock};

type CoreMlHandler = Box<dyn Fn(&[f32]) -> Result<Vec<f32>, String> + Send + Sync>;

static COREML_HANDLER: OnceLock<Mutex<Option<CoreMlHandler>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<CoreMlHandler>> {
    COREML_HANDLER.get_or_init(|| Mutex::new(None))
}

pub fn set_coreml_handler(handler: CoreMlHandler) {
    if let Ok(mut guard) = slot().lock() {
        *guard = Some(handler);
    }
}

pub fn clear_coreml_handler() {
    if let Ok(mut guard) = slot().lock() {
        *guard = None;
    }
}

pub fn has_coreml_handler() -> bool {
    slot()
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|_| true))
        .unwrap_or(false)
}

pub fn execute_coreml(input: &[f32]) -> Vec<f32> {
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
        set_coreml_handler(Box::new(|input| {
            Ok(input.iter().map(|x| x * 3.0).collect())
        }));
        let out = execute_coreml(&[1.0, 2.0]);
        assert_eq!(out, vec![3.0, 6.0]);
        clear_coreml_handler();
    }

    #[test]
    fn passthrough_when_unset() {
        clear_coreml_handler();
        let out = execute_coreml(&[3.0, 4.0]);
        assert_eq!(out, vec![3.0, 4.0]);
    }
}
