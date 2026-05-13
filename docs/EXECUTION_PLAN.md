# Memristor v1 Execution Plan (memoristorCursor)

## Phase 1
- Stabilize `memristor` crate API.
- Keep ASCII-safe naming (`OmegaVChip`).
- Add deterministic unit tests for cell and crossbar behavior.

## Phase 2
- Integrate KV-cache compression into runtime. *(done: stored tensor + budget bytes)*
- Add session budget accounting and not-found error paths. *(done: `DecompressKvError::SessionNotFound`)*
- Gate rollout with feature flags. *(done: `memristor-kv` / `memristor-routing`; `--no-default-features` empty surface)*

## Phase 3
- Add hybrid route decision path. *(scaffold: `RouteDecision::Hybrid`, `infer_hybrid`)*
- Keep fallback behavior explicit and tested. *(added `InferenceExecutor`: MLX/CoreML = input passthrough stubs)*
- Windsurf Cascade: `infer_cascade` on `OmegaVChip`, optional `TaskReq::cascade_depth` → `RouteDecision::WindsurfCascade` *(done)*

## Verification
```bash
cargo check --workspace
cargo test --workspace
```

## Phase 4 (optional next)
- Criterion benchmarks behind `memristor-bench` (adds dev-deps when enabled).
- Example binary: `cargo run -p memristor --example windsurf_cascade` *(done)*.
