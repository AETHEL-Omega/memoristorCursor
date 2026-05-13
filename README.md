# memoristorCursor

Cursor workspace project for a compile-safe **Memristor** and **Omega-vChip** v1 scaffold (shared with the aiderDesktop memristor integration path).

## Included
- Rust workspace scaffold
- `memristor` crate (core simulation primitives)
- `aethel_mlx_runtime` crate (integration surface)

## Quick Start
```bash
cargo check --workspace
cargo test --workspace
```

## Crate features (`aethel_mlx_runtime`)

| Feature | Purpose |
|---------|--------|
| `memristor-kv` (default) | `MemristorKvCacheManager`, compress/decompress + session errors |
| `memristor-routing` (default) | `RouteEngine`, `InferenceExecutor` (MLX/CoreML paths are stubs) |

Slim build without integration modules:

```bash
cargo check -p aethel_mlx_runtime --no-default-features
```

KV-only or routing-only:

```bash
cargo test -p aethel_mlx_runtime --no-default-features --features memristor-kv
cargo test -p aethel_mlx_runtime --no-default-features --features memristor-routing
```

## Example: Windsurf Cascade

```bash
cargo run -p memristor --example windsurf_cascade
```
