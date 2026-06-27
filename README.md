# memoristorCursor

Cursor workspace project for a compile-safe **Memristor** and **Omega-vChip** v1 scaffold (shared with the aiderDesktop memristor integration path).

**Apple Silicon / M5:** see [docs/M5_APPLE_SILICON.md](docs/M5_APPLE_SILICON.md) — UMA, CPU vs GPU vs ANE, current limits of this repo, roadmap (Metal / Core ML / NEON).

## Metal GPU MVP (`memristor_metal`, nur macOS)

Compute-Shader für denselben Crossbar-Matvec wie die CPU, mit **Shared Metal Buffers** (UMA-tauglich):

```bash
cargo test -p memristor_metal
cargo run -p memristor_metal --example metal_smoke
# optional: CPU vs GPU timings (pipeline cached via OnceLock; use --release for steady times)
cargo run -p memristor_metal --example compare_cpu_metal --release
```

Siehe [docs/M5_APPLE_SILICON.md](docs/M5_APPLE_SILICON.md) Abschnitt *Metal-MVP*.

## Included
- Rust workspace scaffold
- `memristor` crate (core simulation primitives)
- `memristor_metal` crate (Metal compute MVP, macOS)
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
| `memristor-parallel` | Forwards to `memristor` with Rayon row-parallel `Crossbar::forward` (large `n`) |

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

## Apple-Silicon CPU: parallel crossbar (optional)

For large `n`, enable Rayon row-parallel `Crossbar::forward` (same numerics per row as sequential):

```bash
cargo test -p memristor --features memristor-parallel
cargo build -p aethel_mlx_runtime --features memristor-parallel
```

| Feature (`memristor`) | Effect |
|----------------------|--------|
| `memristor-parallel` | `n ≥ 64`: parallel row pass; smaller `n`: sequential (less Rayon overhead). |

Crossbar on **GPU (Metal)** is in **`memristor_metal`** (macOS). For the full CPU vs GPU vs ANE picture see [docs/M5_APPLE_SILICON.md](docs/M5_APPLE_SILICON.md).
