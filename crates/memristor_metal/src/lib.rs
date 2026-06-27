//! Metal compute path for memristor crossbar forward (`y = M x` with conductance **`G_ij = 1/R_ij`** —
//! gleiche Interpretation wie der NEON‑CPU‑Pfad: `sum_j input[j] * G_ij`).
//!
//! On macOS: uses **`MTLStorageModeShared`** buffers so CPU-filled data is visible to the GPU without a
//! discrete VRAM copy — the typical Unified Memory pattern on Apple silicon.
//!
//! The compute **library and pipeline are built once** (see [`warm_metal_pipeline`]) via an internal
//! `OnceLock` cache; **shared `MTLBuffer`s are pooled** per matrix size `n` and reused across calls (CPU
//! writes into the same storage the GPU reads — Unified Memory). Per-call work is memcpy + encode + GPU.
//!
//! For hot loops, construct a [`MetalRunner`] once and call [`MetalRunner::forward`]: pools live in the
//! runner (no global mutex). [`crossbar_forward_metal`] uses a **thread-local** [`MetalRunner`] so the
//! default API stays lock-free while still reusing buffers per thread.
//!
//! The MSL kernels **unroll** the inner sum (scalar path) or use **simdgroup `simd_sum`**
//! with one threadgroup per output row (large `n`, see [`SIMD_KERNEL_MIN_N`]) or **input tiling**
//! in threadgroup memory at [`TILED_KERNEL_MIN_N`].
//!
//! **Cascade** (Windsurf‑Stil \(y\leftarrow Gy\) mehrfach): [`MetalRunner::forward_cascade_into`] bündelt
//! mehrere Dispatches **plus** Geräteinterne `MTLBlit`-Kopien (Output→Input) in **einem** Command Buffer –
//! weniger Roundtrips/Zeitschlitze gegenüber wiederholt [`MetalRunner::forward_into`].
//!
//! Für wiederholtes **gleiches** \(y=Gx\) (Warm‑Loops, Benchmarks): [`MetalRunner::forward_repeated_into`]
//! encodiert `repeats` unabhängige Forward‑Passes in **einem** Command Buffer mit **einem** GPU‑Sync.
//!
//! On other platforms: [`crossbar_forward_metal`] returns [`MetalForwardError::UnsupportedPlatform`].

use std::fmt;

/// Prime the Metal library + compute pipeline cache (macOS). Safe to call multiple times.
#[cfg(target_os = "macos")]
pub fn warm_metal_pipeline() -> Result<(), MetalForwardError> {
    macos::warm_cache()
}

#[cfg(not(target_os = "macos"))]
pub fn warm_metal_pipeline() -> Result<(), MetalForwardError> {
    Err(MetalForwardError::UnsupportedPlatform)
}

/// One-shot crossbar forward on GPU (macOS). `conductance` must be **`G_ij = 1/R_ij`** (Leitwert‑Matrix).
#[cfg(target_os = "macos")]
pub fn crossbar_forward_metal(
    n: usize,
    conductance: &[f32],
    input: &[f32],
) -> Result<Vec<f32>, MetalForwardError> {
    macos::crossbar_forward_metal_impl(n, conductance, input)
}

#[cfg(not(target_os = "macos"))]
pub fn crossbar_forward_metal(
    _n: usize,
    _conductance: &[f32],
    _input: &[f32],
) -> Result<Vec<f32>, MetalForwardError> {
    Err(MetalForwardError::UnsupportedPlatform)
}

/// `depth` mal \(y\leftarrow Gy\) auf der GPU (**ein** Command Buffer**,** Leitwert `G=1/R` wie [`crossbar_forward_metal`]).
/// `depth == 0`: Rückgabe ist eine Kopie von `input` (kein Kernel).
#[cfg(target_os = "macos")]
pub fn crossbar_forward_metal_cascade(
    n: usize,
    conductance: &[f32],
    input: &[f32],
    depth: usize,
) -> Result<Vec<f32>, MetalForwardError> {
    macos::crossbar_forward_metal_cascade_impl(n, conductance, input, depth)
}

#[cfg(not(target_os = "macos"))]
pub fn crossbar_forward_metal_cascade(
    _n: usize,
    _conductance: &[f32],
    _input: &[f32],
    _depth: usize,
) -> Result<Vec<f32>, MetalForwardError> {
    Err(MetalForwardError::UnsupportedPlatform)
}

/// Ein Schritt `Crossbar::forward` auf der GPU — nutzt [`memristor::memristor::crossbar::Crossbar::conductance_matrix`].
#[cfg(target_os = "macos")]
pub fn forward_crossbar(
    bar: &memristor::memristor::crossbar::Crossbar,
    input: &[f32],
) -> Result<Vec<f32>, MetalForwardError> {
    let n = bar.len();
    crossbar_forward_metal(n, bar.conductance_matrix(), input)
}

/// Cascade \(y\leftarrow Gy\) `depth`‑mal auf der GPU (ein Command Buffer, siehe [`MetalRunner::forward_cascade`]).
#[cfg(target_os = "macos")]
pub fn forward_crossbar_cascade(
    bar: &memristor::memristor::crossbar::Crossbar,
    input: &[f32],
    depth: usize,
) -> Result<Vec<f32>, MetalForwardError> {
    let n = bar.len();
    crossbar_forward_metal_cascade(n, bar.conductance_matrix(), input, depth)
}

/// Wie [`MetalRunner::forward_into`], liest Epoch aus `bar`.
#[cfg(target_os = "macos")]
pub fn forward_crossbar_into(
    runner: &mut MetalRunner,
    bar: &memristor::memristor::crossbar::Crossbar,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), MetalForwardError> {
    let n = bar.len();
    runner.forward_into(
        n,
        bar.conductance_matrix(),
        bar.conductance_epoch(),
        input,
        output,
    )
}

/// Ab dieser Kreuzgröße wird der einzeilige simdgroup‑Kernel statt skalar genutzt.
pub const SIMD_KERNEL_MIN_N: usize = 512;
/// Ab dieser Kreuzgröße: **32 Zeilen / Threadgroup** (`crossbar_forward_simd_batch`, nur Forward).
pub const SIMD_BATCH_MIN_N: usize = 1024;
/// Zeilen pro Threadgroup im Batch‑simd‑Kernel (= 32 SIMD‑Lanes‑Gruppen à 32 Threads).
pub const SIMD_BATCH_ROWS_PER_GROUP: usize = 32;
/// Forward nutzt **kein** Tiling (simdgroup reicht); reserviert für API‑Kompatibilität.
pub const TILED_KERNEL_MIN_N: usize = usize::MAX;
/// Tiling schon ab hier in **Cascade**‑Passes (GPU‑Routing ab 2048).
pub const TILED_CASCADE_MIN_N: usize = 4096;
/// Tile‑Breite in `f32`‑Elementen für Tiling‑Kernel.
pub const TILED_KERNEL_TILE: usize = 256;

/// Welcher Metal‑Compute‑Kernel für gegebenes `n` gewählt wird.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalKernelTier {
    Scalar,
    Simdgroup,
    /// Forward: 32 Ausgabezeilen pro Threadgroup (weniger Dispatch‑Overhead).
    SimdgroupBatch,
    Tiled,
}

/// Kernel‑Stufe für `n` (wie [`macos::encode_matvec_pass`]).
pub fn metal_kernel_tier(n: usize, cascade_pass: bool) -> MetalKernelTier {
    #[cfg(target_os = "macos")]
    {
        return macos::kernel_tier_for_n(n, cascade_pass);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (n, cascade_pass);
        MetalKernelTier::Scalar
    }
}

/// GPU crossbar runner with pooled shared `MTLBuffer`s (macOS only). See crate docs.
#[cfg(target_os = "macos")]
pub use macos::MetalRunner;

#[derive(Debug)]
pub enum MetalForwardError {
    UnsupportedPlatform,
    NoGpuDevice,
    InvalidShape {
        n: usize,
        conductance_len: usize,
        input_len: usize,
    },
    Metal(String),
}

impl fmt::Display for MetalForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalForwardError::UnsupportedPlatform => {
                write!(f, "memristor_metal is only available on macOS")
            }
            MetalForwardError::NoGpuDevice => write!(f, "no default Metal device"),
            MetalForwardError::InvalidShape {
                n,
                conductance_len,
                input_len,
            } => write!(
                f,
                "expected conductance len n*n ({n}²) and input len n ({n}), got conductance={conductance_len} input={input_len}"
            ),
            MetalForwardError::Metal(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for MetalForwardError {}

#[cfg(target_os = "macos")]
mod macos {
    use super::MetalForwardError;
    use metal::{
        Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
        ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSRange,
    };
    use std::cell::RefCell;
    use std::slice;
    use std::sync::OnceLock;

    const SIMD_KERNEL_MIN_N: usize = super::SIMD_KERNEL_MIN_N;
    const SIMD_BATCH_MIN_N: usize = super::SIMD_BATCH_MIN_N;
    const SIMD_BATCH_ROWS: u64 = super::SIMD_BATCH_ROWS_PER_GROUP as u64;
    const TILED_CASCADE_MIN_N: usize = super::TILED_CASCADE_MIN_N;
    const TILED_KERNEL_TILE: usize = super::TILED_KERNEL_TILE;
    const SIMDGROUP_WIDTH: u64 = 32;

    pub(super) fn kernel_tier_for_n(n: usize, cascade_pass: bool) -> super::MetalKernelTier {
        if cascade_pass && n >= TILED_CASCADE_MIN_N {
            super::MetalKernelTier::Tiled
        } else if !cascade_pass && n >= SIMD_BATCH_MIN_N {
            super::MetalKernelTier::SimdgroupBatch
        } else if n >= SIMD_KERNEL_MIN_N {
            super::MetalKernelTier::Simdgroup
        } else {
            super::MetalKernelTier::Scalar
        }
    }

    fn use_simd_batch_kernel(n: usize, cascade_pass: bool) -> bool {
        !cascade_pass && n >= SIMD_BATCH_MIN_N
    }

    fn use_tiled_kernel(n: usize, cascade_pass: bool) -> bool {
        cascade_pass && n >= TILED_CASCADE_MIN_N
    }

    fn use_simd_kernel(n: usize, cascade_pass: bool) -> bool {
        n >= SIMD_KERNEL_MIN_N
            && !use_simd_batch_kernel(n, cascade_pass)
            && !use_tiled_kernel(n, cascade_pass)
    }

    const MSL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Leitwerte G = 1/R (wie NEON‑CPU): y[i] = sum_j fma(input[j], G[i*N+j], …). Weniger Divisionen.
kernel void crossbar_forward(
    device const float *conductance [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    const device uint *N_buf [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    uint N = N_buf[0];
    if (i >= N) return;
    float sum = 0.0f;
    uint j = 0;
    for (; j + 3 < N; j += 4) {
        uint b = i * N + j;
        sum = fma(input[j],     conductance[b],     sum);
        sum = fma(input[j + 1], conductance[b + 1], sum);
        sum = fma(input[j + 2], conductance[b + 2], sum);
        sum = fma(input[j + 3], conductance[b + 3], sum);
    }
    for (; j < N; j++) {
        uint b = i * N + j;
        sum = fma(input[j], conductance[b], sum);
    }
    output[i] = sum;
}

// Große N: ein Threadgroup pro Ausgabezeile; 32 SIMD‑Lanes teilen die innere Summe.
kernel void crossbar_forward_simd(
    device const float *conductance [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    const device uint *N_buf [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]])
{
    uint N = N_buf[0];
    if (row >= N) return;
    float sum = 0.0f;
    uint base = row * N;
    uint j = lane;
    for (; j + 3 * 32 < N; j += 32 * 4) {
        sum = fma(input[j],           conductance[base + j],           sum);
        sum = fma(input[j + 32],       conductance[base + j + 32],       sum);
        sum = fma(input[j + 64],       conductance[base + j + 64],       sum);
        sum = fma(input[j + 96],       conductance[base + j + 96],       sum);
    }
    for (; j < N; j += 32) {
        sum = fma(input[j], conductance[base + j], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}

// Forward großes N: 32 Zeilen / Threadgroup (32×32 Threads), pro Zeile simdgroup‑Reduktion.
kernel void crossbar_forward_simd_batch(
    device const float *conductance [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    const device uint *N_buf [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]])
{
    uint N = N_buf[0];
    uint row = tg * 32u + lid / 32u;
    if (row >= N) return;
    uint lane = lid % 32u;
    float sum = 0.0f;
    uint base = row * N;
    uint j = lane;
    for (; j + 3 * 32 < N; j += 32 * 4) {
        sum = fma(input[j],           conductance[base + j],           sum);
        sum = fma(input[j + 32],       conductance[base + j + 32],       sum);
        sum = fma(input[j + 64],       conductance[base + j + 64],       sum);
        sum = fma(input[j + 96],       conductance[base + j + 96],       sum);
    }
    for (; j < N; j += 32) {
        sum = fma(input[j], conductance[base + j], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}

// Sehr großes N: Input‑Segmente in threadgroup‑Speicher, dann simdgroup‑Reduktion pro Tile.
#define TILE 256

kernel void crossbar_forward_tiled(
    device const float *conductance [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    const device uint *N_buf [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    threadgroup float *tg_input [[threadgroup(0)]])
{
    uint N = N_buf[0];
    if (row >= N) return;
    float sum = 0.0f;
    uint base = row * N;
    for (uint tile_start = 0; tile_start < N; tile_start += TILE) {
        uint tile_len = (N - tile_start < (uint)TILE) ? (N - tile_start) : (uint)TILE;
        for (uint k = lane; k < TILE; k += 32) {
            tg_input[k] = (k < tile_len) ? input[tile_start + k] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint j = lane;
        for (; j + 3 * 32 < tile_len; j += 32 * 4) {
            sum = fma(tg_input[j],           conductance[base + tile_start + j],           sum);
            sum = fma(tg_input[j + 32],       conductance[base + tile_start + j + 32],       sum);
            sum = fma(tg_input[j + 64],       conductance[base + tile_start + j + 64],       sum);
            sum = fma(tg_input[j + 96],       conductance[base + tile_start + j + 96],       sum);
        }
        for (; j < tile_len; j += 32) {
            sum = fma(tg_input[j], conductance[base + tile_start + j], sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum;
    }
}
"#;

    struct CachedMetal {
        device: Device,
        queue: CommandQueue,
        pipeline: ComputePipelineState,
        pipeline_simd: ComputePipelineState,
        pipeline_simd_batch: ComputePipelineState,
        pipeline_tiled: ComputePipelineState,
    }

    struct PooledBuffers {
        n: usize,
        conductance: Buffer,
        input: Buffer,
        /// Kopie des Host‑`input` für GPU‑Blit‑Reset zwischen Cascade‑Repeats im selben Command Buffer.
        input_anchor: Buffer,
        output: Buffer,
        n_uint: Buffer,
    }

    static CACHE: OnceLock<Result<CachedMetal, String>> = OnceLock::new();

    fn build_cache() -> Result<CachedMetal, String> {
        let device =
            Device::system_default().ok_or_else(|| "no default Metal device".to_string())?;
        let compile_options = CompileOptions::new();
        compile_options.set_fast_math_enabled(false);
        let library = device.new_library_with_source(MSL_SRC, &compile_options)?;
        let func = library.get_function("crossbar_forward", None)?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func)?;
        let func_simd = library.get_function("crossbar_forward_simd", None)?;
        let pipeline_simd = device.new_compute_pipeline_state_with_function(&func_simd)?;
        let func_simd_batch = library.get_function("crossbar_forward_simd_batch", None)?;
        let pipeline_simd_batch =
            device.new_compute_pipeline_state_with_function(&func_simd_batch)?;
        let func_tiled = library.get_function("crossbar_forward_tiled", None)?;
        let pipeline_tiled = device.new_compute_pipeline_state_with_function(&func_tiled)?;
        let queue = device.new_command_queue();
        Ok(CachedMetal {
            device,
            queue,
            pipeline,
            pipeline_simd,
            pipeline_simd_batch,
            pipeline_tiled,
        })
    }

    fn cached() -> Result<&'static CachedMetal, MetalForwardError> {
        let init = CACHE.get_or_init(build_cache);
        init.as_ref()
            .map_err(|e| MetalForwardError::Metal(e.clone()))
    }

    pub fn warm_cache() -> Result<(), MetalForwardError> {
        cached().map(|_| ())
    }

    fn ensure_pooled(
        device: &Device,
        n: usize,
        options: MTLResourceOptions,
        g: &mut Option<PooledBuffers>,
    ) -> Result<(), MetalForwardError> {
        let need_new = g.as_ref().map(|p| p.n != n).unwrap_or(true);
        if need_new {
            let cn = n * n * core::mem::size_of::<f32>();
            let inn = n * core::mem::size_of::<f32>();
            let outn = n * core::mem::size_of::<f32>();
            let conductance = device.new_buffer(cn as u64, options);
            let input = device.new_buffer(inn as u64, options);
            let input_anchor = device.new_buffer(inn as u64, options);
            let output = device.new_buffer(outn as u64, options);
            let n_uint = device.new_buffer(4, options);
            *g = Some(PooledBuffers {
                n,
                conductance,
                input,
                input_anchor,
                output,
                n_uint,
            });
        }
        Ok(())
    }

    fn threadgroup_dims_scalar(cache: &CachedMetal, n: usize) -> (MTLSize, MTLSize) {
        let max_tg = cache.pipeline.max_total_threads_per_threadgroup() as u64;
        let width = cache.pipeline.thread_execution_width().max(1);
        let mut tpg = max_tg.min(1024);
        tpg = (tpg / width) * width;
        if tpg == 0 {
            tpg = width.min(max_tg);
        }
        let threads_per_group = MTLSize::new(tpg, 1, 1);
        let groups_x = (n as metal::NSUInteger + tpg - 1) / tpg;
        let thread_groups = MTLSize::new(groups_x, 1, 1);
        (threads_per_group, thread_groups)
    }

    fn threadgroup_dims_simd(n: usize) -> (MTLSize, MTLSize) {
        (
            MTLSize::new(SIMDGROUP_WIDTH, 1, 1),
            MTLSize::new(n as metal::NSUInteger, 1, 1),
        )
    }

    fn threadgroup_dims_simd_batch(n: usize) -> (MTLSize, MTLSize) {
        let threads = SIMD_BATCH_ROWS * SIMDGROUP_WIDTH;
        let groups = (n as u64 + SIMD_BATCH_ROWS - 1) / SIMD_BATCH_ROWS;
        (MTLSize::new(threads, 1, 1), MTLSize::new(groups, 1, 1))
    }

    fn encode_matvec_pass(
        enc: &ComputeCommandEncoderRef,
        cache: &CachedMetal,
        p: &PooledBuffers,
        n: usize,
        cascade_pass: bool,
    ) {
        enc.set_buffer(0, Some(&p.conductance), 0);
        enc.set_buffer(1, Some(&p.input), 0);
        enc.set_buffer(2, Some(&p.output), 0);
        enc.set_buffer(3, Some(&p.n_uint), 0);
        if use_tiled_kernel(n, cascade_pass) {
            enc.set_compute_pipeline_state(&cache.pipeline_tiled);
            enc.set_threadgroup_memory_length(
                0,
                (TILED_KERNEL_TILE * core::mem::size_of::<f32>()) as u64,
            );
            let (threads_per_group, thread_groups) = threadgroup_dims_simd(n);
            enc.dispatch_thread_groups(thread_groups, threads_per_group);
        } else if use_simd_batch_kernel(n, cascade_pass) {
            enc.set_compute_pipeline_state(&cache.pipeline_simd_batch);
            let (threads_per_group, thread_groups) = threadgroup_dims_simd_batch(n);
            enc.dispatch_thread_groups(thread_groups, threads_per_group);
        } else if use_simd_kernel(n, cascade_pass) {
            enc.set_compute_pipeline_state(&cache.pipeline_simd);
            let (threads_per_group, thread_groups) = threadgroup_dims_simd(n);
            enc.dispatch_thread_groups(thread_groups, threads_per_group);
        } else {
            enc.set_compute_pipeline_state(&cache.pipeline);
            let (threads_per_group, thread_groups) = threadgroup_dims_scalar(cache, n);
            enc.dispatch_thread_groups(thread_groups, threads_per_group);
        }
    }

    fn upload_conductance_only(p: &PooledBuffers, n: usize, conductance: &[f32]) {
        let cn = n * n * core::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                conductance.as_ptr(),
                p.conductance.contents() as *mut f32,
                n * n,
            );
        }
        p.conductance
            .did_modify_range(NSRange::new(0, cn as metal::NSUInteger));
    }

    fn upload_input_only(p: &PooledBuffers, n: usize, input_host: &[f32]) {
        let inn = n * core::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(input_host.as_ptr(), p.input.contents() as *mut f32, n);
            std::ptr::write(p.n_uint.contents() as *mut u32, n as u32);
        }
        p.input
            .did_modify_range(NSRange::new(0, inn as metal::NSUInteger));
        p.n_uint.did_modify_range(NSRange::new(
            0,
            core::mem::size_of::<u32>() as metal::NSUInteger,
        ));
    }

    fn upload_input_anchor_only(p: &PooledBuffers, n: usize, input_host: &[f32]) {
        let inn = n * core::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_host.as_ptr(),
                p.input_anchor.contents() as *mut f32,
                n,
            );
        }
        p.input_anchor
            .did_modify_range(NSRange::new(0, inn as metal::NSUInteger));
    }

    fn blit_input_anchor_to_input(cmd_buf: &CommandBufferRef, p: &PooledBuffers, n: usize) {
        let row_bytes = (n * core::mem::size_of::<f32>()) as metal::NSUInteger;
        let blit = cmd_buf.new_blit_command_encoder();
        blit.copy_from_buffer(&p.input_anchor, 0, &p.input, 0, row_bytes);
        blit.end_encoding();
    }

    fn commit_wait(cmd_buf: &CommandBufferRef) -> Result<(), MetalForwardError> {
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        if cmd_buf.status() != MTLCommandBufferStatus::Completed {
            return Err(MetalForwardError::Metal(format!(
                "command buffer status {:?}",
                cmd_buf.status()
            )));
        }
        Ok(())
    }

    fn copy_output_to_host(p: &PooledBuffers, n: usize, output: &mut [f32]) {
        let ptr = p.output.contents() as *const f32;
        let out_sl = unsafe { slice::from_raw_parts(ptr, n) };
        output.copy_from_slice(out_sl);
    }

    fn encode_repeated_matvec_passes(
        cmd_buf: &CommandBufferRef,
        cache: &CachedMetal,
        p: &PooledBuffers,
        n: usize,
        repeats: usize,
        cascade_pass: bool,
    ) {
        for _ in 0..repeats {
            let enc = cmd_buf.new_compute_command_encoder();
            encode_matvec_pass(enc, cache, p, n, cascade_pass);
            enc.end_encoding();
        }
    }

    fn encode_cascade_matvec_passes(
        cmd_buf: &CommandBufferRef,
        cache: &CachedMetal,
        p: &PooledBuffers,
        n: usize,
        depth: usize,
    ) {
        let row_bytes = (n * core::mem::size_of::<f32>()) as metal::NSUInteger;
        for pass in 0..depth {
            let enc = cmd_buf.new_compute_command_encoder();
            encode_matvec_pass(enc, cache, p, n, true);
            enc.end_encoding();
            if pass + 1 < depth {
                let blit = cmd_buf.new_blit_command_encoder();
                blit.copy_from_buffer(&p.output, 0, &p.input, 0, row_bytes);
                blit.end_encoding();
            }
        }
    }

    /// Owns pooled shared `MTLBuffer`s for repeated `forward` calls. **Not thread-safe** — use one runner
    /// per thread, or call [`crate::crossbar_forward_metal`] (thread-local runner per OS thread).
    pub struct MetalRunner {
        buffers: Option<PooledBuffers>,
        /// Letzter auf die GPU geschriebener [`memristor::memristor::crossbar::Crossbar::conductance_epoch`].
        uploaded_g_epoch: u64,
        uploaded_n: usize,
    }

    const INVALID_G_EPOCH: u64 = u64::MAX;

    impl MetalRunner {
        /// Ensures the global pipeline cache is initialized (same as [`crate::warm_metal_pipeline`]).
        pub fn new() -> Result<Self, MetalForwardError> {
            cached().map(|_| Self {
                buffers: None,
                uploaded_g_epoch: INVALID_G_EPOCH,
                uploaded_n: 0,
            })
        }

        /// Erzwingt den nächsten Aufruf, die volle Leitwert‑Matrix **`G`** auf die GPU zu kopieren.
        pub fn invalidate_conductance_on_gpu(&mut self) {
            self.uploaded_g_epoch = INVALID_G_EPOCH;
        }

        fn needs_g_upload(&self, n: usize, g_epoch: u64) -> bool {
            self.uploaded_g_epoch == INVALID_G_EPOCH
                || self.uploaded_n != n
                || self.uploaded_g_epoch != g_epoch
        }

        fn mark_g_uploaded(&mut self, n: usize, g_epoch: u64) {
            self.uploaded_n = n;
            self.uploaded_g_epoch = g_epoch;
        }

        #[cfg(test)]
        pub(crate) fn uploaded_g_epoch_for_test(&self) -> u64 {
            self.uploaded_g_epoch
        }

        /// `y[i] = sum_j input[j] * G_ij` mit **`G_ij = 1/R_ij`** (`conductance`‑Slice ist die Leitwert‑Matrix).
        ///
        /// Puffer‑Pool gilt unverändert für festes `n`. Für Hot‑Loops ohne pro‑Aufruf `Vec`: [`Self::forward_into`].
        pub fn forward(
            &mut self,
            n: usize,
            conductance: &[f32],
            g_epoch: u64,
            input: &[f32],
        ) -> Result<Vec<f32>, MetalForwardError> {
            let mut out = vec![0.0_f32; n];
            self.forward_into(n, conductance, g_epoch, input, &mut out)?;
            Ok(out)
        }

        /// Wie [`Self::forward`], schreibt in `output` (**Länge `n`**) ohne neue `Vec`‑Allocation.
        ///
        /// `g_epoch` = [`memristor::memristor::crossbar::Crossbar::conductance_epoch`]: unverändertes **`G`**
        /// zwischen Aufrufen ⇒ **kein** erneutes `n²`‑Memcpy der Leitwerte (nur `input`).
        pub fn forward_into(
            &mut self,
            n: usize,
            conductance: &[f32],
            g_epoch: u64,
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), MetalForwardError> {
            self.forward_repeated_into(n, conductance, g_epoch, input, 1, output)
        }

        /// `repeats` mal dasselbe \(y = Gx\) auf der GPU (**ein** Command Buffer, **ein** Sync).
        ///
        /// Nützlich für Warm‑Loops/Benchmarks statt `repeats`× [`Self::forward_into`]. `repeats == 0`:
        /// kein GPU‑Pass (nur Validierung von `output`‑Länge).
        pub fn forward_repeated_into(
            &mut self,
            n: usize,
            conductance: &[f32],
            g_epoch: u64,
            input: &[f32],
            repeats: usize,
            output: &mut [f32],
        ) -> Result<(), MetalForwardError> {
            if repeats == 0 {
                if output.len() != n {
                    return Err(MetalForwardError::Metal(format!(
                        "forward_repeated_into: expected output len {}, got {}",
                        n,
                        output.len()
                    )));
                }
                return Ok(());
            }
            if conductance.len() != n * n || input.len() != n {
                return Err(MetalForwardError::InvalidShape {
                    n,
                    conductance_len: conductance.len(),
                    input_len: input.len(),
                });
            }
            if output.len() != n {
                return Err(MetalForwardError::Metal(format!(
                    "forward_repeated_into: expected output len {}, got {}",
                    n,
                    output.len()
                )));
            }

            let cache = cached()?;
            let options = MTLResourceOptions::CPUCacheModeDefaultCache
                | MTLResourceOptions::StorageModeShared;

            ensure_pooled(&cache.device, n, options, &mut self.buffers)?;
            if self.uploaded_n != n {
                self.uploaded_g_epoch = INVALID_G_EPOCH;
            }

            let upload_g = self.needs_g_upload(n, g_epoch);

            {
                let p = self
                    .buffers
                    .as_mut()
                    .expect("pooled buffers must exist after ensure_pooled");

                if upload_g {
                    upload_conductance_only(p, n, conductance);
                }
                upload_input_only(p, n, input);

                let cmd_buf = cache.queue.new_command_buffer();
                encode_repeated_matvec_passes(cmd_buf, cache, p, n, repeats, false);
                commit_wait(cmd_buf)?;
                copy_output_to_host(p, n, output);
            }
            if upload_g {
                self.mark_g_uploaded(n, g_epoch);
            }
            Ok(())
        }

        /// Mehrfaches \(y\leftarrow Gy\) (gleiche **`G`**, festes **`n`**) auf der GPU: **ein** Command Buffer
        /// mit Ping‑Pong **MTLBlit** (`output`→`input`) zwischen den Dispatches – spart Overhead gegenüber
        /// einer Schleife aus [`Self::forward_into`].
        ///
        /// `depth == 0`: schreibt **`input`** nach `output` (Host‑Kopie, kein GPU‑Pass).
        pub fn forward_cascade_into(
            &mut self,
            n: usize,
            conductance: &[f32],
            g_epoch: u64,
            input: &[f32],
            depth: usize,
            output: &mut [f32],
        ) -> Result<(), MetalForwardError> {
            if depth == 0 {
                if input.len() != n || output.len() != n {
                    return Err(MetalForwardError::Metal(format!(
                        "forward_cascade_into depth 0: expected len n={n}"
                    )));
                }
                output.copy_from_slice(input);
                return Ok(());
            }
            if conductance.len() != n * n || input.len() != n || output.len() != n {
                return Err(MetalForwardError::InvalidShape {
                    n,
                    conductance_len: conductance.len(),
                    input_len: input.len(),
                });
            }

            let cache = cached()?;
            let options = MTLResourceOptions::CPUCacheModeDefaultCache
                | MTLResourceOptions::StorageModeShared;
            ensure_pooled(&cache.device, n, options, &mut self.buffers)?;
            if self.uploaded_n != n {
                self.uploaded_g_epoch = INVALID_G_EPOCH;
            }

            let upload_g = self.needs_g_upload(n, g_epoch);

            let cmd_buf = cache.queue.new_command_buffer();

            {
                let p = self
                    .buffers
                    .as_mut()
                    .expect("pooled buffers must exist after ensure_pooled");

                if upload_g {
                    upload_conductance_only(p, n, conductance);
                }
                upload_input_only(p, n, input);

                encode_cascade_matvec_passes(cmd_buf, cache, p, n, depth);
            }
            if upload_g {
                self.mark_g_uploaded(n, g_epoch);
            }
            commit_wait(cmd_buf)?;

            let ptr = self.buffers.as_ref().expect("pooled").output.contents() as *const f32;
            let out_sl = unsafe { slice::from_raw_parts(ptr, n) };
            output.copy_from_slice(out_sl);
            Ok(())
        }

        /// Wie [`Self::forward_cascade_into`], aber `repeats` unabhängige Cascade‑Läufe in **einem** Sync.
        pub fn forward_cascade_repeated_into(
            &mut self,
            n: usize,
            conductance: &[f32],
            g_epoch: u64,
            input: &[f32],
            depth: usize,
            repeats: usize,
            output: &mut [f32],
        ) -> Result<(), MetalForwardError> {
            if repeats == 0 {
                if output.len() != n {
                    return Err(MetalForwardError::Metal(format!(
                        "forward_cascade_repeated_into: expected output len {n}"
                    )));
                }
                return Ok(());
            }
            if depth == 0 {
                return self.forward_repeated_into(n, conductance, g_epoch, input, repeats, output);
            }
            if conductance.len() != n * n || input.len() != n || output.len() != n {
                return Err(MetalForwardError::InvalidShape {
                    n,
                    conductance_len: conductance.len(),
                    input_len: input.len(),
                });
            }

            let cache = cached()?;
            let options = MTLResourceOptions::CPUCacheModeDefaultCache
                | MTLResourceOptions::StorageModeShared;
            ensure_pooled(&cache.device, n, options, &mut self.buffers)?;
            if self.uploaded_n != n {
                self.uploaded_g_epoch = INVALID_G_EPOCH;
            }

            let upload_g = self.needs_g_upload(n, g_epoch);
            let cmd_buf = cache.queue.new_command_buffer();

            {
                let p = self
                    .buffers
                    .as_mut()
                    .expect("pooled buffers must exist after ensure_pooled");

                if upload_g {
                    upload_conductance_only(p, n, conductance);
                }
                upload_input_only(p, n, input);
                upload_input_anchor_only(p, n, input);

                for rep in 0..repeats {
                    if rep > 0 {
                        blit_input_anchor_to_input(cmd_buf, p, n);
                    }
                    encode_cascade_matvec_passes(cmd_buf, cache, p, n, depth);
                }
            }
            if upload_g {
                self.mark_g_uploaded(n, g_epoch);
            }
            commit_wait(cmd_buf)?;
            copy_output_to_host(self.buffers.as_ref().expect("pooled"), n, output);
            Ok(())
        }

        /// Wie [`Self::forward_cascade_into`], liefert `Vec<f32>`.
        pub fn forward_cascade(
            &mut self,
            n: usize,
            conductance: &[f32],
            g_epoch: u64,
            input: &[f32],
            depth: usize,
        ) -> Result<Vec<f32>, MetalForwardError> {
            if n == 0 {
                if input.is_empty() && conductance.is_empty() {
                    return Ok(vec![]);
                }
                return Err(MetalForwardError::InvalidShape {
                    n: 0,
                    conductance_len: conductance.len(),
                    input_len: input.len(),
                });
            }
            let mut out = vec![0.0_f32; n];
            self.forward_cascade_into(n, conductance, g_epoch, input, depth, &mut out)?;
            Ok(out)
        }
    }

    thread_local! {
        static DEFAULT_RUNNER: RefCell<Option<MetalRunner>> = RefCell::new(None);
    }

    pub fn crossbar_forward_metal_impl(
        n: usize,
        conductance: &[f32],
        input: &[f32],
    ) -> Result<Vec<f32>, MetalForwardError> {
        DEFAULT_RUNNER.with(|cell| -> Result<Vec<f32>, MetalForwardError> {
            let mut g = cell.borrow_mut();
            if g.is_none() {
                *g = Some(MetalRunner::new()?);
            }
            g.as_mut()
                .expect("runner just set")
                .invalidate_conductance_on_gpu();
            g.as_mut()
                .expect("runner just set")
                .forward(n, conductance, 0, input)
        })
    }

    pub fn crossbar_forward_metal_cascade_impl(
        n: usize,
        conductance: &[f32],
        input: &[f32],
        depth: usize,
    ) -> Result<Vec<f32>, MetalForwardError> {
        DEFAULT_RUNNER.with(|cell| -> Result<Vec<f32>, MetalForwardError> {
            let mut g = cell.borrow_mut();
            if g.is_none() {
                *g = Some(MetalRunner::new()?);
            }
            g.as_mut()
                .expect("runner just set")
                .invalidate_conductance_on_gpu();
            g.as_mut()
                .expect("runner just set")
                .forward_cascade(n, conductance, 0, input, depth)
        })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::crossbar_forward_metal;
    use super::crossbar_forward_metal_cascade;
    use super::warm_metal_pipeline;
    use super::MetalRunner;
    use memristor::memristor::crossbar::Crossbar;

    #[test]
    fn metal_matches_cpu_crossbar() {
        warm_metal_pipeline().expect("warm");
        let n = 64usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        let mut conductance = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let r = 1.0 + 0.001 * ((i * n + j) as f32);
                bar.set_resistance(i, j, r).unwrap();
                conductance[i * n + j] = bar.conductance_at(i, j).unwrap();
            }
        }
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let cpu = bar.forward(&input).unwrap();
        let gpu = crossbar_forward_metal(n, &conductance, &input).unwrap();
        assert_eq!(cpu.len(), gpu.len());
        for (a, b) in cpu.iter().zip(gpu.iter()) {
            assert!((a - b).abs() < 1e-4, "cpu {a} gpu {b}");
        }
    }

    #[test]
    fn metal_runner_matches_tls_api() {
        warm_metal_pipeline().expect("warm");
        let n = 64usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        let mut conductance = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let r = 1.0 + 0.001 * ((i * n + j) as f32);
                bar.set_resistance(i, j, r).unwrap();
                conductance[i * n + j] = bar.conductance_at(i, j).unwrap();
            }
        }
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let epoch = bar.conductance_epoch();
        let via_tls = crossbar_forward_metal(n, &conductance, &input).unwrap();
        let mut runner = MetalRunner::new().expect("runner");
        let via_runner = runner.forward(n, &conductance, epoch, &input).unwrap();
        for (a, b) in via_tls.iter().zip(via_runner.iter()) {
            assert!((a - b).abs() < 1e-6, "tls {a} runner {b}");
        }
    }

    #[test]
    fn metal_forward_into_matches_forward() {
        warm_metal_pipeline().expect("warm");
        let n = 32usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        let mut g = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                bar.set_resistance(i, j, 1.0 + 0.01 * ((i * n + j) as f32))
                    .unwrap();
                g[i * n + j] = bar.conductance_at(i, j).unwrap();
            }
        }
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005).collect();
        let mut runner = MetalRunner::new().expect("runner");
        let mut buf = vec![0.0_f32; n];
        let epoch = bar.conductance_epoch();
        let v = runner.forward(n, &g, epoch, &input).unwrap();
        runner.forward_into(n, &g, epoch, &input, &mut buf).unwrap();
        for (a, b) in v.iter().zip(buf.iter()) {
            assert!((a - b).abs() < 1e-5, "fwd {a} into {b}");
        }
    }

    #[test]
    fn metal_cascade_depth_two_matches_cpu_double_forward() {
        warm_metal_pipeline().expect("warm");
        let n = 48usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        let mut g = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                bar.set_resistance(i, j, 0.92 + 0.0009 * ((i * n + j) as f32))
                    .unwrap();
                g[i * n + j] = bar.conductance_at(i, j).unwrap();
            }
        }
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.003 - 0.11).collect();
        let y1 = bar.forward(&input).unwrap();
        let y2_cpu = bar.forward(&y1).unwrap();
        let mut runner = MetalRunner::new().expect("runner");
        let epoch = bar.conductance_epoch();
        let y2_tls = crossbar_forward_metal_cascade(n, &g, &input, 2).unwrap();
        let mut y2_runner = vec![0.0_f32; n];
        runner
            .forward_cascade_into(n, &g, epoch, &input, 2, &mut y2_runner)
            .unwrap();
        for i in 0..n {
            assert!(
                (y2_cpu[i] - y2_tls[i]).abs() < 2e-2,
                "i={} cpu {} tls {}",
                i,
                y2_cpu[i],
                y2_tls[i]
            );
            assert!(
                (y2_cpu[i] - y2_runner[i]).abs() < 2e-2,
                "i={} cpu {} runner {}",
                i,
                y2_cpu[i],
                y2_runner[i]
            );
        }
    }

    #[test]
    fn metal_skips_g_upload_when_epoch_unchanged() {
        warm_metal_pipeline().expect("warm");
        let n = 16usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        for i in 0..n {
            for j in 0..n {
                bar.set_resistance(i, j, 1.0 + 0.01 * ((i * n + j) as f32))
                    .unwrap();
            }
        }
        let g = bar.conductance_matrix();
        let epoch = bar.conductance_epoch();
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mut runner = MetalRunner::new().expect("runner");
        let mut out = vec![0.0_f32; n];
        runner.forward_into(n, g, epoch, &input, &mut out).unwrap();
        assert_eq!(runner.uploaded_g_epoch_for_test(), epoch);
        runner.forward_into(n, g, epoch, &input, &mut out).unwrap();
        assert_eq!(runner.uploaded_g_epoch_for_test(), epoch);
    }

    #[test]
    fn metal_simd_kernel_matches_cpu_at_large_n() {
        warm_metal_pipeline().expect("warm");
        let n = 768usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        let mut conductance = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let r = 0.95 + 0.0005 * ((i * n + j) as f32);
                bar.set_resistance(i, j, r).unwrap();
                conductance[i * n + j] = bar.conductance_at(i, j).unwrap();
            }
        }
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0007 - 0.2).collect();
        let cpu = bar.forward(&input).unwrap();
        let gpu = crossbar_forward_metal(n, &conductance, &input).unwrap();
        for (a, b) in cpu.iter().zip(gpu.iter()) {
            assert!((a - b).abs() < 2e-3, "cpu {a} gpu {b}");
        }
    }

    #[test]
    fn metal_simd_batch_matches_cpu_at_2048() {
        warm_metal_pipeline().expect("warm");
        let n = 2048usize;
        let conductance: Vec<f32> = (0..n * n)
            .map(|k| 1.0 / (1.02 + 0.0000001 * (k as f32)))
            .collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.00005 - 0.01).collect();
        let cpu: f32 = input
            .iter()
            .enumerate()
            .map(|(j, &x)| x * conductance[j])
            .sum();
        let gpu = crossbar_forward_metal(n, &conductance, &input).unwrap();
        assert!((cpu - gpu[0]).abs() < 2e-2, "row0 cpu={cpu} gpu={}", gpu[0]);
    }

    #[test]
    fn metal_kernel_tier_splits_forward_and_cascade() {
        assert_eq!(
            super::metal_kernel_tier(2048, false),
            super::MetalKernelTier::SimdgroupBatch
        );
        assert_eq!(
            super::metal_kernel_tier(2048, true),
            super::MetalKernelTier::Simdgroup
        );
        assert_eq!(
            super::metal_kernel_tier(4096, false),
            super::MetalKernelTier::SimdgroupBatch
        );
        assert_eq!(
            super::metal_kernel_tier(4096, true),
            super::MetalKernelTier::Tiled
        );
    }

    #[test]
    fn metal_tiled_kernel_matches_cpu_rows_at_cascade_min() {
        warm_metal_pipeline().expect("warm");
        let n = super::TILED_CASCADE_MIN_N;
        let conductance: Vec<f32> = (0..n * n)
            .map(|k| 1.0 / (1.05 + 0.00000005 * (k as f32)))
            .collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.00001 - 0.02).collect();
        let gpu = crossbar_forward_metal(n, &conductance, &input).unwrap();
        let sample_rows = [0usize, 31, 512, 1024, n - 1];
        for row in sample_rows {
            let base = row * n;
            let cpu: f32 = input
                .iter()
                .enumerate()
                .map(|(j, &x)| x * conductance[base + j])
                .sum();
            assert!(
                (cpu - gpu[row]).abs() < 3e-2,
                "row={row} cpu={cpu} gpu={}",
                gpu[row]
            );
        }
    }

    #[test]
    fn metal_reuploads_g_after_resistance_change() {
        warm_metal_pipeline().expect("warm");
        let n = 32usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        for i in 0..n {
            for j in 0..n {
                bar.set_resistance(i, j, 1.0 + 0.01 * ((i * n + j) as f32))
                    .unwrap();
            }
        }
        let g = bar.conductance_matrix();
        let epoch0 = bar.conductance_epoch();
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let mut runner = MetalRunner::new().expect("runner");
        let mut out = vec![0.0_f32; n];
        runner.forward_into(n, g, epoch0, &input, &mut out).unwrap();
        assert_eq!(runner.uploaded_g_epoch_for_test(), epoch0);

        bar.set_resistance(0, 0, 3.5).unwrap();
        let epoch1 = bar.conductance_epoch();
        assert!(epoch1 > epoch0);
        let g1 = bar.conductance_matrix();
        let cpu = bar.forward(&input).unwrap();
        runner
            .forward_into(n, g1, epoch1, &input, &mut out)
            .unwrap();
        assert_eq!(runner.uploaded_g_epoch_for_test(), epoch1);
        for (a, b) in cpu.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4, "cpu {a} gpu {b}");
        }
    }

    #[test]
    fn metal_cascade_depth_zero_copies_input() {
        warm_metal_pipeline().expect("warm");
        let n = 9usize;
        let g = vec![1.0_f32; n * n];
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut runner = MetalRunner::new().expect("runner");
        let mut buf = vec![0.5_f32; n];
        runner
            .forward_cascade_into(n, &g, 1, &input, 0, &mut buf)
            .unwrap();
        assert_eq!(buf, input);
        let tls = crossbar_forward_metal_cascade(n, &g, &input, 0).unwrap();
        assert_eq!(tls, input);
    }

    #[test]
    fn metal_forward_repeated_matches_single_and_cpu() {
        warm_metal_pipeline().expect("warm");
        let n = 256usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        for i in 0..n {
            for j in 0..n {
                bar.set_resistance(i, j, 1.0 + 0.002 * ((i * n + j) as f32))
                    .unwrap();
            }
        }
        let g = bar.conductance_matrix();
        let epoch = bar.conductance_epoch();
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.003 - 0.1).collect();
        let cpu = bar.forward(&input).unwrap();
        let mut runner = MetalRunner::new().expect("runner");
        let mut once = vec![0.0_f32; n];
        let mut thrice = vec![0.0_f32; n];
        runner.forward_into(n, g, epoch, &input, &mut once).unwrap();
        runner
            .forward_repeated_into(n, g, epoch, &input, 3, &mut thrice)
            .unwrap();
        for (a, b) in cpu.iter().zip(once.iter()) {
            assert!((a - b).abs() < 1e-3, "single cpu {a} gpu {b}");
        }
        for (a, b) in cpu.iter().zip(thrice.iter()) {
            assert!((a - b).abs() < 1e-3, "repeated cpu {a} gpu {b}");
        }
    }

    #[test]
    fn metal_cascade_repeated_matches_cpu_at_256() {
        warm_metal_pipeline().expect("warm");
        let n = 256usize;
        let mut bar = Crossbar::new(n, 0.001, 0.0);
        for i in 0..n {
            for j in 0..n {
                bar.set_resistance(i, j, 1.0 + 0.003 * ((i * n + j) as f32))
                    .unwrap();
            }
        }
        let g = bar.conductance_matrix();
        let epoch = bar.conductance_epoch();
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002 - 0.2).collect();
        let cpu = bar.forward_cascade(&input, 2).unwrap();
        let mut runner = MetalRunner::new().expect("runner");
        let mut out = vec![0.0_f32; n];
        runner
            .forward_cascade_repeated_into(n, g, epoch, &input, 2, 3, &mut out)
            .unwrap();
        for (a, b) in cpu.iter().zip(out.iter()) {
            assert!((a - b).abs() < 5e-2, "cpu {a} gpu {b}");
        }
    }
}
