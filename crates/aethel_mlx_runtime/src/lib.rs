//! aethel MLX runtime integration surface (Memristor v1 scaffold).

#[cfg(feature = "memristor-routing")]
pub mod coreml_backend;
#[cfg(feature = "memristor-routing")]
pub mod inference_executor;
#[cfg(feature = "memristor-kv")]
pub mod kv_cache;
#[cfg(feature = "memristor-routing")]
pub mod m5_compute;
#[cfg(feature = "memristor-routing")]
pub mod mlx_backend;
#[cfg(feature = "memristor-routing")]
pub mod route_engine;

#[cfg(feature = "memristor-routing")]
pub use coreml_backend::{
    clear_coreml_handler, execute_coreml, has_coreml_handler, set_coreml_handler,
};
#[cfg(feature = "memristor-routing")]
pub use inference_executor::{ExecuteError, ExecuteOutcome, InferenceExecutor};
#[cfg(feature = "memristor-kv")]
pub use kv_cache::{CompressKvError, DecompressKvError, MemristorKvCacheManager, SessionId};
#[cfg(feature = "memristor-routing")]
pub use m5_compute::{
    choose_matvec_device, MatvecDevice, METAL_CASCADE_MIN_DEPTH, METAL_CASCADE_MIN_N,
    METAL_FORWARD_MIN_N, SIMD_KERNEL_MIN_N, TILED_CASCADE_MIN_N, TILED_KERNEL_MIN_N,
    TILED_KERNEL_TILE,
};
#[cfg(all(feature = "memristor-metal", target_os = "macos"))]
pub use memristor_metal::MetalKernelTier;
#[cfg(feature = "memristor-routing")]
pub use mlx_backend::{clear_mlx_handler, execute_mlx, has_mlx_handler, set_mlx_handler};
#[cfg(feature = "memristor-routing")]
pub use route_engine::{
    MemoryGovernor, RouteDecision, RouteEngine, RuntimeState, TaskReq, TaskType,
};
