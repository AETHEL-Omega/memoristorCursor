//! aethel MLX runtime integration surface (Memristor v1 scaffold).

#[cfg(feature = "memristor-kv")]
pub mod kv_cache;
#[cfg(feature = "memristor-routing")]
pub mod route_engine;
#[cfg(feature = "memristor-routing")]
pub mod inference_executor;

#[cfg(feature = "memristor-kv")]
pub use kv_cache::{
    CompressKvError, DecompressKvError, MemristorKvCacheManager, SessionId,
};
#[cfg(feature = "memristor-routing")]
pub use inference_executor::{ExecuteError, InferenceExecutor};
#[cfg(feature = "memristor-routing")]
pub use route_engine::{
    MemoryGovernor, RouteDecision, RouteEngine, RuntimeState, TaskReq, TaskType,
};
