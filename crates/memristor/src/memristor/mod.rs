pub mod cell;
pub mod crossbar;

#[cfg(target_arch = "aarch64")]
pub(crate) mod forward_aarch64;
