//! # Simple succinct data structures
//!
//! These structures are comparable to those in [SDSL](https://github.com/simongog/sdsl-lite) in performance and scalability.
//! As the focus is on (relative) simplicity, ugly low-level optimizations are generally avoided.
//!
//! # Notes
//!
//! * This crate is designed for the x86_64 architecture with the BMI2 instruction set (Intel Haswell / AMD Excavator or later).
//! Some operations may be slow without the POPCNT, LZCNT, TZCNT, and PDEP instructions.
//! * 64-bit ARM is also supported.
//! * Unix-like operating system is required for `mmap()`.
//! * Things may not work if the system is not little-endian or if `usize` is not 64-bit.

pub mod bit_vector;
pub mod bits;
pub mod int_vector;
pub mod ops;
pub mod raw_vector;
pub mod rl_vector;
pub mod serialize;
pub mod sparse_vector;
pub mod support;
pub mod wavelet_matrix;

#[cfg(any(test, feature = "binaries"))]
#[doc(hidden)]
pub mod internal;
