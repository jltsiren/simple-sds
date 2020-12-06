//! Simple succinct data structures.
//!
//! This is a toy project with two goals: to learn Rust and to experiment with the API of basic succinct data structures.
//! The plan is to implement the subset of SDSL I am currently using and to extend it a bit.
//!
//! # Dependencies
//!
//! * [rand](https://crates.io/crates/rand) v0.7.3
//!
//! # Notes
//!
//! * This crate should be compiled for a CPU target with a native 64-bit popcnt instruction.
//! * Things may not work if `usize` is not 64 bits.
//! * The system is assumed to be little-endian.

pub mod bits;
// pub mod int_vector;
pub mod ops;
pub mod raw_vector;
pub mod serialize;
