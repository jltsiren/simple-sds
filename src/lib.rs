//! Simple succinct data structures.
//!
//! This is a toy project with two goals: to learn Rust and to experiment with the API of basic succinct data structures.
//! The plan is to implement the subset of [SDSL](https://github.com/simongog/sdsl-lite) I am currently using and to extend it a bit.
//!
//! # Notes
//!
//! * This crate is designed for the x86_64 architecture with the BMI2 instruction set (Intel Haswell / AMD Excavator or later).
//! * Things may not work if the system is not little-endian or if `usize` is not 64-bit.
//! * Some operations may be slow without the POPCNT, LZCNT, TZCNT, and PDEP instructions.

pub mod bit_vector;
pub mod bits;
pub mod int_vector;
pub mod ops;
pub mod raw_vector;
pub mod serialize;
