# Simple succinct data structures

This is a toy project with two goals: to learn Rust and to experiment with the API of basic succinct data structures. The plan is to implement the subset of [SDSL](https://github.com/simongog/sdsl-lite) I am currently using and to extend it a bit.

## Implemented functionality

### Integer vectors

* `RawVector`: A bit array that supports reading, writing, and appending 1-64 bits at a time. Implemented on top of `Vec<u64>`.
* `RawVectorWriter`: An append-only version of `RawVector` that writes the structure directly to a file.
* `IntVector`: A bit-packed vector of fixed-width integers implemented on top of `RawVector`. Like `sdsl::int_vector` but also supports stack functionality.
* `IntVectorWriter`: An append-only version of `IntVector` that writes the structure directly to a file. Like a subset of `sdsl::int_vector_buffer`.

### Bitvectors

* `BitVector`: A plain immutable bitvector.
  * Supports `rank()`, `rank_zero()`, `select()`, `select_zero()`, `predecessor()`, and `successor()` queries using optional support structures.
  * Iterators over set bits, unset bits, and all bits.
  * Implemented on top of `RawVector`.

## Planned functionality

### Integer vectors

* `TupleVector`: A bit-packed vector of tuples of unsigned integers, with a fixed width for each field in the tuple. Implemented on top of `RawVector`.
* Memory-mapped versions of all vectors.

### Bitvectors

* `SparseVector`: An Elias-Fano encoded immutable bitvector. Comparable to `sdsl::sd_vector`.

## Notes

* The included `.cargo/config.toml` sets the target CPU to `native`.
* This crate is designed for the x86_64 architecture with the BMI2 instruction set (Intel Haswell / AMD Excavator or later).
* Things may not work if the system is not little-endian or if `usize` is not 64-bit.
* Some operations may be slow without the POPCNT, LZCNT, TZCNT, and PDEP instructions.
