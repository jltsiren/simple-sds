# Simple succinct data structures

These structures are comparable to those in [SDSL](https://github.com/simongog/sdsl-lite) in performance and scalability.
As the focus is on (relative) simplicity, ugly low-level optimizations are generally avoided.

## Implemented functionality

### Integer vectors

* `RawVector`: A bit array that supports reading, writing, and appending 1-64 bits at a time. Implemented on top of `Vec<u64>`.
  * `RawVectorWriter`: An append-only version of `RawVector` that writes the structure directly to a file.
  * `RawVectorMapper`: An immutable memory-mapped `RawVector`.
* `IntVector`: A bit-packed vector of fixed-width integers implemented on top of `RawVector`. Like `sdsl::int_vector` but also supports stack functionality.
  * `IntVectorWriter`: An append-only version of `IntVector` that writes the structure directly to a file. Like a subset of `sdsl::int_vector_buffer`.
  * `IntVectorMapper`: An immutable memory-mapped `IntVector`.
* `WaveletMatrix`: An immutable vector of fixed-width integers. Similar to `sdsl::wm_int`.
  * Supports `rank()`, `inverse_select()`, `select()`, `predecessor()`, and `successor()` with each item value.
  * Iterators over all items and over items with a specified value.
  * Implemented using a `BitVector` for each level.

### Bitvectors

* `BitVector`: A plain immutable bitvector.
  * Supports `rank()`, `rank_zero()`, `select()`, `select_zero()`, `predecessor()`, and `successor()` queries using optional support structures.
  * Iterators over set bits, unset bits, and all bits.
  * Implemented on top of `RawVector`.
* `RLVector`: A run-length encoded bitvector.
  * Supports `rank()`, `rank_zero()`, `select()`, `select_zero()`, `predecessor()`, and `successor()` queries.
  * Iterators over set bits and all bits.
  * Space-efficient construction with `RLBuilder`.
* `SparseVector`: An Elias-Fano encoded bitvector.
  * Supports `rank()`, `rank_zero()`, `select()`, `select_zero()`, `predecessor()`, and `successor()` queries .
  * Iterators over set bits and all bits.
  * Space-efficient construction with `SparseBuilder`.

## Planned functionality

### Integer vectors

* Mutable memory-mapped vectors.

### Bitvectors

* Versions of `predecessor()` and `successor()` that return values instead of iterators?
* Slice-like functionality based on iterators?

## Notes

* The included `.cargo/config.toml` sets the target CPU to `native`.
* This crate is designed for the x86_64 architecture with the BMI2 instruction set (Intel Haswell / AMD Excavator or later). Some operations may be slow without the POPCNT, LZCNT, TZCNT, and PDEP instructions.
* 64-bit ARM is also supported.
* Unix-like operating system is required for `mmap()`.
* Things may not work if the system is not little-endian or if `usize` is not 64-bit.
