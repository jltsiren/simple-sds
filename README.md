# Simple succinct data structures

This is a toy project with two goals: to learn Rust and to experiment with the API of basic succinct data structures. The plan is to implement the subset of SDSL I am currently using and to extend it a bit.

## Implemented functionality

### Integer vectors

* `RawVector`: A bit array that supports reading, writing, and appending 1-64 bits at a time. Implemented on top of `Vec<u64>`.
* `RawVectorWriter`: An append-only version of `RawVector` that writes the structure directly to a file.
* `IntVector`: A bit-packed vector of fixed-width integers implemented on top of `RawVector`. Like `sdsl::int_vector` but also supports stack functionality.
* `IntVectorWriter`: An append-only version of `IntVector` that writes the structure directly to a file. Like a subset of `sdsl::int_vector_buffer`.

## Planned functionality

### Integer vectors

* `TupleVector`: A bit-packed vector of tuples of unsigned integers, with a fixed width for each field in the tuple. Implemented on top of `RawVector`.
* Memory-mapped versions of all vectors.

### Bitvectors

* `BitVector`: A plain immutable bitvector supporting `rank()`, `select()`, and `select_0()`. Implemented on top of `RawVector`. The support structures for all three queries are optional.
* `SparseVector`: An Elias-Fano encoded immutable bitvector supporting `rank()` and `select()`. Comparable to `sdsl::sd_vector`.
* `v.rank(i)`: The number of ones in `v[0..i]`.
* Select support implies support for predecessor/successor queries:
  * `v.select(i)`: Value `a[i]`. Note that `select()` is 0-based, unlike in SDSL.
  * `v.predecessor(i)`: Largest `a[j] <= i`.
  * `v.successor(i)`: Smallest `a[j] >= i`.
  * These queries assume that the vector encodes a sorted integer array `a`. They return iterators over pairs `(i, a[i])`.

## Notes

* The included `.cargo/config.toml` sets the target CPU to `native`.
* This crate assumes that `usize` is a 64-bit integer.
