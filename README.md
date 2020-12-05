# Simple succinct data structures

This is a toy project with two goals: to learn Rust and to experiment with the API of basic succinct data structures. The plan is to implement the subset of SDSL I am currently using and to extend it a bit.

## Planned functionality

### Vectors

* `RawVector`: A binary vector that supports reading, writing, and appending 1-64 bits at a time. Implemented on top of `Vec<u64>`.
* `MmapVector`: A memory-mapped fixed-size version of `RawVector` compatible with its serialization format. Comparable to `sdsl::int_vector_mapper` in SDSL 3.
* `VectorBuffer`: An append-only version of `RawVector` that writes its contents to a file. Comparable to a subset of `sdsl::int_vector_buffer` functionality.
* `IntVector`: A packed vector of unsigned fixed-width integers. Implemented on top of `RawVector`. Comparable to `sdsl::int_vector`.
* `TupleVector`: A packed vector of tuples of unsigned integers, with a fixed width for each field in the tuple. Implemented on top of `RawVector`.

### Bitvectors

* `BitVector`: A plain immutable bitvector supporting `rank()`, `select()`, and `select_0()`. Implemented on top of `RawVector`. The support structures for all three queries are optional. May also implement `predecessor()` and `successor()`.
* `SparseVector`: An Elias-Fano encoded immutable bitvector supporting `rank()`, `select()`, `predecessor()`, and `successor()`. Comparable to `sdsl::sd_vector`.
* `v.rank(i)`: The number of ones in `v[0..i]`.
* The other queries assume that the vector encodes a sorted integer array `a`.
  * `v.select(i)`: Value `a[i]`. Note that `select()` is 0-based, unlike in SDSL.
  * `v.predecessor(i)`: Largest `a[j] <= i`.
  * `v.successor(i)`: Smallest `a[j] >= i`.
* These queries return iterators over pairs `(i, a[i])`.
  * It may be convenient to return a final `(count_ones(), len())` guard pair.
  * `predecessor()` may return an empty value or a special guard value if there is no predecessor.

## Notes

* The included `.cargo/config.toml` sets the target CPU to `native`.
* This probably assumes that `usize` is a 64-bit integer.
