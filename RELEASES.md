# Simple-SDS releases

## Current version

### Bitvectors

* `RLVector`: Run-length encoded bitvector similar to the one in RLCSA.
* Consistent conversions between bitvector types:
  * `From` trait between any two bitvector types.
  * Associated function `copy_bit_vec` for copying from a type that implements `Select`.

### Wavelet matrices

* Trait `VectorIndex` for rank/select-type queries over integer vectors.
* Implementations of `VectorIndex`:
  * `WaveletMatrix`: Plain balanced wavelet matrix.

### Other

* Better `Access` trait.
  * The default implementation is immutable.
  * The trait includes an iterator over the values.
  * `AccessIter` is a trivial implementation of the iterator using `Access::get`.
  * `Access::get_or` returns a default value if the index is not valid.

## Simple-SDS 0.3.1 (2022-02-17)

* Minor patch release for the GBZ paper.

## Simple-SDS 0.3.0 (2021-11-17)

### Compressed multisets

* Elias-Fano encoding can store duplicate values, which simplifies some uses of `SparseVector`.
* Built with `SparseBuilder::multiset` or `SparseVector::try_from`.
* Most bitvector operations generalize from sets to multisets naturally, but `rank_zero` does not work.
  * If there are multiple occurrences of value `i`, the predecessor of `i` has a higher rank than its successor.

### Interface changes

* `select` and related queries return `Option` instead of `Result`.
* `Element` trait is now `Vector`.

### Other

* Vectors have items instead of elements to avoid confusion between vector elements and serialization elements.
* Uses `rand` 0.8 instead of 0.7.
* Serialization improvements:
  * `skip_option`, `absent_option`, and `absent_option_size` for dealing with optional structures.
  * `test` for running basic serialization tests for a new type.

## Simple-SDS 0.2.0 (2021-05-02)

This is no longer a pre-release, but things may still change without warning.

### Serialization

* Documentation on the serialization formats: https://github.com/jltsiren/simple-sds/blob/main/SERIALIZATION.md
* Trait `Serializable` that indicates that a fixed-size type can be serialized by copying the bytes as one or more `u64` elements.
  * A `Vec` of `Serializable` values can always be serialized.
* Serialization for `Vec` of `u8` values and `String`.
* Some sanity checks when loading serialized structures.

### Memory mapping

* `MemoryMap` that maps a file as an immutable array of `u64` elements.
* Trait `MemoryMapped` for structures that correspond to an interval in `MemoryMap`.
* Several `MemoryMapped` implementations:
  * `MappedSlice`: `Vec` of `Serializable` values.
  * `MappedBytes`: `Vec` of `u8` values.
  * `MappedStr`: `String`.
  * `MappedOption`:  Optional structure.
  * `RawVectorMapper`: `RawVector`.
  * `IntVectorMapper`: `IntVector`.

### Other

* 64-bit ARM is now supported.
* Removed the unnecessary `Writer` trait and simplified `RawVectorWriter` and `IntVectorWriter`.
* Smaller `SparseVector`, especially for high-density vectors:
  * Approximate the optimal number of buckets / width of the low parts better.
  * Do not create unnecessary buckets.
  * Removed a redundant field from the header.

## Simple-SDS 0.1.0 (2021-01-14)

This initial release implements bit-packed integer vectors, plain bitvectors, and sparse (Elias-Fano encoded) bitvectors as well as various bitvector iterators. The performance is generally comparable to SDSL. As this is a pre-release, anything can change without warning.
