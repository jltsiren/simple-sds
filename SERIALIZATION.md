# Serialization formats

For version 0.4.0. Updated 2022-08-18.

Changes since version 0.2.0 are mentioned in the relevant location.

## Basic structures

The simple-sds serialization format is intended as an interchange format for succinct data structures.
It should be easy to reimplement.
The format is also designed to make memory-mapped data structures straightforward.

A file is an array of **elements**, which are unsigned 64-bit little-endian integers.
As a result, the size of the file must be a multiple of 8 bytes.
All array indexes are 0-based.
The reader is always assumed to know the type of the object they are reading.

A fixed-length type is **serializable** if its size is a multiple of 8 bytes.
A serializable object can be serialized by copying the bytes.

### Vectors

A **vector** is an array of objects of the same type.
Each object in a vector is called an **item**.
The **length** of a vector is the number of items in it.

Serialization format for vectors of serializable items:

1. Length of the vector as an element.
2. Concatenated items from the vector.

Serialization format for vectors of **bytes**:

1. Length of the vector as an element.
2. Concatenated items from the vector.
3. 0 to 7 bytes of padding with byte value 0 to make the total size of the serialized vector a multiple of 8 bytes.

**Strings** are serialized as vectors of bytes using the UTF-8 encoding.

### Optional structures

An **optional** structure may be present in the file or absent from it.
If a structure is needed in some but not all applications of the parent structure, it can be made optional.
Optional structures may also be used for serializing implementation-dependent support structures that can be built from the parent structure.
The length of an optional structure is the number of elements required to serialize the actual structure (if present) or `0` (if absent).

Serialization format for optional structures:

1. Length of the optional structure as an element.
2. The structure, if present.

The reader can easily skip an optional structure, because its length is stored before the structure itself.
If the reader needs to pass through an optional structure without understanding the format, it can be loaded and serialized as a vector of elements.

**Note:** Making a structure optional implies that the content of the structure optional.
An empty structure may often be a better way of representing core data that is not always present.

## Core data structures

### Raw bitvector

A **raw bitvector** (`RawVector`) is a vector of bits.
The items of a raw bitvector are concatenated and stored in a vector of elements in little-endian order.
Bit `i` of the raw bitvector is stored as bit `i % 64` of element `floor(i / 64)`.
A raw bitvector of length `n` requires a vector of `floor((n + 63) / 64)` elements.
Any unused bits in the last element must be set to `0`.

Serialization format for raw bitvectors:

1. Length of the vector as an element.
2. Vector of elements storing the items.

### Integer vector

An **integer vector** (`IntVector`) is a bit-packed vector of integers.
The **width** of the items can be from 1 to 64 bits.
The items of an integer vector are concatenated and stored in a raw bitvector.
An integer vector of `n` items of width `w` bits requires a raw bitvector of of length `n * w`.

Serialization format for integer vectors:

1. Length of the vector as an element.
2. Width of the items as an element.
3. Raw bitvector storing the items.

### Bitvector

A plain **bitvector** (`BitVector`) is vector of bits that supports rank, select, and similar queries.
It extends the functionality of a raw bitvector.

Serialization format for bitvectors:

1. Number of set bits as an element.
2. Raw bitvector storing the items.
3. Optional rank support structure.
4. Optional select support structure for set bits.
5. Optional select support structure for unset bits.

The support structures are often both application-dependent and implementation-dependent and hence optional.

## Compressed bitvectors

### Sparse bitvector

A **sparse bitvector** (`SparseVector`) is an Elias-Fano encoded vector of bits that supports rank, select, and similar queries.
It can be interpreted as a set of integers or a vector of sorted integers, where the integers are the positions of the set bits.

Assume that the length of the vector of bits is `n` and that there are `m` set bits.
Each integer is split into a **low part** and a **high part**.
The low parts are the lowest `w ≈ log2(n) - log2(m)` bits of each integer, with `w >= 1`.
They are stored in an integer vector of length `m` and width `w`.

The high part of integer `x` is `x >> w`.
The integers are placed into buckets by the high part.
A bitvector encodes the number of integers in each bucket in unary.
For each bucket with `k >= 0` integers, in sorted order, the bitvector contains a sequence of `1`s of length `k` followed by `0`.
There must be a bucket for each position in the semiopen interval `0..n` but no additional buckets after them.

The `i`th item in the sorted vector of integers is `low[i] + ((high.select(i) - i) << w)`.

Serialization format for sparse bitvectors:

1. Length of the vector of bits as an element.
2. Bitvector storing the high parts.
3. Integer vector storing the low parts.

**Note:** The encoding also supports multisets / duplicate items, but the semantics are not fully clear yet.

### Run-length encoded bitvector (version 0.4.0)

A **run-length encoded bitvector** (`RLVector`) stores a vector of bits as a sequence of maximal runs of unset and set bits.

If there are `n0` unset bits followed by `n1` set bits, it is encoded as a pair of integers `(n0, n1 - 1)`.
Each integer is encoded in little-endian order using 4-bit code units.
The lowest 3 bits of each code unit contain data.
If the high bit is set, the encoding continues in the next unit.
We partition the encoding into 64-unit (32-byte) blocks that consist of entire runs.
If there is not enough space left for encoding the next `(n0, n1)`, we pad the block with `0` values and move to the next block.
If the final block is not full, it must not contain any padding.

For each block, we store a sample `(n1, n)`, where `n` is the number of bits and `n1` is the number of set bits encoded in all preceding blocks.
This can be interpreted as `(rank(n, 1), n)`.

Serialization format for run-length encoded bitvectors:

1. Length of the vector of bits as an element.
2. Number of set bits as an element.
3. Samples as an integer vector with the minimal width necessary.
4. Concatenated blocks as an integer vector of width 4.

## Wavelet matrices (version 0.4.0)

A **wavelet matrix** is an immutable integer vector that supports rank/select-like queries.
It is effectively the positional BWT of the binary sequences encoding the integers, operating on **levels** (rows) instead of columns.
If `value` is the largest item present in the vector, the **alphabet** of the vector is `0..=value`.

Bitvector `bv[level]` on level `level` represent bit values

> `1 << (width - 1 - level)`.

If `bv[level][i] == 0`, position `i` on level `level` it maps to position

> `bv[level].rank_zero(i)`

on level `level + 1`.
Otherwise it maps to position

> `bv[level].count_zeros() + bv[level].rank(i)`.

The value of the item at offset `i` can be determined by starting from level `0` offset `i`, proceeding down in the matrix, and calculating the sum of values corresponding to set bits.
This process **reorders** the items in the vector by sorting them according to their reverse binary representations.

### Wavelet matrix core

The **core** of a wavelet matrix (`WMCore`) consists of the bitvectors that handle the reordering.

Serialization format for the wavelet matrix core:

1. `width`: Width of the items as an element.
2. `levels`: A `BitVector` for each level in `0..width`.

### Plain wavelet matrix

A **plain** wavelet matrix (`WaveletMatrix`) uses the core directly for representing the vector.

Serialization format for plain wavelet matrices:

1. `len`: Length of the vector as an element.
2. `data`: The core of the wavelet matrix as `WMCore`.
3. `first`: An `IntVector` storing the position of the first occurrence of each value in the reordered vector.

**Note:** `first` is only defined over the values in the alphabet. If a value is not present in the vector, the corresponding position is `len`.

**Note:** `first` must be bit-packed to minimize its width.
