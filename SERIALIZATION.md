# Serialization formats

For version 0.2.0. Updated 2021-04-22.

## Basics

The simple-sds serialization format is intended as an interchange format for succinct data structures.
It should be easy to reimplement.
The format is also designed to make memory-mapped data structures straightforward.

A file is an array of **elements**, which are unsigned 64-bit little-endian integers.
As a result, the size of the file must be a multiple of 8 bytes.
All array indexes are 0-based.
The reader is always assumed to know the type of the object they are reading.

A fixed-length type is **serializable** if its size is a multiple of 8 bytes.
A serializable object can be serialized by copying the bytes.

## Vectors

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

## Optional structures

An **optional** structure may be present in the file or absent from it.
If a structure is needed in some but not all applications of the parent structure, it can be made optional.
Optional structures may also be used for serializing implementation-dependent support structures that can be built from the parent structure.
The length of an optional structure is the number of elements required to serialize the actual structure (if present) or `0` (if absent).

Serialization format for optional structures:

1. Length of the optional structure as an element.
2. The structure, if present.

The reader can easily skip an optional structure, because its length is stored before the structure itself.

## Raw bitvectors

A **raw bitvector** is a vector of bits.
The items of a raw bitvector are concatenated and stored in a vector of elements in little-endian order.
Bit `i` of the raw bitvector is stored as bit `i % 64` of element `floor(i / 64)`.
A raw bitvector of length `n` requires a vector of `floor((n + 63) / 64)` elements.
Any unused bits in the last element must be set to `0`.

Serialization format for raw bitvectors:

1. Length of the vector as an element.
2. Vector of elements storing the items.

## Integer vector

An **integer vector** is a bit-packed vector of integers.
The **width** of the items can be from 1 to 64 bits.
The items of an integer vector are concatenated and stored in a raw bitvector.
An integer vector of `n` items of width `w` bits requires a raw bitvector of of length `n * w`.

Serialization format for integer vectors:

1. Length of the vector as an element.
2. Width of the items as an element.
3. Raw bitvector storing the items.

## Bitvector

A **bitvector** is vector of bits that supports rank, select, and similar queries.
It extends the functionality of a raw bitvector.

Serialization format for bitvectors:

1. Raw bitvector storing the items.
2. Optional rank support structure.
3. Optional select support structure for set bits.
4. Optional select support structure for unset bits.

The support structures are often both application-dependent and implementation-dependent and hence optional.

## Sparse bitvector

A **sparse bitvector** is an Elias-Fano encoded vector of bits that supports rank, select, and similar queries.
It can be interpreted as a set of integers or a vector of sorted integers, where the integers are the positions of the set bits.
The encoding also supports multisets / duplicate items, but the semantics are not fully clear yet.

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
