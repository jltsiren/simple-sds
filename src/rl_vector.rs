//! A run-length encoded bitvector supporting rank, select, and related queries.
//!
//! This is a run-length encoded bitvector similar to the one used in RLCSA:
//!
//! > Mäkinen, Navarro, Sirén, Välimäki: Storage and Retrieval of Highly Repetitive Sequence Collections.  
//! > Journal of Computational Biology, 2010.  
//! > DOI: [10.1089/cmb.2009.0169](https://doi.org/10.1089/cmb.2009.0169)
//!
//! The vector is encoded by storing the maximal runs of unset and set bits.
//! If there is a run of `n0` unset bits followed by a run of `n1` set bits, we encode it as integers `n0` and `n1 - 1`.
//! Each integer is encoded in little-endian order using 4-bit code units.
//! The lowest 3 bits of each code unit contain data.
//! If the high bit is set, the encoding continues in the next unit.
//! We partition the encoding into 64-unit (32-byte) blocks that consist of entire runs.
//! If there is not enough space left for encoding the next `(n0, n1)`, we pad the block with empty code units and start a new block.
//!
//! For each block, we store a sample `(rank(i, 1), i)`, where `i` is the number of bits encoded before that block.
//! Queries use binary search on the samples to find the right block and then decompress the block sequentially.
//! A [`SampleIndex`] is used for narrowing down the range of the binary search.

use crate::int_vector::IntVector;
use crate::ops::{Vector, Access, Push, Resize, BitVec, Rank, Select, SelectZero, PredSucc};
use crate::rl_vector::index::SampleIndex;
use crate::serialize::Serialize;
use crate::bits;

use std::io::{Error, ErrorKind};
use std::iter::FusedIterator;
use std::{cmp, io};

pub mod index;

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

/// An immutable run-length encoded bitvector supporting rank, select, and related queries.
///
/// This type should be used when the bitvector contains long runs of both set and unset bits.
/// Other bitvector types are more appropriate for dense (no long runs) and sparse (long runs of unset bits) bitvectors.
/// The bitvector is immutable, though it would be easy to implement a mutable version by storing the blocks in a B+-tree rather than a vector.
/// The maximum length of the vector is approximately [`usize::MAX`] bits.
///
/// Conversions between various [`BitVec`] types are possible using the [`From`] trait.
///
/// `RLVector` implements the following `simple_sds` traits:
/// * Basic functionality: [`BitVec`]
/// * Queries and operations: [`Rank`], [`Select`], [`SelectZero`], [`PredSucc`]
/// * Serialization: [`Serialize`]
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{BitVec, Rank, Select, SelectZero, PredSucc};
/// use simple_sds::rl_vector::{RLVector, RLBuilder};
///
/// let mut builder = RLBuilder::new();
/// builder.try_set(18, 22);
/// builder.try_set(95, 15);
/// builder.try_set(110, 10);
/// builder.try_set(140, 12);
/// builder.set_len(200);
/// let rv = RLVector::from(builder);
///
/// // BitVec
/// assert_eq!(rv.len(), 200);
/// assert!(!rv.is_empty());
/// assert_eq!(rv.count_ones(), 59);
/// assert_eq!(rv.count_zeros(), 141);
/// assert!(rv.get(119));
/// assert!(!rv.get(120));
/// for (index, value) in rv.iter().enumerate() {
///     assert_eq!(value, rv.get(index));
/// }
///
/// // Rank
/// assert!(rv.supports_rank());
/// assert_eq!(rv.rank(100), 27);
/// assert_eq!(rv.rank(130), 47);
/// assert_eq!(rv.rank_zero(60), 38);
///
/// // Select
/// assert!(rv.supports_select());
/// assert_eq!(rv.select(24), Some(97));
/// let mut iter = rv.select_iter(46);
/// assert_eq!(iter.next(), Some((46, 119)));
/// assert_eq!(iter.next(), Some((47, 140)));
/// let v: Vec<(usize, usize)> = rv.one_iter().take(4).collect();
/// assert_eq!(v, vec![(0, 18), (1, 19), (2, 20), (3, 21)]);
///
/// // SelectZero
/// assert!(rv.supports_select_zero());
/// assert_eq!(rv.select_zero(130), Some(189));
/// let mut iter = rv.select_zero_iter(72);
/// assert_eq!(iter.next(), Some((72, 94)));
/// assert_eq!(iter.next(), Some((73, 120)));
/// let v: Vec<(usize, usize)> = rv.zero_iter().take(4).collect();
/// assert_eq!(v, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
///
/// // PredSucc
/// assert!(rv.supports_pred_succ());
/// assert!(rv.predecessor(17).next().is_none());
/// assert_eq!(rv.predecessor(18).next(), Some((0, 18)));
/// assert_eq!(rv.predecessor(40).next(), Some((21, 39)));
/// assert_eq!(rv.successor(139).next(), Some((47, 140)));
/// assert_eq!(rv.successor(140).next(), Some((47, 140)));
/// assert!(rv.successor(152).next().is_none());
/// ```
///
/// # Notes
///
/// * `RLVector` never panics from I/O errors.
/// * All `RLVector` queries are always enabled without additional support structures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RLVector {
    len: usize,
    ones: usize,
    rank_index: SampleIndex,
    select_index: SampleIndex,
    select_zero_index: SampleIndex,
    // (ones, bits) up to the start of each block.
    samples: IntVector,
    // Concatenated blocks.
    data: IntVector,
}

impl RLVector {
    /// Number of bits in a code unit.
    pub const CODE_SIZE: usize = 4;

    // Number of data bits in a code unit.
    const CODE_SHIFT: usize = Self::CODE_SIZE - 1;

    // If this bit is set in a code unit, the encoding continues in the next unit.
    const CODE_FLAG: u64 = 1 << Self::CODE_SHIFT;

    // Largest value that can be stored in a single code unit.
    const CODE_MASK: u64 = (1 << Self::CODE_SHIFT) - 1;

    /// Number of code units in a block.
    pub const BLOCK_SIZE: usize = 64;

    /// Returns a copy of the source bitvector as `RLVector`.
    ///
    /// The copy is created by iterating over the set bits using [`Select::one_iter`].
    /// [`From`] implementations from other bitvector types should generally use this function.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::bit_vector::BitVector;
    /// use simple_sds::ops::BitVec;
    /// use simple_sds::rl_vector::RLVector;
    /// use std::iter::FromIterator;
    ///
    /// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
    /// let bv = BitVector::from_iter(source);
    /// let rv = RLVector::copy_bit_vec(&bv);
    /// assert_eq!(rv.len(), bv.len());
    /// assert_eq!(rv.count_ones(), bv.count_ones());
    /// ```
    pub fn copy_bit_vec<'a, T: BitVec<'a> + Select<'a>>(source: &'a T) -> Self {
        let mut builder = RLBuilder::new();
        for (_, index) in source.one_iter() {
            unsafe { builder.set_bit_unchecked(index); }
        }
        builder.set_len(source.len());
        RLVector::from(builder)
    }

    /// Returns an iterator over the runs of set bits in the bitvector.
    ///
    /// See [`RunIter`] for an example.
    pub fn run_iter(&self) -> RunIter<'_> {
        RunIter {
            parent: self,
            offset: 0,
            pos: (0, 0),
            limit: self.ones_after(0),
        }
    }

    // Returns the number of blocks in the encoding.
    fn blocks(&self) -> usize {
        self.samples.len() / 2
    }

    // Returns the number of set bits after the given block.
    fn ones_after(&self, block: usize) -> usize {
        if block + 1 < self.blocks() {
            self.samples.get(2 * (block + 1)) as usize
        } else {
            self.count_ones()
        }
    }

    // Decodes a value starting from `offset` and returns (value, new offset).
    fn decode(&self, offset: usize) -> (usize, usize) {
        let mut value = 0;
        let mut offset = offset;
        let mut shift = 0;
        loop {
            let code = self.data.get(offset);
            offset += 1;
            value += ((code & Self::CODE_MASK) << shift) as usize;
            shift += Self::CODE_SHIFT;
            if code & Self::CODE_FLAG == 0 {
                return (value, offset);
            }
        }
    }

    // Returns the identifier of the last block `i` in the range where `f(i) <= value`.
    fn block_for<F: Fn(usize) -> usize>(low: usize, high: usize, value: usize, f: F) -> usize {
        let mut low = low;
        let mut high = high;
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            let candidate = f(mid);
            if candidate <= value {
                low = mid;
            } else {
                high = mid;
            }
        }
        low
    }

    // Returns an iterator covering the given block.
    fn iter_for_block(&self, block: usize) -> RunIter<'_> {
        let pos = if self.samples.is_empty() { (0, 0) } else { (self.samples.get(2 * block) as usize, self.samples.get(2 * block + 1) as usize) };
        RunIter {
            parent: self,
            offset: block * Self::BLOCK_SIZE,
            pos,
            limit: self.ones_after(block),
        }
    }

    // Returns an iterator covering the block that contains the given bit, or an empty iterator if there is no such bit.
    fn iter_for_bit(&self, index: usize) -> RunIter<'_> {
        if index >= self.len() {
            return RunIter::empty_iter(self);
        }

        let range = self.rank_index.range(index);
        let block = Self::block_for(range.start, range.end, index, |i| self.samples.get(2 * i + 1) as usize);
        self.iter_for_block(block)
    }

    // Returns an iterator covering the block that contains the given set bit, or an empty iterator if there is no such bit.
    fn iter_for_one(&self, rank: usize) -> RunIter<'_> {
        if rank >= self.count_ones() {
            return RunIter::empty_iter(self);
        }

        let range = self.select_index.range(rank);
        let block = Self::block_for(range.start, range.end, rank, |i| self.samples.get(2 * i) as usize);
        self.iter_for_block(block)
    }

    // Returns an iterator covering the block that contains the given unset bit, or an empty iterator if there is no such bit.
    fn iter_for_zero(&self, rank: usize) -> RunIter<'_> {
        if rank >= self.count_zeros() {
            return RunIter::empty_iter(self);
        }

        let range = self.select_zero_index.range(rank);
        let block = Self::block_for(range.start, range.end, rank, |i| (self.samples.get(2 * i + 1) - self.samples.get(2 * i)) as usize);
        self.iter_for_block(block)
    }
}

//-----------------------------------------------------------------------------

/// Space-efficient [`RLVector`] construction.
///
/// `RLBuilder` builds an [`RLVector`] incrementally.
/// Bits must be set in order, and they cannot be unset later.
/// Set bits are combined into maximal runs.
/// Once the construction is finished, the builder can be converted into an [`RLVector`] using the [`From`] trait.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::BitVec;
/// use simple_sds::rl_vector::{RLVector, RLBuilder};
///
/// let mut builder = RLBuilder::new();
///
/// builder.try_set(18, 22);
/// assert_eq!(builder.len(), 40);
///
/// // Combine two runs into one.
/// builder.try_set(95, 15);
/// builder.try_set(110, 10);
///
/// // Trying to set earlier bits will fail.
/// assert!(builder.try_set(100, 1).is_err());
///
/// builder.try_set(140, 12);
///
/// // Append a final run of unset bits.
/// builder.set_len(200);
///
/// let rv = RLVector::from(builder);
/// assert_eq!(rv.len(), 200);
/// ```
#[derive(Clone, Debug)]
pub struct RLBuilder {
    len: usize,
    ones: usize,
    // Position after the last encoded run.
    tail: usize,
    run: (usize, usize),
    samples: Vec<(usize, usize)>,
    data: IntVector,
}

impl RLBuilder {
    /// Returns an empty `RLBuilder`.
    pub fn new() -> Self {
        RLBuilder::default()
    }

    /// Returns the length of the bitvector.
    ///
    /// This is the first position that can be set.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the bitvector is empty.
    ///
    /// Makes Clippy happy.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of set bits in the bitvector.
    pub fn count_ones(&self) -> usize {
        self.ones
    }

    /// Returns the number of unset bits in the bitvector.
    pub fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }

    // Returns the position after the last encoded run.
    fn tail(&self) -> usize {
        self.tail
    }

    // Returns the number of blocks in the encoding.
    fn blocks(&self) -> usize {
        self.samples.len()
    }

    /// Encodes a run of set bits of length `len` starting at position `start`.
    ///
    /// Does nothing if `len == 0`.
    /// Returns [`Err`] if `start < self.len()` of `start + len > usize::MAX`.
    ///
    /// # Arguments
    ///
    /// * `start`: Starting position of the run.
    /// * `len`: Length of the run.
    pub fn try_set(&mut self, start: usize, len: usize) -> Result<(), String> {
        if start < self.len() {
            return Err(format!("RLBuilder: Cannot set bit {} when length is {}", start, self.len()));
        }
        if usize::MAX - len < start {
            return Err(format!("RLBuilder: A run of length {} starting at {} exceeds the maximum length of a bitvector", start, len));
        }
        unsafe { self.set_run_unchecked(start, len); }
        Ok(())
    }

    /// Sets the specified bit in the bitvector.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if `index < self.len()`.
    pub unsafe fn set_bit_unchecked(&mut self, index: usize) {
        self.set_run_unchecked(index, 1);
    }

    /// Encodes a run of set bits of length `len` starting at position `start`.
    ///
    /// Does nothing if `len == 0`.
    ///
    /// # Arguments
    ///
    /// * `start`: Starting position of the run.
    /// * `len`: Length of the run.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if `start < self.len()` of `start + len > usize::MAX`.
    pub unsafe fn set_run_unchecked(&mut self, start: usize, len: usize) {
        if len <= 0 {
            return;
        }
        if start == self.len() {
            self.len += len;
            self.ones += len;
            self.run.1 += len;
        } else {
            self.flush();
            self.len = start + len;
            self.ones += len;
            self.run = (start, len);
        }
    }

    /// Sets the length of the bitvector to `len` bits.
    ///
    /// No effect if `self.len() >= len`.
    /// This is intended for appending a final run of unset bits.
    pub fn set_len(&mut self, len: usize) {
        if len > self.len() {
            self.flush();
            self.len = len;
        }
    }

    // Encodes the current run if necessary and sets the active run to `(self.len(), 0)`.
    fn flush(&mut self) {
        if self.run.1 <= 0 {
            return;
        }

        // Add a new block if there is not enough space for the run in the current block.
        let units_needed = Self::code_len(self.run.0 - self.tail()) + Self::code_len(self.run.1 - 1);
        if self.data.len() + units_needed > self.blocks() * RLVector::BLOCK_SIZE {
            self.data.resize(self.blocks() * RLVector::BLOCK_SIZE, 0);
            let sample = (self.ones - self.run.1, self.tail);
            self.samples.push(sample);
        }

        // Encode the run and update the statistics.
        self.encode(self.run.0 - self.tail());
        self.encode(self.run.1 - 1);
        self.tail = self.run.0 + self.run.1;
        self.run = (self.len(), 0);
    }

    // Encodes the given value.
    fn encode(&mut self, value: usize) {
        let mut value = value as u64;
        while value > RLVector::CODE_MASK {
            self.data.push((value & RLVector::CODE_MASK) | RLVector::CODE_FLAG);
            value = value >> RLVector::CODE_SHIFT;
        }
        self.data.push(value);
    }

    // Number of code units required for encoding the value.
    fn code_len(value: usize) -> usize {
        bits::div_round_up(bits::bit_len(value as u64), RLVector::CODE_SHIFT)
    }
}

impl Default for RLBuilder {
    fn default() -> Self {
        RLBuilder {
            len: 0,
            ones: 0,
            tail: 0,
            run: (0, 0),
            samples: Vec::new(),
            data: IntVector::new(RLVector::CODE_SIZE).unwrap(),
        }
    }
}

impl From<RLBuilder> for RLVector {
    fn from(builder: RLBuilder) -> Self {
        let mut builder = builder;
        builder.flush();

        // Build indexes for narrowing down binary search ranges.
        let rank_index = SampleIndex::new(builder.samples.iter().map(|(_, bits)| *bits), builder.len());
        let select_index = SampleIndex::new(builder.samples.iter().map(|(ones, _)| *ones), builder.count_ones());
        let select_zero_index = SampleIndex::new(builder.samples.iter().map(|(ones, bits)| bits - ones), builder.count_zeros());

        // Compress the samples.
        let max_value = builder.samples.last().unwrap_or(&(0, 0)).1;
        let mut samples = IntVector::with_capacity(2 * builder.blocks(), bits::bit_len(max_value as u64)).unwrap();
        for (ones, bits) in builder.samples.iter() {
            samples.push(*ones as u64);
            samples.push(*bits as u64);
        }

        RLVector {
            len: builder.len(),
            ones: builder.count_ones(),
            rank_index,
            select_index,
            select_zero_index,
            samples: samples,
            data: builder.data,
        }
    }
}

//-----------------------------------------------------------------------------

/// A read-only iterator over the runs in [`RLVector`].
///
/// The type of `Item` is `(`[`usize`]`, `[`usize`]`)`.
/// The first value is the starting position of a maximal run of set bits, and the second value is its length.
///
/// Most [`RLVector`] queries use this iterator for iterating over the runs in a block.
///
/// # Examples
///
/// ```
/// use simple_sds::rl_vector::{RLVector, RLBuilder};
///
/// let mut builder = RLBuilder::new();
/// builder.try_set(18, 22);
/// builder.try_set(95, 15);
/// builder.try_set(110, 10); // Merge with the previous run.
/// builder.try_set(140, 12);
/// builder.set_len(200);
/// let rv = RLVector::from(builder);
///
/// let runs: Vec<(usize, usize)> = rv.run_iter().collect();
/// assert_eq!(runs, vec![(18, 22), (95, 25), (140, 12)]);
///
/// let mut iter = rv.run_iter();
/// assert_eq!(iter.offset(), 0);
/// assert_eq!(iter.rank(), 0);
/// assert_eq!(iter.rank_zero(), 0);
/// while let Some(_) = iter.next() {}
/// assert_eq!(iter.offset(), 140 + 12);
/// assert_eq!(iter.rank(), 22 + 25 + 12);
/// assert_eq!(iter.rank_zero(), 140 - 22 - 25);
/// ```
#[derive(Clone, Debug)]
pub struct RunIter<'a> {
    parent: &'a RLVector,
    // Offset in the encoding.
    offset: usize,
    // (rank, index) reached so far.
    pos: (usize, usize),
    // Number of ones after the current block.
    limit: usize,
}

impl<'a> RunIter<'a> {
    /// Returns the position in the bitvector after the latest run.
    pub fn offset(&self) -> usize {
        self.pos.1
    }

    /// Returns the number of set bits up to the end of the latest run.
    pub fn rank(&self) -> usize {
        self.pos.0
    }

    /// Returns the number of unset bits up to the end of the latest run.
    pub fn rank_zero(&self) -> usize {
        self.offset() - self.rank()
    }

    // Returns an empty iterator for the parent bitvector.
    fn empty_iter(parent: &'a RLVector) -> Self {
        RunIter {
            parent,
            offset: parent.data.len(),
            pos: (parent.count_ones(), parent.len()),
            limit: parent.count_ones(),
        }
    }

    // Returns the position in the bitvector for the set bit of given rank, assuming that it is covered by the current run.
    fn offset_for(&self, rank: usize) -> usize {
        self.offset() - (self.rank() - rank)
    }

    // Returns the rank at a given position in the bitvector, assuming that it is covered by the current run.
    fn rank_at(&self, index: usize) -> usize {
        self.rank() - (self.offset() - index)
    }

    // Like `next()`, but only advances if the `advance()` returns `true` for the next run.
    fn advance_if<F: FnMut(Option<<Self as Iterator>::Item>) -> bool>(&mut self, mut advance: F) -> Option<<Self as Iterator>::Item> {
        if self.offset >= self.parent.data.len() {
            // We are at the end anyway, but we will call `advance()` to let the user know
            // what the next run would be.
            let _ = advance(None);
            return None;
        }

        // Move to the next block if we have run out of ones.
        let mut offset = self.offset;
        let mut limit = self.limit;
        if self.rank() >= self.limit {
            let block = bits::div_round_up(offset, RLVector::BLOCK_SIZE);
            offset = block * RLVector::BLOCK_SIZE;
            if block >= self.parent.blocks() {
                if advance(None) {
                    self.offset = offset;
                }
                return None;
            }
            limit = self.parent.ones_after(block);
        }

        // Decode the next run.
        let (gap, offset) = self.parent.decode(offset);
        let start = self.offset() + gap;
        let (len, offset) = self.parent.decode(offset);
        let result = Some((start, len + 1));

        if advance(result) {
            self.offset = offset;
            self.limit = limit;
            self.pos.0 += len + 1;
            self.pos.1 = start + len + 1;
        }
        result
    }
}

impl<'a> Iterator for RunIter<'a> {
    // (start, length)
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.advance_if(|_| true)
    }
}

impl<'a> FusedIterator for RunIter<'a> {}

//-----------------------------------------------------------------------------

/// A read-only iterator over [`RLVector`].
///
/// The type of `Item` is [`bool`].
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{BitVec};
/// use simple_sds::rl_vector::{RLVector, RLBuilder};
///
/// let mut builder = RLBuilder::new();
/// builder.try_set(18, 22);
/// builder.try_set(95, 15);
/// builder.try_set(110, 10);
/// builder.try_set(140, 12);
/// builder.set_len(200);
/// let rv = RLVector::from(builder);
///
/// assert_eq!(rv.iter().len(), rv.len());
/// for (index, value) in rv.iter().enumerate() {
///     assert_eq!(value, rv.get(index));
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Iter<'a> {
    iter: RunIter<'a>,
    // Run from the iterator.
    run: Option<(usize, usize)>,
    // Next bitvector offset.
    pos: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        // Read the next run if we have processed the current one.
        if let Some((start, len)) = self.run {
            if self.pos >= start + len {
                self.run = self.iter.next();
            }
        }

        // Determine the next bit and advance.
        match self.run {
            Some((start, _)) => {
                self.pos += 1;
                Some(self.pos > start)
            },
            None => {
                if self.pos >= self.iter.parent.len() {
                    None
                } else {
                    self.pos += 1;
                    Some(false)
                }
            },
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.iter.parent.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {}

impl<'a> FusedIterator for Iter<'a> {}

//-----------------------------------------------------------------------------

impl<'a> BitVec<'a> for RLVector {
    type Iter = Iter<'a>;

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn count_ones(&self) -> usize {
        self.ones
    }

    fn get(&self, index: usize) -> bool {
        let mut iter = self.iter_for_bit(index);
        while let Some((start, _)) = iter.next() {
            if start > index {
                return false;
            }
            if index < iter.offset() {
                return true;
            }
        }

        // Final run of unset bits.
        false
    }

    fn iter(&'a self) -> Self::Iter {
        Self::Iter {
            iter: self.run_iter(),
            run: Some((0, 0)),
            pos: 0,
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> Rank<'a> for RLVector {
    fn supports_rank(&self) -> bool {
        true
    }

    fn enable_rank(&mut self) {}

    fn rank(&self, index: usize) -> usize {
        let mut iter = self.iter_for_bit(index);
        while let Some((start, len)) = iter.next() {
            // We reached `index` but this run is too late to affect the rank.
            if start >= index {
                return iter.rank() - len;
            }
            // We reached `index` and a part of this run affects the rank.
            if iter.offset() >= index {
                return iter.rank_at(index);
            }
        }

        iter.rank()
    }
}

//-----------------------------------------------------------------------------

/// An iterator over the set bits in [`RLVector`].
///
/// The type of `Item` is `(`[`usize`]`, `[`usize`]`)`.
/// This can be interpreted as:
///
/// * `(index, value)` or `(i, select(i))` in the integer array; or
/// * `(rank(j), j)` in the bit array with `j` such that `self.get(j) == true`.
///
/// Note that `index` is not always the index provided by [`Iterator::enumerate`].
/// Queries may create iterators in the middle of the bitvector.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{BitVec, Select};
/// use simple_sds::rl_vector::{RLVector, RLBuilder};
///
/// let mut builder = RLBuilder::new();
/// builder.try_set(18, 22);
/// builder.try_set(95, 15);
/// builder.try_set(110, 10);
/// builder.try_set(140, 12);
/// builder.set_len(200);
/// let rv = RLVector::from(builder);
///
/// let mut iter = rv.one_iter();
/// assert_eq!(iter.len(), rv.count_ones());
/// assert_eq!(iter.next(), Some((0, 18)));
/// assert_eq!(iter.next(), Some((1, 19)));
/// assert_eq!(iter.next(), Some((2, 20)));
/// ```
#[derive(Clone, Debug)]
pub struct OneIter<'a> {
    iter: RunIter<'a>,
    // Did we get a `None` from the iterator?
    got_none: bool,
    // Rank of the next set bit.
    rank: usize,
}

impl<'a> OneIter<'a> {
    // Returns an empty iterator for the parent bitvector.
    fn empty_iter(parent: &'a RLVector) -> Self {
        OneIter {
            iter: RunIter::empty_iter(parent),
            got_none: true,
            rank: parent.count_ones(),
        }
    }
}

impl<'a> Iterator for OneIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Read the next run if we have processed the current one.
        if !self.got_none && self.rank >= self.iter.rank() {
            self.got_none = self.iter.next().is_none();
        }

        // Determine the next set bit and advance.
        if self.got_none {
            None
        } else {
            let result = (self.rank, self.iter.offset_for(self.rank));
            self.rank += 1;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.iter.parent.count_ones() - self.rank;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for OneIter<'a> {}

impl<'a> FusedIterator for OneIter<'a> {}

//-----------------------------------------------------------------------------

/// An iterator over the unset bits in [`RLVector`].
///
/// The type of `Item` is `(`[`usize`]`, `[`usize`]`)`.
/// This can be interpreted as:
///
/// * `(index, value)` or `(i, select(i))` in the integer array of the complement; or
/// * `(rank(j), j)` in the bit array with `j` such that `self.get(j) == false`.
///
/// Note that `index` is not always the index provided by [`Iterator::enumerate`].
/// Queries may create iterators in the middle of the bitvector.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{BitVec, SelectZero};
/// use simple_sds::rl_vector::{RLVector, RLBuilder};
///
/// let mut builder = RLBuilder::new();
/// builder.try_set(18, 22);
/// builder.try_set(95, 15);
/// builder.try_set(110, 10);
/// builder.try_set(140, 12);
/// builder.set_len(200);
/// let rv = RLVector::from(builder);
///
/// let mut iter = rv.zero_iter();
/// assert_eq!(iter.len(), rv.count_zeros());
/// assert_eq!(iter.next(), Some((0, 0)));
/// assert_eq!(iter.next(), Some((1, 1)));
/// assert_eq!(iter.next(), Some((2, 2)));
/// ```
#[derive(Clone, Debug)]
pub struct ZeroIter<'a> {
    iter: RunIter<'a>,
    // Did we get a `None` from the iterator?
    got_none: bool,
    // (rank, index) for the next unset bit.
    pos: (usize, usize),
}

impl<'a> Iterator for ZeroIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Read the next run of set bits if we have reached the current one.
        if !self.got_none && self.pos.0 >= self.iter.rank_zero() {
            self.pos.1 = self.iter.offset();
            self.got_none = self.iter.next().is_none();
        }

        // Determine the next bit and advance.
        if self.pos.0 >= self.iter.parent.count_zeros() {
            None
        } else {
            let result = self.pos;
            self.pos.0 += 1; self.pos.1 += 1;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.iter.parent.count_zeros() - self.pos.0;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for ZeroIter<'a> {}

impl<'a> FusedIterator for ZeroIter<'a> {}

//-----------------------------------------------------------------------------

impl<'a> Select<'a> for RLVector {
    type OneIter = OneIter<'a>;

    fn supports_select(&self) -> bool {
        true
    }

    fn enable_select(&mut self) {}

    fn one_iter(&'a self) -> Self::OneIter {
        Self::OneIter {
            iter: self.run_iter(),
            got_none: false,
            rank: 0,
        }
    }

    fn select(&'a self, rank: usize) -> Option<usize> {
        if rank >= self.count_ones() {
            return None;
        }

        let mut iter = self.iter_for_one(rank);
        while iter.rank() <= rank {
            let _ = iter.next();
        }
        Some(iter.offset_for(rank))
    }

    fn select_iter(&'a self, rank: usize) -> Self::OneIter {
        if rank >= self.count_ones() {
            return Self::OneIter::empty_iter(self);
        }

        let mut iter = self.iter_for_one(rank);
        while iter.rank() < rank {
            let _ = iter.next();
        }
        Self::OneIter {
            iter,
            got_none: false,
            rank,
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> SelectZero<'a> for RLVector {
    type ZeroIter = ZeroIter<'a>;

    fn supports_select_zero(&self) -> bool {
        true
    }

    fn enable_select_zero(&mut self) {}

    fn zero_iter(&'a self) -> Self::ZeroIter {
        let mut iter = self.run_iter();
        // We must take the first run instead of using (0, 0).
        // Otherwise `ZeroIter` would not work if the first run starts at 0.
        let got_none = iter.next().is_none();
        Self::ZeroIter {
            iter,
            got_none,
            pos: (0, 0),
        }
    }

    fn select_zero(&'a self, rank: usize) -> Option<usize> {
        if rank >= self.count_zeros() {
            return None;
        }

        // Determine the number of set bits before the relevant run of unset bits.
        let mut iter = self.iter_for_zero(rank);
        let mut ones = iter.rank();
        loop {
            match iter.next() {
                Some(_) => {
                    if iter.rank_zero() > rank {
                        return Some(rank + ones);
                    }
                    ones = iter.rank();
                },
                None => {
                    return Some(rank + ones);
                },
            }
        }
    }

    fn select_zero_iter(&'a self, rank: usize) -> Self::ZeroIter {
        if rank >= self.count_zeros() {
            return Self::ZeroIter {
                iter: RunIter::empty_iter(self),
                got_none: true,
                pos: (self.count_zeros(), self.len()),
            }
        }

        // Determine the number of set bits before the relevant run of unset bits.
        let mut iter = self.iter_for_zero(rank);
        let mut ones = iter.rank();
        loop {
            match iter.next() {
                Some(_) => {
                    if iter.rank_zero() > rank {
                        return Self::ZeroIter {
                            iter,
                            got_none: false,
                            pos: (rank, rank + ones),
                        };
                    }
                    ones = iter.rank();
                },
                None => {
                    return Self::ZeroIter {
                        iter,
                        got_none: true,
                        pos: (rank, rank + ones),
                    }
                },
            }
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> PredSucc<'a> for RLVector {
    type OneIter = OneIter<'a>;

    fn supports_pred_succ(&self) -> bool {
        true
    }

    fn enable_pred_succ(&mut self) {}

    fn predecessor(&'a self, value: usize) -> Self::OneIter {
        if self.is_empty() {
            return Self::OneIter::empty_iter(self);
        }

        // A predecessor past the end is the same as predecessor at the end.
        let value = cmp::min(value, self.len() - 1);

        // Find the block that would contain the value. Then advance the iterator
        // until the next run starts after the value we are interested in.
        let mut iter = self.iter_for_bit(value);
        let mut iterate = true;
        while iterate {
            let _ = iter.advance_if(|next| {
                if next.is_none() {
                    iterate = false;
                    return false;
                }
                let (start, _) = next.unwrap();

                iterate = start <= value;
                return iterate;
            });
        }

        // If we are before the first run, there is no predecessor.
        if iter.rank() == 0 {
            return Self::OneIter::empty_iter(self);
        }

        let rank = if iter.offset() > value { iter.rank_at(value) } else { iter.rank() - 1 };
        Self::OneIter {
            iter,
            got_none: false,
            rank,
        }
    }

    fn successor(&'a self, value: usize) -> Self::OneIter {
        if value >= self.len() {
            return Self::OneIter::empty_iter(self);
        }

        // Find the block that would contain the value. Then advance the iterator
        // until the last run covers a position at or after the value we are interested in.
        let mut iter = self.iter_for_bit(value);
        let rank;
        loop {
            let result = iter.next();
            if result.is_none() {
                return Self::OneIter::empty_iter(self);
            }
            let (start, len) = result.unwrap();
            if start > value {
                rank = iter.rank() - len;
                break;
            }
            if iter.offset() > value {
                rank = iter.rank_at(value);
                break;
            }
        }

        Self::OneIter {
            iter,
            got_none: false,
            rank,
        }
    }
}

//-----------------------------------------------------------------------------

impl Serialize for RLVector {
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.len.serialize(writer)?;
        self.ones.serialize(writer)?;
        Ok(())
    }

    fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.samples.serialize(writer)?;
        self.data.serialize(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let len = usize::load(reader)?;
        let ones = usize::load(reader)?;
        let samples = IntVector::load(reader)?;
        let data = IntVector::load(reader)?;

        // Sanity checks.
        let sample_blocks = samples.len() / 2;
        let data_blocks = bits::div_round_up(data.len(), Self::BLOCK_SIZE);
        if sample_blocks != data_blocks {
            return Err(Error::new(ErrorKind::InvalidData, "Mismatch between number of blocks and samples"));
        }

        // Rebuild indexes for narrowing down binary search ranges.
        let rank_index = SampleIndex::new((0..sample_blocks).map(|block| samples.get(2 * block + 1) as usize), len);
        let select_index = SampleIndex::new((0..sample_blocks).map(|block| samples.get(2 * block) as usize), ones);
        let select_zero_index = SampleIndex::new((0..sample_blocks).map(|block| (samples.get(2 * block + 1) - samples.get(2 * block)) as usize), len - ones);

        let result = RLVector {
            len, ones, rank_index, select_index, select_zero_index, samples, data,
        };
        Ok(result)
    }

    fn size_in_elements(&self) -> usize {
        self.len.size_in_elements() +
        self.ones.size_in_elements() +
        self.samples.size_in_elements() +
        self.data.size_in_elements()
    }
}

//-----------------------------------------------------------------------------
