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
//! For each block, we store a sample `(i, rank(i, 1))`, where `i` is the number of bits encoded before that block.
//! Queries use binary search on the samples to find the right block and then decompress the block sequentially.
// FIXME rank/select indexes

// FIXME document serialization format

use crate::int_vector::IntVector;
use crate::ops::{Vector, Access, Push, Resize, BitVec, Select};
use crate::bits;

use std::iter::FusedIterator;

// FIXME the rank/select index structure as a submodule

// FIXME tests
//#[cfg(test)]
//mod tests;

//-----------------------------------------------------------------------------

// FIXME document, example, tests
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RLVector {
    len: usize,
    ones: usize,
    // FIXME rank/select indexes and divisors as in RLCSA
    // (bits, ones) at the start of each block.
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

    // FIXME example
    /// Returns a copy of the source bitvector as `RLVector`.
    ///
    /// The copy is created by iterating over the set bits using [`Select::one_iter`].
    /// [`From`] implementations from other bitvector types should generally use this function.
    pub fn copy_bit_vec<'a, T: BitVec<'a> + Select<'a>>(source: &'a T) -> Self {
        let mut builder = RLBuilder::new();
        for (_, index) in source.one_iter() {
            unsafe { builder.set_unchecked(index); }
        }
        builder.set_len(source.len());
        RLVector::from(builder)
    }

    // FIXME document
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
            self.samples.get(2 * (block + 1) + 1) as usize
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

    // Returns the identifier of the last block `i` in the range where `f(i) < value`.
    fn block_for<F: Fn(usize) -> usize>(low: usize, high: usize, value: usize, f: F) -> usize {
        let mut low = low;
        let mut high = high;
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            let candidate = f(mid);
            if candidate < value {
                low = mid;
            } else {
                high = mid;
            }
        }
        low
    }

    // Returns the identifier of the block containing the given bit.
    fn block_for_bit(&self, index: usize) -> Option<usize> {
        if index >= self.len() {
            return None;
        }
        // FIXME use the index
        Some(Self::block_for(0, self.blocks(), index, |i| self.samples.get(2 * i) as usize))
    }

    // Returns the identifier of the block containing the set bit of given rank.
    fn block_for_one(&self, rank: usize) -> Option<usize> {
        if rank >= self.count_ones() {
            return None;
        }
        // FIXME use the index
        Some(Self::block_for(0, self.blocks(), rank, |i| self.samples.get(2 * i + 1) as usize))
    }

    // Returns the indentifier of the block containing the unset bit of given rank.
    fn block_for_zero(&self, rank: usize) -> Option<usize> {
        if rank >= self.count_zeros() {
            return None;
        }
        // FIXME use the index
        Some(Self::block_for(0, self.blocks(), rank, |i| (self.samples.get(2 * i) - self.samples.get(2 * i + 1)) as usize))
    }
}

//-----------------------------------------------------------------------------

// FIXME document, example, tests
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
    // FIXME document
    pub fn new() -> Self {
        RLBuilder {
            len: 0,
            ones: 0,
            tail: 0,
            run: (0, 0),
            samples: Vec::new(),
            data: IntVector::new(RLVector::CODE_SIZE).unwrap(),
        }
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
    pub unsafe fn set_unchecked(&mut self, index: usize) {
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
            self.len = len;
            self.flush();
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
            self.samples.push((self.tail, self.ones - self.run.1));
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
            self.data.push((value | RLVector::CODE_MASK) | RLVector::CODE_FLAG);
            value = value >> RLVector::CODE_SHIFT;
        }
        self.data.push(value);
    }

    // Number of code units required for encoding the value.
    fn code_len(value: usize) -> usize {
        bits::div_round_up(bits::bit_len(value as u64), RLVector::CODE_SIZE)
    }
}

impl From<RLBuilder> for RLVector {
    fn from(builder: RLBuilder) -> Self {
        let mut builder = builder;
        builder.flush();

        // FIXME build rank/select indexes

        // Compress the samples.
        let max_value = builder.samples.last().unwrap_or(&(0, 0)).0;
        let mut samples = IntVector::with_capacity(2 * builder.blocks(), bits::bit_len(max_value as u64)).unwrap();
        for (bits, ones) in builder.samples.iter() {
            samples.push(*bits as u64);
            samples.push(*ones as u64);
        }

        RLVector {
            len: builder.len,
            ones: builder.ones,
            samples: samples,
            data: builder.data,
        }
    }
}

//-----------------------------------------------------------------------------

// FIXME document, example, tests
pub struct RunIter<'a> {
    parent: &'a RLVector,
    // Offset in the encoding.
    offset: usize,
    // (bitvector offset, number of ones)
    pos: (usize, usize),
    // Number of ones after the current block.
    limit: usize,
}

impl<'a> Iterator for RunIter<'a> {
    // (start, length)
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.parent.data.len() {
            return None;
        }

        // Move to the next block if we have run out of ones.
        if self.pos.1 > self.limit {
            let block = bits::div_round_up(self.offset, RLVector::BLOCK_SIZE);
            self.offset = block * RLVector::BLOCK_SIZE;
            if block >= self.parent.blocks() {
                return None;
            }
            self.limit = self.parent.ones_after(block);
        }

        // Decode the next run.
        let (gap, next_offset) = self.parent.decode(self.offset);
        let start = self.pos.0 + gap;
        let (len, next_offset) = self.parent.decode(next_offset);
        self.offset = next_offset;
        self.pos.0 = start + len + 1;
        self.pos.1 += len + 1;

        Some((start, len + 1))
    }
}

impl<'a> FusedIterator for RunIter<'a> {}

//-----------------------------------------------------------------------------

// FIXME document, example, tests
pub struct Iter<'a> {
    iter: RunIter<'a>,
    // Run from the iterator.
    run: Option<(usize, usize)>,
    // Bitvector offset.
    pos: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        // Read the next run if we have processed the current one.
        if self.run.is_some() {
            let (start, len) = self.run.unwrap();
            if self.pos >= start + len {
                self.run = self.iter.next();
            }
        }

        // Determine the next bit and advance.
        self.pos += 1;
        match self.run {
            Some((start, _)) => Some(self.pos > start),
            None => if self.pos > self.iter.parent.len() { None } else { Some(false) },
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.iter.parent.len() - self.pos;
        (remaining, Some(remaining))
    }
}

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
        let block = self.block_for_bit(index).unwrap();
        let mut iter = RunIter {
            parent: self,
            offset: block * Self::BLOCK_SIZE,
            pos: (self.samples.get(2 * block) as usize, self.samples.get(2 * block + 1) as usize),
            limit: self.ones_after(block),
        };

        while let Some((start, len)) = iter.next() {
            if start > index {
                return false;
            }
            if index < start + len {
                return true;
            }
        }

        panic!("RLVector::get(): Cannot find bit {}", index);
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

// FIXME Rank

//-----------------------------------------------------------------------------

// FIXME OneIter

// FIXME Iterator, ExactSizeIterator, FusedIterator

//-----------------------------------------------------------------------------

// FIXME ZeroIter

// FIXME Iterator, ExactSizeIterator, FusedIterator

//-----------------------------------------------------------------------------

// FIXME Select

//-----------------------------------------------------------------------------

// FIXME SelectZero

//-----------------------------------------------------------------------------

// FIXME PredSucc

//-----------------------------------------------------------------------------

// FIXME Serialize

//-----------------------------------------------------------------------------

// FIXME From other bitvector types

//-----------------------------------------------------------------------------
