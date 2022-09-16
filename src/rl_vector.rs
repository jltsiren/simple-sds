// FIXME document
// FIXME document serialization format

use crate::int_vector::IntVector;
use crate::ops::{Vector, Push, Resize, BitVec, Select};
use crate::bits;

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

// FIXME copy_bit_vec
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

// FIXME RunIter

// FIXME Iterator, ExactSizeIterator, FusedIterator

//-----------------------------------------------------------------------------

// FIXME Iter

// FIXME Iterator, ExactSizeIterator, FusedIterator

//-----------------------------------------------------------------------------

// FIXME BitVec

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

// FIXME TryFrom RLBuilder

//-----------------------------------------------------------------------------
