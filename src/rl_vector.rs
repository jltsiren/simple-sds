// FIXME document

use crate::int_vector::IntVector;

// FIXME tests
//#[cfg(test)]
//mod tests;

//-----------------------------------------------------------------------------

// FIXME document, example, tests
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RLVector {
    len: usize,
    ones: usize,
    // FIXME divisors for using rank/select indexes
    rank_index: IntVector,
    select_index: IntVector,
    // (bits, ones) until the start of each block.
    samples: IntVector,
    // Concatenated blocks.
    data: IntVector,
}

// FIXME impl
impl RLVector {
    /// Number of bits in a code unit.
    pub const CODE_SIZE: usize = 4;

    /// Number of code units in a block.
    pub const BLOCK_SIZE: usize = 64;
}

//-----------------------------------------------------------------------------

// FIXME document, example, tests
#[derive(Clone, Debug)]
pub struct RLBuilder {
    len: usize,
    ones: usize,
    // Position after the last run.
    tail: usize,
    run: (usize, usize),
    samples: Vec<(usize, usize)>,
    data: IntVector,
}

// FIXME document
impl RLBuilder {
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

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn count_ones(&self) -> usize {
        self.ones
    }

    pub fn tail(&self) -> usize {
        self.tail
    }

    // FIXME set
    // FIXME set_run
    // FIXME unchecked versions?
    // FIXME set_len

    // FIXME flush
}

// FIXME extend

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
