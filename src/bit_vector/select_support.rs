//! Select queries on plain bitvectors.
//!
//! The structure is the same as `select_support_mcl` in [SDSL](https://github.com/simongog/sdsl-lite):
//!
//! > Gog, Petri: Optimized succinct data structures for massive data.  
//! > Software: Practice and Experience, 2014.  
//! > DOI: [10.1002/spe.2198](https://doi.org/10.1002/spe.2198)
//!
//! This is a simplified version of the original three-level structure:
//!
//! > Clark: Compact Pat Trees.  
//! > Ph.D. Thesis (Section 2.2.2), University of Waterloo, 1996.  
//! > [http://www.nlc-bnc.ca/obj/s4/f2/dsk3/ftp04/nq21335.pdf](http://www.nlc-bnc.ca/obj/s4/f2/dsk3/ftp04/nq21335.pdf)
//!
//! We divide the integer array into superblocks of 2^12 = 4096 values.
//! For each superblock, we sample the first value.
//! Let `x` and `y` be two consecutive superblock samples and let `u` be the universe size of the integer array.
//! There are now two cases:
//!
//! 1. The superblock is long: `y - x >= log^4 u`.
//!    We can store all values in the superblock explicitly.
//! 2. The superblock is short.
//!    We divide the superblock into blocks of 2^6 = 64 values.
//!    For each block, we sample the first value relative to the superblock sample.
//!    We then search for the value in the bit array one 64-bit word at a time.
//!
//! The space overhead is 18.75% in the worst case.

use crate::bit_vector::BitVector;
use crate::int_vector::IntVector;
use crate::ops::{Element, Resize, Pack, BitVec};
use crate::serialize::Serialize;
use crate::bits;

use std::io;

//-----------------------------------------------------------------------------

/// Select support structure for plain bitvectors.
///
/// The structure depends on the parent bitvector and assumes that the parent remains unchanged.
/// Using the [`BitVector`] interface is usually more convenient.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectSupport {
    // (superblock sample, 2 * index + is_short) for each superblock.
    samples: IntVector,

    // If the superblock is long, the index from the superblock array points to the
    // first value in this array.
    long: IntVector,

    // If the superblock is short, the index from the superblock array points to the
    // first block sample in this array.
    short: IntVector,
}

impl SelectSupport {
    const SUPERBLOCK_SIZE: usize = 4096;
    const BLOCKS_IN_SUPERBLOCK: usize = 64;

    /// Returns the number superblocks in the bitvector.
    pub fn superblocks(&self) -> usize {
        self.samples.len() / 2
    }

    /// Returns the number of long superblocks in the bitvector.
    pub fn long_superblocks(&self) -> usize {
        self.long.len() / Self::SUPERBLOCK_SIZE
    }

    /// Returns the number of short superblocks in the bitvector.
    pub fn short_superblocks(&self) -> usize {
        self.short.len() / Self::BLOCKS_IN_SUPERBLOCK
    }

    // FIXME document, example
    pub fn new(parent: &BitVector) -> SelectSupport {
//        let words = bits::bits_to_words(parent.len());
        let superblocks = (parent.len() + Self::SUPERBLOCK_SIZE - 1) / Self::SUPERBLOCK_SIZE;
        let mut samples = IntVector::default(); samples.reserve(superblocks * 2);
        let mut long = IntVector::default();
        let mut short = IntVector::default();

        // FIXME build using OneIter. collect SUPERBLOCK_SIZE + 1 ones at a time

        let samples = samples.pack();
        let long = long.pack();
        let short = short.pack();
        SelectSupport {
            samples: samples,
            long: long,
            short: short,
        }
    }

    // FIXME select()
}

//-----------------------------------------------------------------------------

impl Serialize for SelectSupport {
    fn serialize_header<T: io::Write>(&self, _: &mut T) -> io::Result<()> {
        Ok(())
    }

    fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.samples.serialize(writer)?;
        self.long.serialize(writer)?;
        self.short.serialize(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let samples = IntVector::load(reader)?;
        let long = IntVector::load(reader)?;
        let short = IntVector::load(reader)?;
        Ok(SelectSupport {
            samples: samples,
            long: long,
            short: short,
        })
    }

    fn size_in_bytes(&self) -> usize {
        self.samples.size_in_bytes() + self.long.size_in_bytes() + self.short.size_in_bytes()
    }
}

//-----------------------------------------------------------------------------

// FIXME tests: empty, non-empty, serialize

//-----------------------------------------------------------------------------
