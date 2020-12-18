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
use crate::ops::{Element, Resize, Pack, Access, Push, BitVec, Select};
use crate::raw_vector::GetRaw;
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
    const SUPERBLOCK_MASK: usize = 0xFFF;
    const BLOCKS_IN_SUPERBLOCK: usize = 64;
    const BLOCK_SIZE: usize = 64;
    const BLOCK_MASK: usize = 0x3F;

    /// Returns the number superblocks in the bitvector.
    pub fn superblocks(&self) -> usize {
        self.samples.len() / 2
    }

    /// Returns the number of long superblocks in the bitvector.
    pub fn long_superblocks(&self) -> usize {
        (self.long.len() + Self::SUPERBLOCK_SIZE - 1) / Self::SUPERBLOCK_SIZE
    }

    /// Returns the number of short superblocks in the bitvector.
    pub fn short_superblocks(&self) -> usize {
        (self.short.len() + Self::BLOCKS_IN_SUPERBLOCK - 1) / Self::BLOCKS_IN_SUPERBLOCK
    }

    // Append a superblock. The buffer should contain all elements in the superblock
    // and either the first element in the next superblock or a past-the-end sentinel
    // for the last superblock.
    fn add_superblock(&mut self, buf: &[u64], long_superblock_min: usize) {
        let len: usize = (buf[buf.len() - 1] - buf[0]) as usize;
        let superblock_ptr: u64;
        if len >= long_superblock_min {
            superblock_ptr = (2 * self.long.len()) as u64;
            for (index, value) in buf.iter().enumerate() {
                if index + 1 < buf.len() {
                    self.long.push(*value);
                }
            }
        }
        else {
            superblock_ptr = (2 * self.short.len() + 1) as u64;
            for (index, value) in buf.iter().enumerate() {
                if index + 1 < buf.len() && (index & Self::BLOCK_MASK) == 0 {
                    self.short.push(value - buf[0]);
                }
            }
        }
        self.samples.push(buf[0]);
        self.samples.push(superblock_ptr);
    }

    /// Builds a select support structure for the parent bitvector.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::bit_vector::BitVector;
    /// use simple_sds::bit_vector::select_support::SelectSupport;
    ///
    /// let mut data = vec![false, true, true, false, true, false, true, true, false, false, false];
    /// let bv: BitVector = data.into_iter().collect();
    /// let ss = SelectSupport::new(&bv);
    /// assert_eq!(ss.superblocks(), 1);
    /// assert_eq!(ss.long_superblocks(), 0);
    /// assert_eq!(ss.short_superblocks(), 1);
    /// ```
    pub fn new(parent: &BitVector) -> SelectSupport {
        let superblocks = (parent.count_ones() + Self::SUPERBLOCK_SIZE - 1) / Self::SUPERBLOCK_SIZE;
        let log4 = bits::bit_len(parent.len() as u64);
        let log4 = log4 * log4;
        let log4 = log4 * log4;

        let mut result = SelectSupport {
            samples: IntVector::default(),
            long: IntVector::default(),
            short: IntVector::default(),
        };
        result.samples.reserve(superblocks * 2);

        // The buffer will hold one superblock and a sentinel value from the next superblock.
        let mut buf: Vec<u64> = Vec::with_capacity(Self::SUPERBLOCK_SIZE + 1);
        for (_, value) in parent.one_iter() {
            buf.push(value as u64);
            if buf.len() > Self::SUPERBLOCK_SIZE {
                result.add_superblock(&buf, log4);
                buf[0] = buf[Self::SUPERBLOCK_SIZE];
                buf.resize(1, 0);
            }
        }
        if buf.len() > 0 {
            buf.push(parent.len() as u64);
            result.add_superblock(&buf, log4);
        }

        result.samples = result.samples.pack();
        result.long = result.long.pack();
        result.short = result.short.pack();
        result
    }

    /// Returns the value of the specified rank in the parent bitvector.
    ///
    /// # Arguments
    ///
    /// * `parent`: The parent bitvector.
    /// * `rank`: Index in the integer array or rank of a set bit in the bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::bit_vector::BitVector;
    /// use simple_sds::bit_vector::select_support::SelectSupport;
    ///
    /// let mut data = vec![false, true, true, false, true, false, true, true, false, false, false];
    /// let bv: BitVector = data.into_iter().collect();
    /// let ss = SelectSupport::new(&bv);
    /// assert_eq!(ss.select(&bv, 0), 1);
    /// assert_eq!(ss.select(&bv, 1), 2);
    /// assert_eq!(ss.select(&bv, 4), 7);
    /// ```
    ///
    /// # Panics
    ///
    /// May panic if `rank >= parent.count_ones()`.
    pub fn select(&self, parent: &BitVector, rank: usize) -> usize {
        let (superblock, offset) = (rank / Self::SUPERBLOCK_SIZE, rank & Self::SUPERBLOCK_MASK);
        let mut result: usize = self.samples.get(2 * superblock) as usize;
        if offset == 0 {
            return result;
        }

        let ptr = self.samples.get(2 * superblock + 1) as usize;
        let (ptr, is_short) = (ptr / 2, ptr & 1);
        if is_short == 0 {
            result += self.long.get(ptr + offset) as usize;
        } else {
            let (block, mut relative_rank) = (offset / Self::BLOCK_SIZE, offset & Self::BLOCK_MASK);
            result += self.short.get(ptr + block) as usize;
            // Search within the block until we find the set bit of relative rank `relative_rank`
            // from the start of the current word.
            if relative_rank > 0 {
                let (mut word, word_offset) = bits::split_offset(result);
                let mut value: u64 = parent.data.word(word) & !bits::low_set(word_offset);
                loop {
                    let ones = value.count_ones() as usize;
                    if ones > relative_rank {
                        result = bits::bit_offset(word, bits::select(value, relative_rank));
                        break;
                    }
                    relative_rank -= ones;
                    word += 1;
                    value = parent.data.word(word);
                }
            }
        }

        result
    }
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

// FIXME tests: empty, non-empty, with long blocks, serialize

//-----------------------------------------------------------------------------
