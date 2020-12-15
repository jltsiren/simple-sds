//! Rank queries on plain bitvectors.
//!
//! The structure is the same as `rank_support_v` in [SDSL](https://github.com/simongog/sdsl-lite):
//!
//! > Gog, Petri: Optimized succinct data structures for massive data.
//! > Software: Practice and Experience, 2014.
//! > [DOI: 10.1002/spe.2198](https://doi.org/10.1002/spe.2198)
//!
//! The original version is called rank9:
//!
//! > Vigna: Broadword Implementation of Rank/Select Queries. WEA 2008.
//! > [DOI: 10.1007/978-3-540-68552-4_12](https://doi.org/10.1007/978-3-540-68552-4_12)
//!
//! We divide the bitvector into blocks of 2^9 = 512 bits.
//! Each block is further divided into 8 words of 64 bits each.
//! For each block, we store two 64-bit integers:
//!
//! * The first stores the number of ones in previous blocks.
//! * The second stores, for each word except the first, the number of ones in previous words using 9 bits per word.
//!
//! The space overhead is 25%.

use crate::bit_vector::BitVector;
use crate::raw_vector::GetRaw;
//use crate::serialize::Serialize;
use crate::bits;

//use std::io;

//-----------------------------------------------------------------------------

/// Rank support structure for plain bitvectors.
///
/// It is usually more convenient to use this through [`BitVector`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankSupport<'a> {
    parent: &'a BitVector,
    // No RawVector or bits::read_int because we want to avoid branching.
    samples: Vec<u64>,
}

//-----------------------------------------------------------------------------

// FIXME document, examples

impl<'a> RankSupport<'a> {

    const BLOCK_SIZE: usize = 512;
    const RELATIVE_RANK_BITS: usize = 9;
    const RELATIVE_RANK_MASK: usize = 0x1FF;
    const WORDS_PER_BLOCK: usize = 8;
    const WORD_MASK: usize = 0x7;

    pub fn new(parent: &BitVector) -> RankSupport {
        let samples: Vec<u64> = Vec::new();
        // FIXME build samples
        RankSupport {
            parent: parent,
            samples: samples,
        }
    }

    pub fn rank(&self, index: usize) -> usize {
        let block = index / Self::BLOCK_SIZE;
        let (word, offset) = bits::split_offset(index);

        // Rank at the start of the block.
        let block_start = self.samples[2 * block] as usize;

        // Transform the absolute word index into a relative word index within the block.
        // Then reorder the words 0..8 to 1..8, 0, because the second sample stores relative
        // ranks for words 1..8.
        let relative = ((word & Self::WORD_MASK) + Self::WORDS_PER_BLOCK - 1) & Self::WORD_MASK;

        // Relative rank at the start of the word.
        let word_start = (self.samples[2 * block + 1] >> (relative * Self::RELATIVE_RANK_BITS)) as usize & Self::RELATIVE_RANK_MASK;

        // Relative rank within the word.
        let within_word = (self.parent.data.word(word) & bits::low_set(offset)).count_ones() as usize;

        block_start + word_start + within_word
    }
}

//-----------------------------------------------------------------------------

// FIXME impl: SerializeSupport
// FIXME we need a variant where we provide the parent reference during loading

//-----------------------------------------------------------------------------

// FIXME small tests. large ones are in the tests module for bit_vector

//-----------------------------------------------------------------------------

