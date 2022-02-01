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

use crate::bit_vector::{BitVector, Transformation};
use crate::int_vector::IntVector;
use crate::ops::{Vector, Resize, Pack, Access, Push, BitVec};
use crate::serialize::Serialize;
use crate::bits;

use std::io::{Error, ErrorKind};
use std::{io, marker};

//-----------------------------------------------------------------------------

/// Select support structure for plain bitvectors.
///
/// The structure depends on the parent bitvector and assumes that the parent remains unchanged.
/// Using the [`BitVector`] interface is usually more convenient.
///
/// This type must be parametrized with a [`Transformation`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectSupport<T: Transformation> {
    // (superblock sample, 2 * index + is_short) for each superblock.
    samples: IntVector,

    // If the superblock is long, the index from the superblock array points to the
    // first value in this array.
    long: IntVector,

    // If the superblock is short, the index from the superblock array points to the
    // first block sample in this array.
    short: IntVector,

    // We use `T` only for accessing static methods.
    _marker: marker::PhantomData<T>,
}

impl<T: Transformation> SelectSupport<T> {
    /// Number of ones per superblock (4096).
    pub const SUPERBLOCK_SIZE: usize = 4096;

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

    /// Builds a select support structure for the parent bitvector.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::bit_vector::{BitVector, Identity};
    /// use simple_sds::bit_vector::select_support::SelectSupport;
    ///
    /// let mut data = vec![false, true, true, false, true, false, true, true, false, false, false];
    /// let bv: BitVector = data.into_iter().collect();
    /// let ss = SelectSupport::<Identity>::new(&bv);
    /// assert_eq!(ss.superblocks(), 1);
    /// assert_eq!(ss.long_superblocks(), 0);
    /// assert_eq!(ss.short_superblocks(), 1);
    /// ```
    pub fn new(parent: &BitVector) -> SelectSupport<T> {
        let superblocks = (T::count_ones(parent) + Self::SUPERBLOCK_SIZE - 1) / Self::SUPERBLOCK_SIZE;
        let log4 = bits::bit_len(parent.len() as u64);
        let log4 = log4 * log4;
        let log4 = log4 * log4;

        let mut result = SelectSupport {
            samples: IntVector::default(),
            long: IntVector::default(),
            short: IntVector::default(),
            _marker: marker::PhantomData,
        };
        result.samples.reserve(superblocks * 2);

        // `sample_iter` will iterate over superblock samples.
        let mut sample_iter = T::one_iter(parent);
        let mut sample = sample_iter.next();

        // `iter` will iterate over the values we store in `self.long` and `self.short`.
        let mut iter = T::one_iter(parent);
        let mut value = iter.next();

        while sample != None {
            let start = sample.unwrap();
            let next_sample = sample_iter.nth(Self::SUPERBLOCK_SIZE - 1);
            let limit = match next_sample {
                Some(v) => v,
                None => (T::count_ones(parent), parent.len()),
            };
            result.samples.push(start.1 as u64);
            // Long superblock.
            if limit.1 - start.1 >= log4 {
                result.samples.push((2 * result.long.len()) as u64);
                let values = limit.0 - start.0;
                for _ in 0..values {
                    result.long.push((value.unwrap().1 - start.1) as u64);
                    value = iter.next();
                }
            }
            // Short superblock.
            else {
                result.samples.push((2 * result.short.len() + 1) as u64);
                let blocks = ((limit.0 - start.0) + Self::BLOCK_SIZE - 1) / Self::BLOCK_SIZE;
                for _ in 0..blocks {
                    result.short.push((value.unwrap().1 - start.1) as u64);
                    value = iter.nth(Self::BLOCK_SIZE - 1);
                }
            }
            sample = next_sample;
        }

        result.samples.pack();
        result.long.pack();
        result.short.pack();
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
    /// use simple_sds::bit_vector::{BitVector, Identity};
    /// use simple_sds::bit_vector::select_support::SelectSupport;
    ///
    /// let mut data = vec![false, true, true, false, true, false, true, true, false, false, false];
    /// let bv: BitVector = data.into_iter().collect();
    /// let ss = SelectSupport::<Identity>::new(&bv);
    /// assert_eq!(ss.select(&bv, 0), 1);
    /// assert_eq!(ss.select(&bv, 1), 2);
    /// assert_eq!(ss.select(&bv, 4), 7);
    /// ```
    ///
    /// # Panics
    ///
    /// May panic if `rank >= T::count_ones(parent)`.
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
                let mut value: u64 = T::word(parent, word) & unsafe { !bits::low_set_unchecked(word_offset) };
                loop {
                    let ones = value.count_ones() as usize;
                    if ones > relative_rank {
                        result = bits::bit_offset(word, unsafe { bits::select(value, relative_rank) });
                        break;
                    }
                    relative_rank -= ones;
                    word += 1;
                    value = T::word(parent, word);
                }
            }
        }

        result
    }

    /// Unsafe version of [`SelectSupport::select`] without some bounds checks.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if `rank >= T::count_ones(parent)`.
    pub unsafe fn select_unchecked(&self, parent: &BitVector, rank: usize) -> usize {
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
                let mut value: u64 = T::word_unchecked(parent, word) & !bits::low_set_unchecked(word_offset);
                loop {
                    let ones = value.count_ones() as usize;
                    if ones > relative_rank {
                        result = bits::bit_offset(word, bits::select(value, relative_rank));
                        break;
                    }
                    relative_rank -= ones;
                    word += 1;
                    value = T::word_unchecked(parent, word);
                }
            }
        }

        result
    }
}

//-----------------------------------------------------------------------------

impl<T: Transformation> Serialize for SelectSupport<T> {
    fn serialize_header<W: io::Write>(&self, _: &mut W) -> io::Result<()> {
        Ok(())
    }

    fn serialize_body<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.samples.serialize(writer)?;
        self.long.serialize(writer)?;
        self.short.serialize(writer)?;
        Ok(())
    }

    fn load<W: io::Read>(reader: &mut W) -> io::Result<Self> {
        let samples = IntVector::load(reader)?;
        let long = IntVector::load(reader)?;
        let short = IntVector::load(reader)?;
        let result = SelectSupport {
            samples,
            long,
            short,
            _marker: marker::PhantomData,
        };
        if result.superblocks() != result.long_superblocks() + result.short_superblocks() {
            Err(Error::new(ErrorKind::InvalidData, "Invalid long/short superblock counts"))
        }
        else {
            Ok(result)
        }
    }

    fn size_in_elements(&self) -> usize {
        self.samples.size_in_elements() + self.long.size_in_elements() + self.short.size_in_elements()
    }
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_vector::{BitVector, Identity};
    use crate::ops::BitVec;
    use crate::raw_vector::{RawVector, PushRaw};
    use crate::serialize;
    use rand::distributions::{Bernoulli, Distribution};

    #[test]
    fn empty_vector() {
        let bv = BitVector::from(RawVector::new());
        let ss = SelectSupport::<Identity>::new(&bv);
        assert_eq!(ss.superblocks(), 0, "Non-zero select superblocks for empty vector");
        assert_eq!(ss.long_superblocks(), 0, "Non-zero long superblocks for empty vector");
        assert_eq!(ss.short_superblocks(), 0, "Non-zero short superblocks for empty vector");
    }

    fn with_density(len: usize, density: f64) -> RawVector {
        let mut data = RawVector::with_capacity(len);
        let mut rng = rand::thread_rng();
        let dist = Bernoulli::new(density).unwrap();
        let mut iter = dist.sample_iter(&mut rng);
        while data.len() < len {
            data.push_bit(iter.next().unwrap());
        }
        assert_eq!(data.len(), len, "Invalid length for random RawVector");
        data
    }

    fn test_vector(len: usize, density: f64) {
        let data = with_density(len, density);
        let bv = BitVector::from(data.clone());
        let ss = SelectSupport::<Identity>::new(&bv);
        assert_eq!(bv.len(), len, "test_vector({}, {}): invalid bitvector length", len, density);

        let superblocks = ss.superblocks();
        let long = ss.long_superblocks();
        let short = ss.short_superblocks();
        assert_eq!(superblocks, long + short, "test_vector({}, {}): block counts do not match", len, density);

        // This test assumes that the number of ones is within 6 stdevs of the expected.
        let ones: f64 = bv.count_ones() as f64;
        let expected: f64 = len as f64 * density;
        let stdev: f64 = (len as f64 * density * (1.0 - density)).sqrt();
        assert!(ones >= expected - 6.0 * stdev && ones <= expected + 6.0 * stdev,
            "test_vector({}, {}): unexpected number of ones: {}", len, density, ones);

        let mut next: usize = 0;
        for i in 0..bv.count_ones() {
            let value = ss.select(&bv, i);
            assert!(value >= next, "test_vector({}, {}): select({}) == {}, expected at least {}", len, density, i, value, next);
            assert!(bv.get(value), "test_vector({}, {}): select({}) == {} is not set", len, density, i, value);
            next = value + 1;
        }

        unsafe {
            let mut next: usize = 0;
            for i in 0..bv.count_ones() {
                let value = ss.select_unchecked(&bv, i);
                assert!(value >= next, "test_vector({}, {}): select_unchecked({}) == {}, expected at least {}", len, density, i, value, next);
                assert!(bv.get(value), "test_vector({}, {}): select_unchecked({}) == {} is not set", len, density, i, value);
                next = value + 1;
            }
        }
    }

    #[test]
    fn non_empty_vector() {
        test_vector(8131, 0.1);
        test_vector(8192, 0.5);
        test_vector(8266, 0.9);
    }

    #[test]
    fn serialize() {
        let data = with_density(5187, 0.5);
        let bv = BitVector::from(data);
        let original = SelectSupport::<Identity>::new(&bv);
        let _ = serialize::test(&original, "select-support", None, true);
    }
}

//-----------------------------------------------------------------------------
