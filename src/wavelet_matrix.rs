//! An immutable integer vector supporting rank/select-type queries.
//!
//! The wavelet matrix was first described in:
//!
//! > Claude, Navarro, Ordóñez: The wavelet matrix: An efficient wavelet tree for large alphabets.  
//! > Information Systems, 2015.  
//! > DOI: [10.1016/j.is.2014.06.002](https://doi.org/10.1016/j.is.2014.06.002)
//!
//! The structure can be understood as the positional BWT (PBWT) of the binary sequences corresponding to the integers.
//! Each level in the wavelet matrix corresponds to a column in the PBWT.
//! Bitvector `bv[level]` on level `level` represent bit values
//!
//! > `1 << (width - 1 - level)`.
//!
//! If `bv[level][i] == 0`, position `i` on level `level` it maps to position
//!
//! > `bv[level].rank_zero(i)`
//!
//! on level `level + 1`.
//! Otherwise it maps to position
//!
//! > `bv[level].count_zeros() + bv[level].rank(i)`.
//!
//! As in wavelet trees, access and rank queries proceed down from level `0`, while select queries go up from level `width - 1`.

use crate::bit_vector::BitVector;
use crate::int_vector::IntVector;
use crate::ops::{Vector, Access, AccessIter, VectorIndex, Pack, BitVec, Rank, Select, SelectZero, PredSucc};
use crate::raw_vector::{RawVector, PushRaw};
use crate::serialize::Serialize;
use crate::bits;

use std::io::{Read, Write};
use std::iter::FusedIterator;
use std::io;

// FIXME tests

//-----------------------------------------------------------------------------

/// An immutable integer vector supporting rank/select-type queries.
///
/// Each item consists of the lowest 1 to 64 bits of a [`u64`] value, as specified by the width of the vector.
/// The width determines the number of levels in the `WaveletMatrix`.
/// Each level is represented using a [`BitVector`].
/// There is also an [`IntVector`] storing the starting position of each possible item value after the reordering done by the wavelet matrix.
/// Hence a `WaveletMatrix` should only be used when most values in `0..(1 << width)` are in use.
/// The maximum length of the vector is approximately [`usize::MAX`] items.
///
/// A `WaveletMatrix` can be built from a [`Vec`] of unsigned integers using the [`From`] trait.
/// The construction requires several passes over the input and uses the input vector as working space.
/// Using smaller integer types helps reducing the space overhead during construction.
///
/// `WaveletMatrix` implements the following `simple_sds` traits:
/// * Basic functionality: [`Vector`], [`Access`]
/// * Queries and operations: [`VectorIndex`]
/// * Serialization: [`Serialize`]
///
/// Overridden default implementations:
/// * [`VectorIndex::has_item`] has a simple constant-time implementation.
/// * [`VectorIndex::inverse_select`] is effectively the same as [`Access::get`].
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{Vector, Access, VectorIndex};
/// use simple_sds::wavelet_matrix::WaveletMatrix;
///
/// // Construction
/// let source: Vec<u64> = vec![1, 0, 3, 1, 1, 2, 4, 5, 1, 2, 1, 7, 0, 1];
/// let wm = WaveletMatrix::from(source.clone());
///
/// // Access
/// assert_eq!(wm.len(), source.len());
/// assert_eq!(wm.width(), 3);
/// for i in 0..wm.len() {
///     assert_eq!(wm.get(i), source[i]);
/// }
/// assert!(wm.iter().eq(source.iter().cloned()));
///
/// // Rank
/// assert_eq!(wm.rank(5, 3), 1);
/// assert_eq!(wm.rank(10, 2), 2);
///
/// // Select
/// assert_eq!(wm.select(2, 1), Some(4));
/// assert!(wm.select(1, 7).is_none());
/// assert_eq!(wm.select_iter(1, 2).next(), Some((1, 9)));
///
/// // Inverse select
/// let index = 7;
/// let inverse = wm.inverse_select(index).unwrap();
/// assert_eq!(inverse, (0, 5));
/// assert_eq!(wm.select(inverse.0, inverse.1), Some(index));
///
/// // Predecessor / successor
/// assert!(wm.predecessor(1, 3).next().is_none());
/// assert_eq!(wm.predecessor(2, 3).next(), Some((0, 2)));
/// assert_eq!(wm.successor(12, 0).next(), Some((1, 12)));
/// assert!(wm.successor(13, 0).next().is_none());
/// ```
///
/// # Notes
///
/// * `WaveletMatrix` never panics from I/O errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaveletMatrix {
    len: usize,
    data: Vec<BitVector>,
    // Starting offset of each value after reordering by the wavelet matrix, or `len` if the value does not exist.
    first: IntVector,
}

impl WaveletMatrix {
    // Returns the starting offset of the value after reordering.
    fn start(&self, value: <Self as Vector>::Item) -> usize {
        self.first.get(value as usize) as usize
    }

    // Returns the bit value for the given level.
    fn bit_value(&self, level: usize) -> u64 {
        1 << (self.width() - 1 - level)
    }

    // Maps the index to the next level with a set bit.
    fn map_down_one(&self, index: usize, level: usize) -> usize {
        self.data[level].count_zeros() + self.data[level].rank(index)
    }

    // Maps the index to the next level with an unset bit.
    fn map_down_zero(&self, index: usize, level: usize) -> usize {
        self.data[level].rank_zero(index)
    }

    // Maps the index from the next level with a set bit.
    fn map_up_one(&self, index: usize, level: usize) -> Option<usize> {
        self.data[level].select(index - self.data[level].count_zeros())
    }

    // Maps the index from the next level with an unset bit.
    fn map_up_zero(&self, index: usize, level: usize) -> Option<usize> {
        self.data[level].select_zero(index)
    }

    // Initializes the support structures for the bitvectors.
    fn init_support(&mut self) {
        for bv in self.data.iter_mut() {
            bv.enable_rank();
            bv.enable_select();
            bv.enable_select_zero();
            bv.enable_pred_succ();
        }
    }

    // Computes `start` from an iterator.
    fn start_offsets<Iter: Iterator<Item = u64>>(iter: Iter, len: usize, max_value: u64) -> IntVector {
        // Count the number of occurrences of each value.
        let mut counts: Vec<(u64, usize)> = Vec::with_capacity((max_value + 1) as usize);
        for i in 0..=max_value {
            counts.push((i, 0));
        }
        for value in iter {
            counts[value as usize].1 += 1;
        }

        // Sort the counts in reverse bit order to get the order below the final level.
        counts.sort_unstable_by_key(|(value, _)| value.reverse_bits());

        // Replace occurrence counts with the prefix sum in the sorted order.
        let mut cumulative = 0;
        for (_, count) in counts.iter_mut() {
            if *count == 0 {
                *count = len;
            } else {
                let increment = *count;
                *count = cumulative;
                cumulative += increment;
            }
        }

        // Sort the prefix sums by symbol to get starting offsets and then return the offsets.
        counts.sort_unstable_by_key(|(value, _)| *value);
        let mut result: IntVector = counts.into_iter().map(|(_, offset)| offset).collect();
        result.pack();
        result
    }
}

macro_rules! wavelet_matrix_from {
    ($t:ident) => {
        impl From<Vec<$t>> for WaveletMatrix {
            fn from(source: Vec<$t>) -> Self {
                let mut source = source;
                let max_value = source.iter().cloned().max().unwrap_or(0);
                let width = bits::bit_len(max_value as u64);

                let first = Self::start_offsets(source.iter().map(|x| *x as u64), source.len(), max_value as u64);

                let mut data: Vec<BitVector> = Vec::new();
                for level in 0..width {
                    let bit_value: $t = 1 << (width - 1 - level);
                    let mut zeros: Vec<$t> = Vec::new();
                    let mut ones: Vec<$t> = Vec::new();
                    let mut raw_data = RawVector::with_capacity(source.len());

                    // Determine if the current bit is set in each value.
                    for value in source.iter() {
                        if value & bit_value != 0 {
                            ones.push(*value);
                            raw_data.push_bit(true);
                        } else {
                            zeros.push(*value);
                            raw_data.push_bit(false);
                        }
                    }

                    // Sort the values stably by the current bit.
                    source.clear();
                    source.extend(zeros);
                    source.extend(ones);
        
                    // Create the bitvector for the current level.
                    data.push(BitVector::from(raw_data));
                }

                let mut result = WaveletMatrix { len: source.len(), data, first, };
                result.init_support();
                result
            }
        }
    }
}

wavelet_matrix_from!(u8);
wavelet_matrix_from!(u16);
wavelet_matrix_from!(u32);
wavelet_matrix_from!(u64);
wavelet_matrix_from!(usize);

//-----------------------------------------------------------------------------

impl Vector for WaveletMatrix {
    type Item = u64;

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn width(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn max_len(&self) -> usize {
        usize::MAX
    }
}

impl<'a> Access<'a> for WaveletMatrix {
    type Iter = AccessIter<'a, Self>;

    fn get(&self, index: usize) -> <Self as Vector>::Item {
        self.inverse_select(index).unwrap().1
    }

    fn iter(&'a self) -> Self::Iter {
        Self::Iter::new(self)
    }
}

impl<'a> VectorIndex<'a> for WaveletMatrix {
    type ValueIter = ValueIter<'a>;

    fn has_item(&self, value: <Self as Vector>::Item) -> bool {
        (value as usize) < self.first.len() && self.start(value) < self.len()
    }

    fn rank(&self, index: usize, value: <Self as Vector>::Item) -> usize {
        if !self.has_item(value) {
            return 0;
        }

        let mut index = index;
        for level in 0..self.width() {
            if value & self.bit_value(level) != 0 {
                index = self.map_down_one(index, level);
            } else {
                index = self.map_down_zero(index, level);
            }
        }

        index - self.start(value)
    }

    fn inverse_select(&self, index: usize) -> Option<(usize, <Self as Vector>::Item)> {
        if index >= self.len() {
            return None;
        }

        let mut index = index;
        let mut value = 0;
        for level in 0..self.width() {
            if self.data[level].get(index) {
                index = self.map_down_one(index, level);
                value += self.bit_value(level);
            } else {
                index = self.map_down_zero(index, level);
            }
        }

        Some((index - self.start(value), value))
    }

    fn value_iter(&'a self, value: <Self as Vector>::Item) -> Self::ValueIter {
        Self::ValueIter {
            parent: self,
            value,
            rank: 0,
        }
    }

    fn value_of(iter: &Self::ValueIter) -> <Self as Vector>::Item {
        iter.value
    }

    fn select(&self, rank: usize, value: <Self as Vector>::Item) -> Option<usize> {
        if !self.has_item(value) {
            return None;
        }

        let mut index = self.start(value) + rank;
        for level in (0..self.width()).rev() {
            if value & self.bit_value(level) != 0 {
                index = self.map_up_one(index, level)?;
            } else {
                index = self.map_up_zero(index, level)?;
            }
        }

        Some(index)
    }

    fn select_iter(&'a self, rank: usize, value: <Self as Vector>::Item) -> Self::ValueIter {
        Self::ValueIter {
            parent: self,
            value,
            rank,
        }
    }
}

impl Serialize for WaveletMatrix {
    fn serialize_header<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        self.len.serialize(writer)
    }

    fn serialize_body<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        let width = self.data.len();
        width.serialize(writer)?;
        for bv in self.data.iter() {
            bv.serialize(writer)?;
        }
        self.first.serialize(writer)?;
        Ok(())
    }

    fn load<T: Read>(reader: &mut T) -> io::Result<Self> {
        let len = usize::load(reader)?;
        let width = usize::load(reader)?;
        let mut data: Vec<BitVector> = Vec::new();
        for _ in 0..width {
            let bv = BitVector::load(reader)?;
            data.push(bv);
        }
        let first = IntVector::load(reader)?;
        let mut result = WaveletMatrix { len, data, first, };
        result.init_support();
        Ok(result)
    }

    fn size_in_elements(&self) -> usize {
        let mut result = self.len.size_in_elements();
        result += 1; // Width.
        for bv in self.data.iter() {
            result += bv.size_in_elements();
        }
        result += self.first.size_in_elements();
        result
    }
}

//-----------------------------------------------------------------------------

/// A read-only iterator over the occurrences of a specific value in [`WaveletMatrix`].
///
/// The type of `Item` is [`(usize, usize)`] representing a pair (rank, index).
/// The item at position `index` has the given value, and the rank of that value at that position is `rank`.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::VectorIndex;
/// use simple_sds::wavelet_matrix::WaveletMatrix;
///
/// // Construction
/// let source: Vec<u64> = vec![1, 0, 3, 1, 1, 2, 4, 5, 1, 2, 1, 7, 0, 1];
/// let wm = WaveletMatrix::from(source.clone());
///
/// // Iteration over values
/// let mut iter = wm.value_iter(2);
/// assert_eq!(WaveletMatrix::value_of(&iter), 2);
/// assert_eq!(iter.next(), Some((0, 5)));
/// assert_eq!(iter.next(), Some((1, 9)));
/// assert!(iter.next().is_none());
/// ```
#[derive(Clone, Debug)]
pub struct ValueIter<'a> {
    parent: &'a WaveletMatrix,
    value: <WaveletMatrix as Vector>::Item,
    // The first rank we have not seen.
    rank: usize,
}

impl<'a> Iterator for ValueIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.rank >= self.parent.len() {
            return None;
        }
        if let Some(index) = self.parent.select(self.rank, self.value) {
            let result = Some((self.rank, index));
            self.rank += 1;
            result
        } else {
            self.rank = self.parent.len();
            None
        }
    }
}

impl<'a> FusedIterator for ValueIter<'a> {}

//-----------------------------------------------------------------------------

// FIXME IntoIter

//-----------------------------------------------------------------------------
