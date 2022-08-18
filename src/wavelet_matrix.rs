//! An immutable integer vector supporting rank/select-type queries.
//!
//! The wavelet matrix was first described in:
//!
//! > Claude, Navarro, Ordóñez: The wavelet matrix: An efficient wavelet tree for large alphabets.  
//! > Information Systems, 2015.  
//! > DOI: [10.1016/j.is.2014.06.002](https://doi.org/10.1016/j.is.2014.06.002)
//!
//! See [`wm_core`] for a low-level description.
//! As in wavelet trees, access and rank queries proceed down from level `0`, while select queries go up from level `width - 1`.

use crate::int_vector::IntVector;
use crate::ops::{Vector, Access, AccessIter, VectorIndex, Pack};
use crate::serialize::Serialize;
use crate::wavelet_matrix::wm_core::WMCore;

use std::io::{Error, ErrorKind};
use std::iter::FusedIterator;
use std::io;

pub mod wm_core;

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

/// An immutable integer vector supporting rank/select-type queries.
///
/// Each item consists of the lowest 1 to 64 bits of a [`u64`] value, as specified by the width of the vector.
/// The vector is represented using [`WMCore`].
/// There is also an [`IntVector`] storing the starting position of each possible item value after the reordering done by the core.
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
/// * [`VectorIndex::contains`] has a simple constant-time implementation.
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
    data: WMCore,
    // Starting offset of each value after reordering by the wavelet matrix, or `len` if the value does not exist.
    first: IntVector,
}

impl WaveletMatrix {
    // Returns the starting offset of the value after reordering.
    fn start(&self, value: <Self as Vector>::Item) -> usize {
        self.first.get(value as usize) as usize
    }

    // Computes `first` from an iterator.
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
                let len = source.len();
                let max_value = source.iter().cloned().max().unwrap_or(0);
                let first = Self::start_offsets(source.iter().map(|x| *x as u64), source.len(), max_value as u64);
                let data = WMCore::from(source);
                WaveletMatrix { len, data, first, }
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
        self.data.width()
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

    fn contains(&self, value: <Self as Vector>::Item) -> bool {
        (value as usize) < self.first.len() && self.start(value) < self.len()
    }

    fn rank(&self, index: usize, value: <Self as Vector>::Item) -> usize {
        if !self.contains(value) {
            return 0;
        }
        self.data.map_down_with(index, value) - self.start(value)
    }

    fn inverse_select(&self, index: usize) -> Option<(usize, <Self as Vector>::Item)> {
        self.data.map_down(index).map(|(index, value)| (index - self.start(value), value))
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
        if !self.contains(value) {
            return None;
        }
        self.data.map_up_with(self.start(value) + rank, value)
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
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.len.serialize(writer)
    }

    fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.data.serialize(writer)?;
        self.first.serialize(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let len = usize::load(reader)?;
        let data = WMCore::load(reader)?;
        if data.len() != len {
            return Err(Error::new(ErrorKind::InvalidData, "Core length does not match wavelet matrix length"));
        }

        let first = IntVector::load(reader)?;
        Ok(WaveletMatrix { len, data, first, })
    }

    fn size_in_elements(&self) -> usize {
        let mut result = self.len.size_in_elements();
        result += self.data.size_in_elements();
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

/// [`WaveletMatrix`] transformed into an iterator.
///
/// The type of `Item` is [`u64`].
///
/// # Examples
///
/// ```
/// use simple_sds::wavelet_matrix::WaveletMatrix;
///
/// let source: Vec<u64> = vec![1, 0, 3, 1, 1, 2, 4, 5, 1, 2, 1, 7, 0, 1];
/// let wm = WaveletMatrix::from(source.clone());
/// let target: Vec<u64> = wm.into_iter().collect();
/// assert_eq!(target, source);
/// ```
#[derive(Clone, Debug)]
pub struct IntoIter {
    parent: WaveletMatrix,
    index: usize,
}

impl Iterator for IntoIter {
    type Item = <WaveletMatrix as Vector>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.parent.len() {
            None
        } else {
            let result = Some(self.parent.get(self.index));
            self.index += 1;
            result
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.parent.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IntoIter {}

impl FusedIterator for IntoIter {}

impl IntoIterator for WaveletMatrix {
    type Item = <Self as Vector>::Item;
    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            parent: self,
            index: 0,
        }
    }
}

//-----------------------------------------------------------------------------
