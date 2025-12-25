//! An immutable run-length encoded integer vector supporting rank/select-type queries.
//!
//! The wavelet matrix was first described in:
//!
//! > Claude, Navarro, Ordóñez: The wavelet matrix: An efficient wavelet tree for large alphabets.
//! > Information Systems, 2015.
//! > DOI: [10.1016/j.is.2014.06.002](https://doi.org/10.1016/j.is.2014.06.002)
//!
//! See [`wm_core`] for a low-level description and [`crate::wavelet_matrix`] for a plain variant.
//! As in wavelet trees, access and rank queries proceed down from level `0`, while select queries go up from level `width - 1`.

use crate::int_vector::IntVector;
use crate::ops::{Vector, Access, VectorIndex};
use crate::rl_vector::RLVector;
use crate::serialize::Serialize;
use crate::wavelet_matrix::wm_core::WMCore;
use crate::wavelet_matrix::wm_core;

use std::io::{Error, ErrorKind};
use std::iter::FusedIterator;
use std::{cmp, io};

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

/// An immutable run-length encoded integer vector supporting rank/select-type queries.
///
/// Each item consists of the lowest 1 to 64 bits of a [`u64`] value, as specified by the width of the vector.
/// The vector is represented using [`WMCore`] with [`RLVector`] as the underlying bitvector type.
/// There is also an [`IntVector`] storing the starting position of each possible item value after the reordering done by the core.
/// Hence a `RLWM` should only be used when most values in `0..(1 << width)` are in use.
/// The maximum length of the vector is approximately [`usize::MAX`] items.
///
/// A `RLWM` can be built from a [`Vec`] of (value ([`u64`]), length ([`usize`])) runs using the [`From`] trait.
/// The construction requires several passes over the input and uses the input vector as working space.
///
/// `RLWM` implements the following `simple_sds` traits:
/// * Basic functionality: [`Vector`], [`Access`]
/// * Queries and operations: [`VectorIndex`]
/// * Serialization: [`Serialize`]
///
/// Both iterators ([`AccessIter`] and [`ValueIter`]) support iterating over the runs using `next_run`.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{Vector, Access, VectorIndex};
/// use simple_sds::rlwm::RLWM;
///
/// // Construction
/// let runs: Vec<(u64, usize)> = vec![
///     (1, 3), (2, 2), (3, 4), (1, 2), (0, 1), (2, 3)
/// ];
/// let wm = RLWM::from(runs.clone());
/// let values: Vec<u64> = runs.iter().flat_map(|(value, length)|
///     std::iter::repeat(*value).take(*length)
/// ).collect();
///
/// // Access
/// assert_eq!(wm.len(), values.len());
/// assert_eq!(wm.width(), 2);
/// for i in 0..wm.len() {
///     assert_eq!(wm.get(i), values[i]);
/// }
/// assert!(wm.iter().eq(values.iter().copied()));
///
/// // Access by runs
/// let mut offset = 0;
/// for run in runs.iter() {
///     let (value, length) = wm.get_run(offset);
///     assert_eq!((value, length), *run);
///     offset += length;
/// }
///
/// // Rank
/// assert_eq!(wm.rank(5, 1), 3);
/// assert_eq!(wm.rank(10, 2), 2);
///
/// // Select
/// assert_eq!(wm.select(2, 3), Some(7));
/// assert_eq!(wm.select_run(2, 3), Some((7, 2)));
/// assert!(wm.select(1, 5).is_none());
/// assert_eq!(wm.select_iter(1, 2).next(), Some((1, 4)));
///
/// // Inverse select
/// let index = 7;
/// let inverse = wm.inverse_select(index).unwrap();
/// assert_eq!(inverse, (2, 3));
/// assert_eq!(wm.select(inverse.0, inverse.1), Some(index));
///
/// // Predecessor / successor
/// assert!(wm.predecessor(1, 2).next().is_none());
/// assert_eq!(wm.predecessor(4, 1).next(), Some((2, 2)));
/// assert_eq!(wm.successor(10, 0).next(), Some((0, 11)));
/// assert!(wm.successor(12, 0).next().is_none());
/// ```
///
/// # Notes
///
/// * `RLWM` never panics from I/O errors.
/// * Because `RLWM` uses a separate bitvector for each level, it is not space-efficient for short vectors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RLWM<'a> {
    len: usize,
    data: WMCore<'a, RLVector>,
    // Starting offset of each value after reordering by the wavelet matrix, or `len` if the value does not exist.
    first: IntVector,
}

impl<'a> RLWM<'a> {
    // Returns the starting offset of the value after reordering.
    fn start(&self, value: <Self as Vector>::Item) -> usize {
        self.first.get(value as usize) as usize
    }

    // Computes `first` from an iterator.
    fn start_offsets<Iter: Iterator<Item = (u64, usize)>>(iter: Iter, len: usize, max_value: u64) -> IntVector {
        // Count the number of occurrences of each value.
        let mut counts: Vec<(u64, usize)> = Vec::with_capacity((max_value + 1) as usize);
        for i in 0..=max_value {
            counts.push((i, 0));
        }
        for (value, len) in iter {
            counts[value as usize].1 += len;
        }
        wm_core::counts_to_first(counts, len)
    }
}

impl<'a> From<Vec<(u64, usize)>> for RLWM<'a> {
    fn from(runs: Vec<(u64, usize)>) -> Self {
        let mut len = 0;
        let mut max_value = 0;
        for (value, length) in runs.iter() {
            len += *length;
            if *value > max_value {
                max_value = *value;
            }
        }
        let first = Self::start_offsets(runs.iter().copied(), len, max_value);
        let data = WMCore::from(runs);
        RLWM { len, data, first }
    }
}

//-----------------------------------------------------------------------------

impl<'a> Vector for RLWM<'a> {
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

impl<'a> Access<'a> for RLWM<'a> {
    type Iter = AccessIter<'a>;

    fn get(&self, index: usize) -> <Self as Vector>::Item {
        self.inverse_select(index).unwrap().1
    }

    fn iter(&'a self) -> Self::Iter {
        AccessIter {
            parent: self,
            index: 0,
            limit: self.len(),
            value: 0,
            run_len: 0,
        }
    }
}

impl<'a> VectorIndex<'a> for RLWM<'a> {
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
            rank: 0,
            value,
            index: 0,
            run_len: 0,
        }
    }

    fn value_of(&self, iter: &Self::ValueIter) -> <Self as Vector>::Item {
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
            rank,
            value,
            index: 0,
            run_len: 0,
        }
    }

    // TODO: implement predecessor/successor using the same functionality in RLVector?
}

impl <'a> RLWM<'a> {
    /// Returns the right-maximal run of values starting at the given index.
    ///
    /// The returned tuple is (value, length).
    /// See also [`Access::get`].
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn get_run(&self, index: usize) -> (u64, usize) {
        let result = self.data.map_down_with_run(index).unwrap();
        (result.1, result.2)
    }

    /// Returns the right-maximal run of the vector that starts with occurrence of item `value` of rank `rank`.
    ///
    /// The returned tuple is (starting index, run length).
    /// Returns [`None`] if there is no such occurrence.
    /// See also [`VectorIndex::select`].
    pub fn select_run(&self, rank: usize, value: <Self as Vector>::Item) -> Option<(usize, usize)> {
        if !self.contains(value) {
            return None;
        }
        self.data.map_up_with_run(self.start(value) + rank, value)
    }
}

impl<'a> Serialize for RLWM<'a> {
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
        Ok(RLWM { len, data, first, })
    }

    fn size_in_elements(&self) -> usize {
        let mut result = self.len.size_in_elements();
        result += self.data.size_in_elements();
        result += self.first.size_in_elements();
        result
    }
}

//-----------------------------------------------------------------------------

/// A read-only iterator over the items in [`RLWM`].
///
/// This is a more efficient override of the default iterator provided by [`Access`].
/// It accesses the vector one run at a time using [`RLWM::get_run`] instead of one item at a time using [`Access::get`].
///
/// Backward iteration with [`DoubleEndedIterator::next_back`] is supported but less efficient.
/// Method [`AccessIter::next_run`] can be used for iterating over runs of values.
///
/// See [`RLWM`] for an example.
#[derive(Clone, Debug)]
pub struct AccessIter<'a> {
    parent: &'a RLWM<'a>,
    // The first index we have not seen.
    index: usize,
    // The first index we should not visit.
    limit: usize,
    // Item value for the current run.
    value: <RLWM<'a> as Vector>::Item,
    // Remaining length of the current run.
    run_len: usize,
}

impl<'a> AccessIter<'a> {
    // Ensures that the next run is cached.
    fn ensure_next_run(&mut self) -> Option<()> {
        if self.index >= self.limit {
            return None;
        }
        if self.run_len == 0 {
            (self.value, self.run_len) = self.parent.get_run(self.index);
        }
        Some(())
    }

    // TODO: separate RunIter?
    /// Returns the run starting at the position [`Self::next`] would return next.
    ///
    /// The returned tuple is (starting index, value, run length).
    /// Returns [`None`] if the iterator is exhausted.
    ///
    /// Advances the iterator to the end of the run.
    /// By default, the returned runs are maximal.
    /// Using [`Iterator::next`] and [`DoubleEndedIterator::next_back`] may break maximality by consuming individual items.
    pub fn next_run(&mut self) -> Option<(usize, <RLWM<'a> as Vector>::Item, usize)> {
        self.ensure_next_run()?;
        let run_len = cmp::min(self.run_len, self.limit - self.index);
        let result = Some((self.index, self.value, run_len));
        self.index += run_len;
        self.run_len -= run_len;
        result
    }
}

impl<'a> Iterator for AccessIter<'a> {
    type Item = <RLWM<'a> as Vector>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.ensure_next_run()?;
        self.index += 1;
        self.run_len -= 1;
        Some(self.value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.limit - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> DoubleEndedIterator for AccessIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.limit {
            return None;
        }
        self.limit -= 1;
        Some(self.parent.get(self.limit))
    }
}

impl<'a> ExactSizeIterator for AccessIter<'a> {}

impl<'a> FusedIterator for AccessIter<'a> {}

//-----------------------------------------------------------------------------

/// A read-only iterator over the occurrences of a specific value in [`RLWM`].
///
/// The type of `Item` is [`(usize, usize)`] representing a pair (rank, index).
/// The item at position `index` has the given value, and the rank of that value at that position is `rank`.
///
/// Method [`ValueIter::next_run`] can be used for iterating over runs of the given value.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::VectorIndex;
/// use simple_sds::rlwm::RLWM;
///
/// // Construction
/// let runs: Vec<(u64, usize)> = vec![
///     (1, 3), (2, 2), (3, 4), (1, 2), (0, 1), (2, 3)
/// ];
/// let wm = RLWM::from(runs.clone());
///
/// // Iteration over values
/// let mut iter = wm.value_iter(2);
/// assert_eq!(wm.value_of(&iter), 2);
/// assert_eq!(iter.next(), Some((0, 3)));
/// assert_eq!(iter.next(), Some((1, 4)));
/// assert_eq!(iter.next(), Some((2, 12)));
/// assert_eq!(iter.next_run(), Some((3, 13, 2)));
/// assert!(iter.next().is_none());
/// ```
#[derive(Clone, Debug)]
pub struct ValueIter<'a> {
    parent: &'a RLWM<'a>,
    // The first rank we have not seen.
    rank: usize,
    // Item value being iterated.
    value: <RLWM<'a> as Vector>::Item,
    // Starting index of the current run.
    index: usize,
    // Remaining length of the current run.
    run_len: usize,
}

impl<'a> ValueIter<'a> {
    // Ensures that the next run is cached.
    fn ensure_next_run(&mut self) -> Option<()> {
        if self.index >= self.parent.len() {
            return None;
        }
        if self.run_len == 0 {
            if let Some((index, run_len)) = self.parent.select_run(self.rank, self.value) {
                self.index = index;
                self.run_len = run_len;
            } else {
                self.index = self.parent.len();
                return None;
            }
        }
        Some(())
    }

    // TODO: separate ValueRunIter?
    /// Returns the run starting at the position [`Self::next`] would return next.
    ///
    /// The returned tuple is (rank, starting index, run length).
    /// Returns [`None`] if the iterator is exhausted.
    ///
    /// Advances the iterator to the end of the run.
    /// By default, the returned runs are maximal.
    /// Using [`Iterator::next`] may break left-maximality by consuming individual items.
    pub fn next_run(&mut self) -> Option<(usize, usize, usize)> {
        self.ensure_next_run()?;
        let result = Some((self.rank, self.index, self.run_len));
        self.rank += self.run_len;
        self.index += self.run_len;
        self.run_len = 0;
        result
    }
}

impl<'a> Iterator for ValueIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.ensure_next_run()?;
        let result = Some((self.rank, self.index));
        self.rank += 1;
        self.index += 1;
        self.run_len -= 1;
        result
    }
}

impl<'a> FusedIterator for ValueIter<'a> {}

//-----------------------------------------------------------------------------

/// [`RLWM`] transformed into an iterator over its items.
///
/// The type of `Item` is [`u64`].
///
/// # Examples
///
/// ```
/// use simple_sds::rlwm::RLWM;
///
/// // Construction
/// let runs: Vec<(u64, usize)> = vec![
///     (1, 3), (2, 2), (3, 4), (1, 2), (0, 1), (2, 3)
/// ];
/// let wm = RLWM::from(runs.clone());
/// let values: Vec<u64> = runs.iter().flat_map(|(value, length)|
///     std::iter::repeat(*value).take(*length)
/// ).collect();
///
/// // Into iterator
/// let v: Vec<u64> = wm.into_iter().collect();
/// assert_eq!(v, values);
/// ```
#[derive(Clone, Debug)]
pub struct IntoIter<'a> {
    parent: RLWM<'a>,
    index: usize,
}

impl<'a> Iterator for IntoIter<'a> {
    type Item = <RLWM<'a> as Vector>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.parent.len() {
            return None;
        }
        let value = self.parent.get(self.index);
        self.index += 1;
        Some(value)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.parent.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for IntoIter<'a> {}

impl<'a> FusedIterator for IntoIter<'a> {}

impl<'a> IntoIterator for RLWM<'a> {
    type Item = <Self as Vector>::Item;
    type IntoIter = IntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            parent: self,
            index: 0,
        }
    }
}
//-----------------------------------------------------------------------------