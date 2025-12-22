//! An immutable run-length encoded integer vector supporting rank/select-type queries.

// FIXME document

use crate::int_vector::IntVector;
use crate::ops::{Vector, Access, VectorIndex};
use crate::rl_vector::RLVector;
use crate::serialize::Serialize;
use crate::wavelet_matrix::wm_core::WMCore;
use crate::wavelet_matrix::wm_core;

use std::io::{Error, ErrorKind};
use std::iter::FusedIterator;
use std::io;

// FIXME tests

//-----------------------------------------------------------------------------

// FIXME example
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
/// Overridden default implementations:
/// * [`VectorIndex::contains`] has a simple constant-time implementation.
/// * [`VectorIndex::inverse_select`] is effectively the same as [`Access::get`].
/// * [`Access::iter`] returns a more efficient iterator that accesses the vector one run at a time.
///
/// Both iterators ([`AccessIter`] and [`ValueIter`]) support iterating over the runs using `next_run`.
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
            rank,
            value,
            index: 0,
            run_len: 0,
        }
    }
}

impl <'a> RLWM<'a> {
    // FIXME: test, examples
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

    // FIXME: test, examples
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

// FIXME: tests, examples
/// A read-only iterator over the items in [`RLWM`].
///
/// This is a more efficient override of the default iterator provided by [`Access`].
/// It accesses the vector one run at a time using [`RLWM::get_run`] instead of one item at a time using [`Access::get`].
///
/// Method [`AccessIter::next_run`] can be used for iterating over runs of values.
#[derive(Clone, Debug)]
pub struct AccessIter<'a> {
    parent: &'a RLWM<'a>,
    // The first index we have not seen.
    index: usize,
    // Item value for the current run.
    value: <RLWM<'a> as Vector>::Item,
    // Remaining length of the current run.
    run_len: usize,
}

impl<'a> AccessIter<'a> {
    // Ensures that the next run is cached.
    fn ensure_next_run(&mut self) -> Option<()> {
        if self.index >= self.parent.len() {
            return None;
        }
        if self.run_len == 0 {
            (self.value, self.run_len) = self.parent.get_run(self.index);
        }
        Some(())
    }

    // TODO: separate RunIter?
    /// Returns the right-maximal run starting at the position [`Self::next`] would return next.
    ///
    /// The returned tuple is (starting index, value, run length).
    /// Advances the iterator to the end of the run.
    /// Returns [`None`] if the iterator is exhausted.
    pub fn next_run(&mut self) -> Option<(usize, <RLWM<'a> as Vector>::Item, usize)> {
        self.ensure_next_run()?;
        let result = Some((self.index, self.value, self.run_len));
        self.index += self.run_len;
        self.run_len = 0;
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
}

impl<'a> ExactSizeIterator for AccessIter<'a> {
    fn len(&self) -> usize {
        self.parent.len() - self.index
    }
}

impl<'a> FusedIterator for AccessIter<'a> {}

//-----------------------------------------------------------------------------

// FIXME: example, tests
/// A read-only iterator over the occurrences of a specific value in [`RLWM`].
///
/// The type of `Item` is [`(usize, usize)`] representing a pair (rank, index).
/// The item at position `index` has the given value, and the rank of that value at that position is `rank`.
///
/// Method [`ValueIter::next_run`] can be used for iterating over runs of the given value.
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
    /// Returns the right-maximal run starting at the position [`Self::next`] would return next.
    ///
    /// The returned tuple is (rank, starting index, run length).
    /// Advances the iterator to the end of the run.
    /// Returns [`None`] if the iterator is exhausted.
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

// FIXME IntoIter

//-----------------------------------------------------------------------------