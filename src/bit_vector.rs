//! An immutable binary array supporting rank, select, and related queries.

use crate::bit_vector::rank_support::RankSupport;
use crate::bit_vector::select_support::SelectSupport;
///use crate::ops::{BitVec, Rank, Select, Complement};
use crate::ops::{BitVec, Rank, Select};
use crate::raw_vector::{RawVector, GetRaw, PushRaw};
use crate::serialize::Serialize;
use crate::bits;

use std::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, FromIterator};
use std::io;

pub mod rank_support;
pub mod select_support;

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

// TODO: Usage example from BitVec trait
/// An immutable binary array supporting, rank, select, and related queries.
///
/// This structure contains [`RawVector`], which is in turn contains [`Vec`].
/// Because most queries require separate support structures, the binary array itself is immutable.
/// Conversions between `BitVector` and [`RawVector`] are possible using the [`From`] trait.
/// The maximum length of the vector is `usize::MAX` bits.
///
/// `BitVector` implements the following `simple_sds` traits:
/// * Basic functionality: [`BitVec`]
/// * Queries and operations: [`Rank`], [`Select`], [`Complement`]
/// * Serialization: [`Serialize`]
///
/// See [`rank_support`] and [`select_support`] for algorithmic details on rank/select queries.
///
/// # Notes
///
/// * `BitVector` never panics from I/O errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitVector {
    ones: usize,
    data: RawVector,
    rank: Option<RankSupport>,
    select: Option<SelectSupport>,
}

//-----------------------------------------------------------------------------

/// A read-only iterator over [`BitVector`].
///
/// The type of `Item` is `bool`.
///
/// # Examples
///
/// ```
/// use simple_sds::bit_vector::BitVector;
/// use simple_sds::ops::BitVec;
///
/// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
/// let bv: BitVector = source.iter().cloned().collect();
/// assert_eq!(bv.iter().len(), source.len());
/// for (index, value) in bv.iter().enumerate() {
///     assert_eq!(source[index], value);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Iter<'a> {
    parent: &'a BitVector,
    // The first index we have not visited.
    next: usize,
    // The first index we should not visit.
    limit: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.limit {
            None
        } else {
            let result = Some(self.parent.get(self.next));
            self.next += 1;
            result
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.limit - self.next;
        (remaining, Some(remaining))
    }
}

impl<'a> DoubleEndedIterator for Iter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.next >= self.limit {
            None
        } else {
            self.limit -= 1;
            Some(self.parent.get(self.limit))
        }
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {}

impl<'a> FusedIterator for Iter<'a> {}

//-----------------------------------------------------------------------------

impl<'a> BitVec<'a> for BitVector {
    type Iter = Iter<'a>;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn count_ones(&self) -> usize {
        self.ones
    }

    fn get(&self, index: usize) -> bool {
        self.data.bit(index)
    }

    fn iter(&'a self) -> Self::Iter {
        Self::Iter {
            parent: self,
            next: 0,
            limit: self.len(),
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> Rank<'a> for BitVector {
    fn supports_rank(&self) -> bool {
        self.rank != None
    }

    fn enable_rank(&mut self) {
        if !self.supports_rank() {
            let rank_support = RankSupport::new(self);
            self.rank = Some(rank_support);
        }
    }

    fn rank(&self, index: usize) -> usize {
        if index >= self.len() {
            return self.count_ones();
        }
        let rank_support = self.rank.as_ref().unwrap();
        rank_support.rank(self, index)
    }
}

//-----------------------------------------------------------------------------

// TODO recommend SparseVector for sparse bitvectors
/// An iterator over the set bits in a [`BitVector`].
///
/// The type of `Item` is `(usize, usize)`.
/// This can be interpreted as:
///
/// * `(index, value)` or `(i, select(i))` in the integer array; or
/// * `(rank(j), j)` in the binary array with `j` such that `self.get(j) == true`.
///
/// Note that `index` is not always the index provided by [`Iterator::enumerate`].
/// Many [`Select`] queries create iterators in the middle of the bitvector.
///
/// The bitvector is assumed to be at least somewhat dense.
/// If the frequency of ones is o(1 / 64), iteration may be inefficient.
///
/// # Examples
///
/// ```
/// use simple_sds::bit_vector::BitVector;
/// use simple_sds::ops::{BitVec, Select};
///
/// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
/// let bv: BitVector = source.into_iter().collect();
/// let mut iter = bv.one_iter();
/// assert_eq!(iter.len(), bv.count_ones());
/// assert_eq!(iter.next(), Some((0, 0)));
/// assert_eq!(iter.next(), Some((1, 2)));
/// assert_eq!(iter.next(), Some((2, 3)));
/// assert_eq!(iter.next(), Some((3, 5)));
/// assert_eq!(iter.next(), Some((4, 6)));
/// assert_eq!(iter.next(), None);
/// ```
#[derive(Clone, Debug)]
pub struct OneIter<'a> {
    parent: &'a BitVector,
    // The first (i, candidate for select(i)) we have not visited.
    next: (usize, usize),
    // The first (i, candidate for select(i)) we should not visit.
    limit: (usize, usize),
}

impl<'a> Iterator for OneIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.0 >= self.limit.0 {
            None
        } else {
            let (mut index, mut offset) = bits::split_offset(self.next.1);
            loop {
                let word = self.parent.data.word(index) & !bits::low_set(offset);
                if word == 0 {
                    index += 1; offset = 0;
                } else {
                    offset = word.trailing_zeros() as usize;
                    let result = (self.next.0, bits::bit_offset(index, offset));
                    self.next = (result.0 + 1, result.1 + 1);
                    return Some(result);
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.limit.0 - self.next.0;
        (remaining, Some(remaining))
    }
}

impl<'a> DoubleEndedIterator for OneIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.next.0 >= self.limit.0 {
            None
        } else {
            self.limit = (self.limit.0 - 1, self.limit.1 - 1);
            let (mut index, mut offset) = bits::split_offset(self.limit.1);
            loop {
                let word = self.parent.data.word(index) & bits::low_set(offset + 1);
                if word == 0 {
                    index -= 1; offset = 63;
                } else {
                    offset = bits::WORD_BITS - 1 - (word.leading_zeros() as usize);
                    self.limit.1 = bits::bit_offset(index, offset);
                    return Some(self.limit);
                }
            }
        }
    }
}

impl<'a> ExactSizeIterator for OneIter<'a> {}

impl<'a> FusedIterator for OneIter<'a> {}

//-----------------------------------------------------------------------------

impl<'a> Select<'a> for BitVector {
    type OneIter = OneIter<'a>;

    fn supports_select(&self) -> bool {
        self.select != None
    }

    fn enable_select(&mut self) {
        if !self.supports_select() {
            let select_support = SelectSupport::new(self);
            self.select = Some(select_support);
        }
    }

    fn one_iter(&'a self) -> Self::OneIter {
        Self::OneIter {
            parent: self,
            next: (0, 0),
            limit: (self.count_ones(), self.len()),
        }
    }

    fn select(&'a self, _: usize) -> Self::OneIter {
        // FIXME
        self.one_iter()
    }

    fn predecessor(&'a self, _: usize) -> Self::OneIter {
        // FIXME
        self.one_iter()
    }

    fn successor(&'a self, _: usize) -> Self::OneIter {
        // FIXME
        self.one_iter()
    }
}

//-----------------------------------------------------------------------------

// FIXME Complement + ZeroIter

//-----------------------------------------------------------------------------

impl Serialize for BitVector {
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.ones.serialize(writer)?;
        Ok(())
    }

    fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.data.serialize(writer)?;
        self.rank.serialize(writer)?;
        self.select.serialize(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let ones = usize::load(reader)?;
        let data = RawVector::load(reader)?;
        let rank = Option::<RankSupport>::load(reader)?;
        let select = Option::<SelectSupport>::load(reader)?;
        let result = BitVector {
            ones: ones,
            data: data,
            rank: rank,
            select: select,
        };
        Ok(result)
    }

    fn size_in_bytes(&self) -> usize {
        self.ones.size_in_bytes() +
        self.data.size_in_bytes() +
        self.rank.size_in_bytes() +
        self.select.size_in_bytes()
    }
}

//-----------------------------------------------------------------------------

impl AsRef<RawVector> for BitVector {
    fn as_ref(&self) -> &RawVector {
        &(self.data)
    }
}

impl From<RawVector> for BitVector {
    fn from(data: RawVector) -> Self {
        let ones = data.count_ones();
        BitVector {
            ones: ones,
            data: data,
            rank: None,
            select: None,
        }
    }
}

impl From<BitVector> for RawVector {
    fn from(source: BitVector) -> Self {
        source.data
    }
}

impl FromIterator<bool> for BitVector {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let mut iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();
        let mut data = RawVector::with_capacity(lower_bound);
        while let Some(value) = iter.next() {
            data.push_bit(value);
        }
        let ones = data.count_ones();
        BitVector {
            ones: ones,
            data: data,
            rank: None,
            select: None,
        }
    }
}

//-----------------------------------------------------------------------------

/// [`BitVector`] transformed into an iterator.
///
/// The type of `Item` is `bool`.
///
/// # Examples
///
/// ```
/// use simple_sds::bit_vector::BitVector;
///
/// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
/// let bv: BitVector = source.iter().cloned().collect();
/// let target: Vec<bool> = bv.into_iter().collect();
/// assert_eq!(target, source);
/// ```
#[derive(Clone, Debug)]
pub struct IntoIter {
    parent: BitVector,
    index: usize,
}

impl Iterator for IntoIter {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.parent.len() {
            None
        } else {
            let result = Some(self.parent.get(self.index));
            self.index += 1;
            result
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.parent.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IntoIter {}

impl FusedIterator for IntoIter {}

impl<'a> IntoIterator for BitVector {
    type Item = bool;
    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            parent: self,
            index: 0,
        }
    }
}

//-----------------------------------------------------------------------------
