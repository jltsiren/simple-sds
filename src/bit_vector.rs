//! An immutable bit array supporting rank, select, and related queries.

use crate::bit_vector::rank_support::RankSupport;
use crate::bit_vector::select_support::SelectSupport;
use crate::ops::{BitVec, Rank, Select, PredSucc};
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

/// An immutable bit array supporting, rank, select, and related queries.
///
/// This structure contains [`RawVector`], which is in turn contains [`Vec`].
/// Because most queries require separate support structures, the bit array itself is immutable.
/// Conversions between `BitVector` and [`RawVector`] are possible using the [`From`] trait.
/// The maximum length of the vector is `usize::MAX` bits.
///
/// `BitVector` implements the following `simple_sds` traits:
/// * Basic functionality: [`BitVec`]
/// * Queries and operations: [`Rank`], [`Select`], [`PredSucc`], [`Complement`]
/// * Serialization: [`Serialize`]
///
/// See [`rank_support`] and [`select_support`] for algorithmic details on rank/select queries.
/// Predecessor and successor queries depend on both support structures.
///
/// # Examples
///
/// ```
/// use simple_sds::bit_vector::BitVector;
/// use simple_sds::ops::{BitVec, Rank, Select, PredSucc};
/// use simple_sds::raw_vector::{RawVector, SetRaw};
///
/// let mut data = RawVector::with_len(137, false);
/// data.set_bit(1, true); data.set_bit(33, true); data.set_bit(95, true); data.set_bit(123, true);
/// let mut bv = BitVector::from(data);
///
/// // BitVec
/// assert_eq!(bv.len(), 137);
/// assert!(!bv.is_empty());
/// assert_eq!(bv.count_ones(), 4);
/// assert!(bv.get(33));
/// assert!(!bv.get(34));
/// for (index, value) in bv.iter().enumerate() {
///     assert_eq!(value, bv.get(index));
/// }
///
/// // Rank
/// bv.enable_rank();
/// assert!(bv.supports_rank());
/// assert_eq!(bv.rank(33), 1);
/// assert_eq!(bv.rank(34), 2);
/// assert_eq!(bv.complement_rank(65), 63);
///
/// // Select
/// bv.enable_select();
/// assert!(bv.supports_select());
/// assert_eq!(bv.select(2).next(), Some((2, 95)));
/// let v: Vec<(usize, usize)> = bv.one_iter().collect();
/// assert_eq!(v, vec![(0, 1), (1, 33), (2, 95), (3, 123)]);
///
/// // PredSucc
/// bv.enable_pred_succ();
/// assert!(bv.supports_pred_succ());
/// assert_eq!(bv.predecessor(0).next(), None);
/// assert_eq!(bv.predecessor(1).next(), Some((0, 1)));
/// assert_eq!(bv.predecessor(2).next(), Some((0, 1)));
/// assert_eq!(bv.successor(122).next(), Some((3, 123)));
/// assert_eq!(bv.successor(123).next(), Some((3, 123)));
/// assert_eq!(bv.successor(124).next(), None);
/// ```
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
/// * `(rank(j), j)` in the bit array with `j` such that `self.get(j) == true`.
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

impl<'a> OneIter<'a> {
    // Build an empty iterator for the parent bitvector.
    fn empty_iter(parent: &'a BitVector) -> OneIter<'a> {
        OneIter {
            parent: parent,
            next: (parent.count_ones(), parent.len()),
            limit: (parent.count_ones(), parent.len()),
        }
    }
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

    fn select(&'a self, index: usize) -> Self::OneIter {
         if index >= self.count_ones() {
             Self::OneIter::empty_iter(self)
        } else {
            let select_support = self.select.as_ref().unwrap();
            let value = select_support.select(self, index);
            Self::OneIter {
                parent: self,
                next: (index, value),
                limit: (self.count_ones(), self.len()),
            }
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> PredSucc<'a> for BitVector {
    type OneIter = OneIter<'a>;

    fn supports_pred_succ(&self) -> bool {
        self.rank != None && self.select != None
    }

    fn enable_pred_succ(&mut self) {
        self.enable_rank();
        self.enable_select();
    }

    fn predecessor(&'a self, index: usize) -> Self::OneIter {
        let rank = self.rank(index + 1);
        if rank == 0 {
            Self::OneIter::empty_iter(self)
        } else {
            self.select(rank - 1)
        }
    }

    fn successor(&'a self, index: usize) -> Self::OneIter {
        let rank = self.rank(index);
        if rank >= self.count_ones() {
            Self::OneIter::empty_iter(self)
        } else {
            self.select(rank)
        }
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
