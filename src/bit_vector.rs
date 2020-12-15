//! An immutable binary array supporting rank, select, and related queries.

///use crate::ops::{BitVec, Rank, Select, Complement};
//use crate::bit_vector::rank_support::RankSupport;
use crate::ops::{BitVec};
use crate::raw_vector::{RawVector, GetRaw, PushRaw};
use crate::serialize::Serialize;

use std::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, FromIterator};
use std::io;

pub mod rank_support;

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

// TODO: algorithms section: rank implementation, select implementation
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
/// # Notes
///
/// * `BitVector` never panics from I/O errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitVector {
    ones: usize,
    data: RawVector,
//    rank: Option<RankSupport>,
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
    // The first index we have not used.
    next: usize,
    // The first index we should not use.
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

// FIXME Rank

//-----------------------------------------------------------------------------

// FIXME Select + OneIter

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
//        self.rank.serialize(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let ones = usize::load(reader)?;
        let data = RawVector::load(reader)?;
//        let rank = Option<RankSupport>::load(reader)?;
        Ok(BitVector {
            ones: ones,
            data: data,
//            rank: rank,
        })
    }

    fn size_in_bytes(&self) -> usize {
//        self.ones.size_in_bytes() + self.data.size_in_bytes() + self.rank.size_in_bytes()
        self.ones.size_in_bytes() + self.data.size_in_bytes()
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
//            rank: None,
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
//            rank: None,
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

impl IntoIterator for BitVector {
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
