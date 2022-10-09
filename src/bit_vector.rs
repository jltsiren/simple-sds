//! An immutable bit array supporting rank, select, and related queries.

use crate::bit_vector::rank_support::RankSupport;
use crate::bit_vector::select_support::SelectSupport;
use crate::ops::{BitVec, Rank, Select, SelectZero, PredSucc};
use crate::raw_vector::{RawVector, AccessRaw, PushRaw};
use crate::serialize::Serialize;
use crate::bits;

use std::io::{Error, ErrorKind};
use std::iter::{FusedIterator, FromIterator};
use std::{cmp, io, marker};

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
/// The maximum length of the vector is approximately [`usize::MAX`] bits.
///
/// Conversions between various [`BitVec`] types are possible using the [`From`] trait.
///
/// `BitVector` implements the following `simple_sds` traits:
/// * Basic functionality: [`BitVec`]
/// * Queries and operations: [`Rank`], [`Select`], [`PredSucc`], [`SelectZero`]
/// * Serialization: [`Serialize`]
///
/// See [`rank_support`] and [`select_support`] for algorithmic details on rank/select queries.
/// Predecessor and successor queries depend on both support structures.
///
/// The support structures are serialized as [`Option`] and hence may be missing.
/// When `BitVector` is used as a part of another structure, the user should enable the required support structures after loading.
/// This makes interoperation with other libraries easier, as the other library only has to serialize the bitvector itself.
/// Enabling support structures is fast if they were present in the serialized data.
///
/// # Examples
///
/// ```
/// use simple_sds::bit_vector::BitVector;
/// use simple_sds::ops::{BitVec, Rank, Select, SelectZero, PredSucc};
/// use simple_sds::raw_vector::{RawVector, AccessRaw};
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
/// assert_eq!(bv.rank_zero(65), 63);
///
/// // Select
/// bv.enable_select();
/// assert!(bv.supports_select());
/// assert_eq!(bv.select(1), Some(33));
/// let mut iter = bv.select_iter(2);
/// assert_eq!(iter.next(), Some((2, 95)));
/// assert_eq!(iter.next(), Some((3, 123)));
/// assert!(iter.next().is_none());
/// let v: Vec<(usize, usize)> = bv.one_iter().collect();
/// assert_eq!(v, vec![(0, 1), (1, 33), (2, 95), (3, 123)]);
///
/// // SelectZero
/// bv.enable_select_zero();
/// assert!(bv.supports_select_zero());
/// assert_eq!(bv.select_zero(2), Some(3));
/// let v: Vec<(usize, usize)> = bv.zero_iter().take(4).collect();
/// assert_eq!(v, vec![(0, 0), (1, 2), (2, 3), (3, 4)]);
///
/// // PredSucc
/// bv.enable_pred_succ();
/// assert!(bv.supports_pred_succ());
/// assert!(bv.predecessor(0).next().is_none());
/// assert_eq!(bv.predecessor(1).next(), Some((0, 1)));
/// assert_eq!(bv.predecessor(2).next(), Some((0, 1)));
/// assert_eq!(bv.successor(122).next(), Some((3, 123)));
/// assert_eq!(bv.successor(123).next(), Some((3, 123)));
/// assert!(bv.successor(124).next().is_none());
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
    select: Option<SelectSupport<Identity>>,
    select_zero: Option<SelectSupport<Complement>>,
}

impl BitVector {
    /// Returns a copy of the source bitvector as `BitVector`.
    ///
    /// The copy is created by iterating over the set bits using [`Select::one_iter`].
    /// [`From`] implementations from other bitvector types should generally use this function.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::bit_vector::BitVector;
    /// use simple_sds::ops::BitVec;
    /// use std::iter::FromIterator;
    ///
    /// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
    /// let bv = BitVector::from_iter(source);
    /// let copy = BitVector::copy_bit_vec(&bv);
    /// assert_eq!(copy.len(), bv.len());
    /// assert_eq!(copy.count_ones(), bv.count_ones());
    /// ```
    pub fn copy_bit_vec<'a, T: BitVec<'a> + Select<'a>>(source: &'a T) -> Self {
        let mut data = RawVector::with_len(source.len(), false);
        for (_, index) in source.one_iter() {
            data.set_bit(index, true);
        }
        BitVector::from(data)
    }
}

//-----------------------------------------------------------------------------

/// A read-only iterator over [`BitVector`].
///
/// The type of `Item` is [`bool`].
/// There are efficient implementations of [`Iterator::nth`] and [`DoubleEndedIterator::nth_back`].
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

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.next += cmp::min(n, self.limit - self.next);
        self.next()
    }

    #[inline]
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

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.limit -= cmp::min(n, self.limit - self.next);
        self.next_back()
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {}

impl<'a> FusedIterator for Iter<'a> {}

//-----------------------------------------------------------------------------

impl<'a> BitVec<'a> for BitVector {
    type Iter = Iter<'a>;

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn count_ones(&self) -> usize {
        self.ones
    }

    #[inline]
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
        unsafe { rank_support.rank_unchecked(self, index) }
    }
}

//-----------------------------------------------------------------------------

/// An implicit transformation of [`BitVector`] into another vector of the same length.
///
/// Types that implement this trait can be used as parameters for [`SelectSupport`] and [`OneIter`].
pub trait Transformation {
    /// Reads a bit from the transformed bitvector.
    ///
    /// # Arguments
    ///
    /// * `parent`: The parent bitvector.
    /// * `index`: Index in the bit array.
    ///
    /// # Panics
    ///
    /// May panic if `index` is not a valid offset in the bit array.
    fn bit(parent: &BitVector, index: usize) -> bool;

    /// Reads a 64-bit word from the transformed bitvector.
    ///
    /// # Arguments
    ///
    /// * `parent`: The parent bitvector.
    /// * `index`: Read the word starting at offset `index * 64` of the bit array.
    ///
    /// # Panics
    ///
    /// May panic if `index * 64` is not a valid offset in the bit array.
    fn word(parent: &BitVector, index: usize) -> u64;

    /// Unsafe version of [`Transformation::word`] without bounds checks.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if `index * 64` is not a valid offset in the bit array.
    unsafe fn word_unchecked(parent: &BitVector, index: usize) -> u64;

    /// Returns the length of the integer array or the number of ones in the bit array of the transformed bitvector.
    fn count_ones(parent: &BitVector) -> usize;

    /// Returns an iterator over the set bits in the transformed bitvector.
    fn one_iter(parent: &BitVector) -> OneIter<'_, Self>;
}

/// The bitvector as it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {}

impl Transformation for Identity {
    #[inline]
    fn bit(parent: &BitVector, index: usize) -> bool {
        parent.get(index)
    }

    #[inline]
    fn word(parent: &BitVector, index: usize) -> u64 {
        parent.data.word(index)
    }

    #[inline]
    unsafe fn word_unchecked(parent: &BitVector, index: usize) -> u64 {
        parent.data.word_unchecked(index)
    }

    #[inline]
    fn count_ones(parent: &BitVector) -> usize {
        parent.count_ones()
    }

    fn one_iter(parent: &BitVector) -> OneIter<'_, Self> {
        parent.one_iter()
    }
}

/// The bitvector implicitly transformed into its complement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Complement {}

impl Transformation for Complement {
    #[inline]
    fn bit(parent: &BitVector, index: usize) -> bool {
        !parent.get(index)
    }

    fn word(parent: &BitVector, index: usize) -> u64 {
        let (last_index, offset) = bits::split_offset(parent.len());
        if index >= last_index {
            (!parent.data.word(index)) & unsafe { bits::low_set_unchecked(offset) }
        } else {
            unsafe { !parent.data.word_unchecked(index) }
        }
    }

    unsafe fn word_unchecked(parent: &BitVector, index: usize) -> u64 {
        let (last_index, offset) = bits::split_offset(parent.len());
        if index >= last_index {
            (!parent.data.word_unchecked(index)) & bits::low_set_unchecked(offset)
        } else {
            !parent.data.word_unchecked(index)
        }
    }

    #[inline]
    fn count_ones(parent: &BitVector) -> usize {
        parent.count_zeros()
    }

    fn one_iter(parent: &BitVector) -> OneIter<'_, Self> {
        parent.zero_iter()
    }
}

//-----------------------------------------------------------------------------

/// An iterator over the set bits in an implicitly transformed [`BitVector`].
///
/// The type of `Item` is `(`[`usize`]`, `[`usize`]`)`.
/// This can be interpreted as:
///
/// * `(index, value)` or `(i, select(i))` in the integer array; or
/// * `(rank(j), j)` in the bit array with `j` such that `self.get(j) == true`.
///
/// Note that `index` is not always the index provided by [`Iterator::enumerate`].
/// Queries may create iterators in the middle of the bitvector.
///
/// The bitvector is assumed to be at least somewhat dense.
/// If the frequency of ones is less than 1/64, iteration may be inefficient.
/// The implementation of [`Iterator::nth`] is optimized to skip set bits without determining their positions.
///
/// This type must be parameterized with a [`Transformation`].
///
/// # Examples
///
/// ```
/// use simple_sds::bit_vector::BitVector;
/// use simple_sds::ops::{BitVec, Select, SelectZero};
///
/// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false, true];
/// let bv: BitVector = source.into_iter().collect();
///
/// let mut iter = bv.one_iter();
/// assert_eq!(iter.len(), bv.count_ones());
/// assert_eq!(iter.next(), Some((0, 0)));
/// assert_eq!(iter.nth(0), Some((1, 2)));
/// assert_eq!(iter.nth(1), Some((3, 5)));
/// assert_eq!(iter.next(), Some((4, 6)));
/// assert!(iter.nth(1).is_none());
///
/// let mut iter = bv.zero_iter();
/// assert_eq!(iter.nth(1), Some((1, 4)));
/// assert_eq!(iter.next(), Some((2, 7)));
/// assert!(iter.next().is_none());
/// ```
#[derive(Clone, Debug)]
pub struct OneIter<'a, T: Transformation + ?Sized> {
    parent: &'a BitVector,
    // The first (i, candidate for select(i)) we have not visited.
    next: (usize, usize),
    // The first (i, candidate for select(i)) we should not visit.
    limit: (usize, usize),
    // We use `T` only for accessing static methods.
    _marker: marker::PhantomData<T>,
}

impl<'a, T: Transformation + ?Sized> OneIter<'a, T> {
    // Build an empty iterator for the parent bitvector.
    fn empty_iter(parent: &'a BitVector) -> OneIter<'a, T> {
        OneIter {
            parent,
            next: (T::count_ones(parent), parent.len()),
            limit: (T::count_ones(parent), parent.len()),
            _marker: marker::PhantomData,
        }
    }
}

impl<'a, T: Transformation + ?Sized> Iterator for OneIter<'a, T> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.0 >= self.limit.0 {
            None
        } else {
            let (mut index, offset) = bits::split_offset(self.next.1);
            // We trust that the iterator has been initialized properly and the above check
            // guarantees that `self.next.1 < self.limit.1` and `self.limit.1 <= self.parent.len()`.
            let mut word = unsafe { T::word_unchecked(self.parent, index) & !bits::low_set_unchecked(offset) };
            while word == 0 {
                index += 1;
                word = unsafe { T::word_unchecked(self.parent, index) };
            }
            let offset = word.trailing_zeros() as usize;
            let result = (self.next.0, bits::bit_offset(index, offset));
            self.next = (result.0 + 1, result.1 + 1);
            Some(result)
        }
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if self.next.0 + n >= self.limit.0 {
            self.next = self.limit;
            return None;
        }
        let (mut index, offset) = bits::split_offset(self.next.1);
        let mut word = unsafe { T::word_unchecked(self.parent, index) & !bits::low_set_unchecked(offset) };
        let mut relative_rank = n;
        let mut ones = word.count_ones() as usize;
        while ones <= relative_rank {
            index += 1;
            word = unsafe { T::word_unchecked(self.parent, index) };
            relative_rank -= ones;
            ones = word.count_ones() as usize;
        }
        let offset = unsafe { bits::select(word, relative_rank) };
        let result = (self.next.0 + n, bits::bit_offset(index, offset));
        self.next = (result.0 + 1, result.1 + 1);
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.limit.0 - self.next.0;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Transformation + ?Sized> DoubleEndedIterator for OneIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.next.0 >= self.limit.0 {
            None
        } else {
            self.limit = (self.limit.0 - 1, self.limit.1 - 1);
            let (mut index, offset) = bits::split_offset(self.limit.1);
            // We trust that the iterator has been initialized properly and the above check
            // guarantees that `self.next.1 <= self.limit.1` and `self.limit.1 < self.parent.len()`.
            let mut word = unsafe { T::word_unchecked(self.parent, index) & bits::low_set_unchecked(offset + 1) };
            while word == 0 {
                index -= 1;
                word = unsafe { T::word_unchecked(self.parent, index) };
            }
            let offset = bits::WORD_BITS - 1 - (word.leading_zeros() as usize);
            self.limit.1 = bits::bit_offset(index, offset);
            Some(self.limit)
        }
    }
}

impl<'a, T: Transformation + ?Sized> ExactSizeIterator for OneIter<'a, T> {}

impl<'a, T: Transformation + ?Sized> FusedIterator for OneIter<'a, T> {}

//-----------------------------------------------------------------------------

impl<'a> Select<'a> for BitVector {
    type OneIter = OneIter<'a, Identity>;

    fn supports_select(&self) -> bool {
        self.select != None
    }

    fn enable_select(&mut self) {
        if !self.supports_select() {
            let select_support = SelectSupport::<Identity>::new(self);
            self.select = Some(select_support);
        }
    }

    fn one_iter(&'a self) -> Self::OneIter {
        Self::OneIter {
            parent: self,
            next: (0, 0),
            limit: (Identity::count_ones(self), self.len()),
            _marker: marker::PhantomData,
        }
    }

    fn select(&'a self, rank: usize) -> Option<usize> {
         if rank >= Identity::count_ones(self) {
             None
        } else {
            let select_support = self.select.as_ref().unwrap();
            let value = unsafe { select_support.select_unchecked(self, rank) };
            Some(value)
        }
    }

    fn select_iter(&'a self, rank: usize) -> Self::OneIter {
         if rank >= Identity::count_ones(self) {
             Self::OneIter::empty_iter(self)
        } else {
            let select_support = self.select.as_ref().unwrap();
            let value = unsafe { select_support.select_unchecked(self, rank) };
            Self::OneIter {
                parent: self,
                next: (rank, value),
                limit: (Identity::count_ones(self), self.len()),
                _marker: marker::PhantomData,
            }
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> SelectZero<'a> for BitVector {
    type ZeroIter = OneIter<'a, Complement>;

    fn supports_select_zero(&self) -> bool {
        self.select_zero != None
    }

    fn enable_select_zero(&mut self) {
        if !self.supports_select_zero() {
            let select_support = SelectSupport::<Complement>::new(self);
            self.select_zero = Some(select_support);
        }
    }

    fn zero_iter(&'a self) -> Self::ZeroIter {
        Self::ZeroIter {
            parent: self,
            next: (0, 0),
            limit: (Complement::count_ones(self), self.len()),
            _marker: marker::PhantomData,
        }
    }

    fn select_zero(&'a self, rank: usize) -> Option<usize> {
         if rank >= Complement::count_ones(self) {
             None
        } else {
            let select_support = self.select_zero.as_ref().unwrap();
            let value = unsafe { select_support.select_unchecked(self, rank) };
            Some(value)
        }
    }

    fn select_zero_iter(&'a self, rank: usize) -> Self::ZeroIter {
         if rank >= Complement::count_ones(self) {
             Self::ZeroIter::empty_iter(self)
        } else {
            let select_support = self.select_zero.as_ref().unwrap();
            let value = unsafe { select_support.select_unchecked(self, rank) };
            Self::ZeroIter {
                parent: self,
                next: (rank, value),
                limit: (Complement::count_ones(self), self.len()),
                _marker: marker::PhantomData,
            }
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> PredSucc<'a> for BitVector {
    type OneIter = OneIter<'a, Identity>;

    fn supports_pred_succ(&self) -> bool {
        self.rank != None && self.select != None
    }

    fn enable_pred_succ(&mut self) {
        self.enable_rank();
        self.enable_select();
    }

    fn predecessor(&'a self, value: usize) -> Self::OneIter {
        let rank = self.rank(value + 1);
        if rank == 0 {
            Self::OneIter::empty_iter(self)
        } else {
            self.select_iter(rank - 1)
        }
    }

    fn successor(&'a self, value: usize) -> Self::OneIter {
        let rank = self.rank(value);
        if rank >= self.count_ones() {
            Self::OneIter::empty_iter(self)
        } else {
            self.select_iter(rank)
        }
    }
}

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
        self.select_zero.serialize(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let ones = usize::load(reader)?;
        let data = RawVector::load(reader)?;
        if ones > data.len() {
            return Err(Error::new(ErrorKind::InvalidData, "Too many set bits"));
        }

        let rank = Option::<RankSupport>::load(reader)?;
        if let Some(value) = rank.as_ref() {
            if value.blocks() != bits::div_round_up(data.len(), RankSupport::BLOCK_SIZE) {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid number of rank blocks"))
            }
        }

        let select = Option::<SelectSupport<Identity>>::load(reader)?;
        if let Some(value) = select.as_ref() {
            if value.superblocks() != bits::div_round_up(ones, SelectSupport::<Identity>::SUPERBLOCK_SIZE) {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid number of select superblocks"))
            }
        }

        let select_zero = Option::<SelectSupport<Complement>>::load(reader)?;
        if let Some(value) = select_zero.as_ref() {
            if value.superblocks() != bits::div_round_up(data.len() - ones, SelectSupport::<Complement>::SUPERBLOCK_SIZE) {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid number of select_zero superblocks"))
            }
        }

        Ok(BitVector {
            ones, data, rank, select, select_zero,
        })
    }

    fn size_in_elements(&self) -> usize {
        self.ones.size_in_elements() +
        self.data.size_in_elements() +
        self.rank.size_in_elements() +
        self.select.size_in_elements() +
        self.select_zero.size_in_elements()
    }
}

//-----------------------------------------------------------------------------

impl AsRef<RawVector> for BitVector {
    #[inline]
    fn as_ref(&self) -> &RawVector {
        &(self.data)
    }
}

impl From<RawVector> for BitVector {
    fn from(data: RawVector) -> Self {
        let ones = data.count_ones();
        BitVector {
            ones,
            data,
            rank: None,
            select: None,
            select_zero: None,
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
        let iter = iter.into_iter();
        let (lower_bound, _) = iter.size_hint();
        let mut data = RawVector::with_capacity(lower_bound);
        for value in iter {
            data.push_bit(value);
        }
        let ones = data.count_ones();
        BitVector {
            ones,
            data,
            rank: None,
            select: None,
            select_zero: None,
        }
    }
}

//-----------------------------------------------------------------------------
