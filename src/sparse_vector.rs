//! An Elias-Fano encoded array supporting rank, select, and related queries.
//!
//! This structure is equivalent to `sd_vector` in [SDSL](https://github.com/simongog/sdsl-lite).
//! It is also known in the literature as sdarray:
//!
//! > Okanohara, Sadakane: Practical Entropy-Compressed Rank/Select Dictionary.  
//! > Proc. ALENEX 2007.  
//! > DOI: [10.1137/1.9781611972870.6](https://doi.org/10.1137/1.9781611972870.6)
//!
//! The rule for selecting the number of low bits and buckets is from:
//!
//! > Ma, Puglisi, Raman, Zhukova: On Elias-Fano for Rank Queries in FM-Indexes.  
//! > Proc. DCC 2021.  
//! > DOI: [10.1109/DCC50243.2021.00030](https://doi.org/10.1109/DCC50243.2021.00030)
//!
//! Assume that we have a bitvector of length `n` with `m` set bits, with `m` much smaller than `n`.
//! Let `w ≈ log(n) - log(m)`.
//! In the integer array interpretation (see [`BitVec`]), we take the low `w` bits from each value and store them in an [`IntVector`].
//! We place the values in buckets by the remaining high bits.
//! A [`BitVector`] encodes the number of values in each bucket in unary.
//! If there are `k >= 0` values in a bucket, the bitvector will contain `k` set bits followed by an unset bit.
//! Then
//!
//! > `select(i) = low[i] + ((high.select(i) - i) << w)`.
//!
//! Rank, predecessor, and successor queries use `select_zero` on `high` followed by a linear scan.
//!
//! The `select_zero` implementation is based on finding the right run of unset bits using binary search with `select`.
//! It is not particularly efficient.
//!
//! We can also support multisets that contain duplicate values (in the integer array interpretation).
//! Rank/select queries for unset bits do not work correctly with multisets.

use crate::bit_vector::BitVector;
use crate::int_vector::IntVector;
use crate::ops::{Vector, Access, BitVec, Rank, Select, PredSucc, SelectZero};
use crate::raw_vector::{RawVector, AccessRaw};
use crate::serialize::Serialize;
use crate::bits;

use std::convert::TryFrom;
use std::io::{Error, ErrorKind};
use std::iter::FusedIterator;
use std::{cmp, io};

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

/// An immutable Elias-Fano encoded bitvector supporting, rank, select, and related queries.
///
/// This structure should be used for sparse bitvectors, where frequency of set bits is low.
/// For dense bitvectors or when [`SelectZero`] is needed, [`BitVector`] is usually a better choice.
/// Because most queries require support structures for one of the components, the bitvector itself is immutable.
/// The maximum length of the vector is approximately [`usize::MAX`] bits.
///
/// Conversions between various [`BitVec`] types are possible using the [`From`] trait.
///
/// `SparseVector` supports partial multiset semantics.
/// A multiset bitvector is one that contains duplicate values in the integer array interpretation.
/// Queries that operate on present values work correctly with a multiset, while [`Rank::rank_zero`] and [`SelectZero`] do not.
/// Multiset vectors can be built with [`SparseBuilder::multiset`] and [`SparseVector::try_from_iter`].
///
/// `SparseVector` implements the following `simple_sds` traits:
/// * Basic functionality: [`BitVec`]
/// * Queries and operations: [`Rank`], [`Select`], [`SelectZero`], [`PredSucc`]
/// * Serialization: [`Serialize`]
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{BitVec, Rank, Select, SelectZero, PredSucc};
/// use simple_sds::sparse_vector::{SparseVector, SparseBuilder};
/// use std::convert::TryFrom;
///
/// let mut builder = SparseBuilder::new(137, 4).unwrap();
/// builder.set(1); builder.set(33); builder.set(95); builder.set(123);
/// let sv = SparseVector::try_from(builder).unwrap();
///
/// // BitVec
/// assert_eq!(sv.len(), 137);
/// assert!(!sv.is_empty());
/// assert_eq!(sv.count_ones(), 4);
/// assert_eq!(sv.count_zeros(), 133);
/// assert!(sv.get(33));
/// assert!(!sv.get(34));
/// for (index, value) in sv.iter().enumerate() {
///     assert_eq!(value, sv.get(index));
/// }
///
/// // Rank
/// assert!(sv.supports_rank());
/// assert_eq!(sv.rank(33), 1);
/// assert_eq!(sv.rank(34), 2);
/// assert_eq!(sv.rank_zero(65), 63);
///
/// // Select
/// assert!(sv.supports_select());
/// assert_eq!(sv.select(1), Some(33));
/// let mut iter = sv.select_iter(2);
/// assert_eq!(iter.next(), Some((2, 95)));
/// assert_eq!(iter.next(), Some((3, 123)));
/// assert!(iter.next().is_none());
/// let v: Vec<(usize, usize)> = sv.one_iter().collect();
/// assert_eq!(v, vec![(0, 1), (1, 33), (2, 95), (3, 123)]);
///
/// // SelectZero
/// assert!(sv.supports_select_zero());
/// assert_eq!(sv.select_zero(35), Some(37));
/// let mut iter = sv.select_zero_iter(92);
/// assert_eq!(iter.next(), Some((92, 94)));
/// assert_eq!(iter.next(), Some((93, 96)));
///
/// // PredSucc
/// assert!(sv.supports_pred_succ());
/// assert!(sv.predecessor(0).next().is_none());
/// assert_eq!(sv.predecessor(1).next(), Some((0, 1)));
/// assert_eq!(sv.predecessor(2).next(), Some((0, 1)));
/// assert_eq!(sv.successor(122).next(), Some((3, 123)));
/// assert_eq!(sv.successor(123).next(), Some((3, 123)));
/// assert!(sv.successor(124).next().is_none());
/// ```
///
/// # Notes
///
/// * `SparseVector` never panics from I/O errors.
/// * All `SparseVector` queries are always enabled without additional support structures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseVector {
    len: usize,
    high: BitVector,
    low: IntVector,
}

// Bitvector index encoded as offsets in `high` and `low`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pos {
    high: usize,
    low: usize,
}

// Bitvector index encoded as high and low parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Parts {
    high: usize,
    low: usize,
}

impl SparseVector {
    // Stop binary search in `select_zero` when there are at most this many runs left.
    const BINARY_SEARCH_THRESHOLD: usize = 16;

    /// Returns a copy of the source bitvector as `SparseVector`.
    ///
    /// The copy is created by iterating over the set bits using [`Select::one_iter`].
    /// [`From`] implementations from other bitvector types should generally use this function.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::bit_vector::BitVector;
    /// use simple_sds::ops::BitVec;
    /// use simple_sds::sparse_vector::SparseVector;
    /// use std::iter::FromIterator;
    ///
    /// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
    /// let bv = BitVector::from_iter(source);
    /// let sv = SparseVector::copy_bit_vec(&bv);
    /// assert_eq!(sv.len(), bv.len());
    /// assert_eq!(sv.count_ones(), bv.count_ones());
    /// assert!(!sv.is_multiset());
    /// ```
    pub fn copy_bit_vec<'a, T: BitVec<'a> + Select<'a>>(source: &'a T) -> Self {
        let mut builder = SparseBuilder::new(source.len(), source.count_ones()).unwrap();
        for (_, index) in source.one_iter() {
            unsafe { builder.set_unchecked(index); }
        }
        SparseVector::try_from(builder).unwrap()
    }

    /// Builds a vector from the values in the iterator using multiset semantics.
    ///
    /// Returns an error message if the values are not sorted.
    /// Universe size is set to be barely large enough for the values.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::sparse_vector::SparseVector;
    /// use simple_sds::ops::{BitVec, Select};
    ///
    /// let source: Vec<usize> = vec![3, 4, 4, 7, 11, 19];
    /// let sv = SparseVector::try_from_iter(source.iter().cloned()).unwrap();
    /// assert_eq!(sv.len(), 20);
    /// assert_eq!(sv.count_ones(), source.len());
    /// assert!(sv.is_multiset());
    ///
    /// for (index, value) in sv.one_iter() {
    ///     assert_eq!(value, source[index]);
    /// }
    /// ```
    pub fn try_from_iter<T: Iterator<Item = usize> + DoubleEndedIterator + ExactSizeIterator>(iter: T) -> Result<SparseVector, &'static str> {
        let mut iter = iter;
        let (ones, _) = iter.size_hint();
        let universe = if let Some(pos) = iter.next_back() { pos + 1 } else { 0 };
        let mut builder = SparseBuilder::multiset(universe, ones);
        for pos in iter {
            builder.try_set(pos)?;
        }
        if universe > 0 {
            builder.try_set(universe - 1)?;
        }
        SparseVector::try_from(builder)
    }

    /// Returns `true` if the vector is a multiset (contains duplicate values).
    ///
    /// This method is somewhat expensive, as it iterates over the vector.
    pub fn is_multiset(&self) -> bool {
        let mut prev = self.len();
        for (_, value) in self.one_iter() {
            if value == prev {
                return true;
            }
            prev = value;
        }
        false
    }

    // Split a bitvector index into high and low parts.
    fn split(&self, index: usize) -> Parts {
        Parts {
            high: index >> self.low.width(),
            low: index & unsafe { bits::low_set_unchecked(self.low.width()) as usize },
        }
    }

    // Get (rank, bitvector index) from the offsets in `high` and `low`.
    fn combine(&self, pos: Pos) -> (usize, usize) {
        (pos.low, ((pos.high - pos.low) << self.low.width()) + (self.low.get(pos.low) as usize))
    }

    // Get the offsets in `high` and `low` for the set bit of the given rank.
    fn pos(&self, rank: usize) -> Pos {
        Pos {
            high: self.high.select(rank).unwrap(),
            low: rank,
        }
    }

    // Get a `Pos` that points to the first value with this high part or to the following
    // unset bit if no such values exist.
    fn lower_bound(&self, high_part: usize) -> Pos {
        if high_part == 0 {
            Pos { high: 0, low: 0, }
        } else {
            let high_offset = self.high.select_zero(high_part - 1).unwrap() + 1;
            Pos { high: high_offset, low: high_offset - high_part, }
        }
    }

    // Get a `Pos` that points to the unset bit after the values with the this high part.
    fn upper_bound(&self, high_part: usize) -> Pos {
        let high_offset = self.high.select_zero(high_part).unwrap();
        Pos { high: high_offset, low: high_offset - high_part, }
    }

    // Returns (run rank, one_iter past the run) for the run of 0s that contains
    // unset bit of the given rank.
    fn find_zero_run(&self, rank: usize) -> (usize, OneIter) {
        let mut low = 0;
        let mut high = self.count_ones();
        let mut result = (0, self.one_iter());

        // Invariant: `self.rank_zero(high) > rank`.
        while high - low > Self::BINARY_SEARCH_THRESHOLD {
            let mid = low + (high - low) / 2;
            let mut iter = self.select_iter(mid);
            let (_, mid_pos) = iter.next().unwrap();
            if mid_pos - mid <= rank {
                result = (mid + 1, iter);
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        // Once we have only a few runs left, a linear scan is faster than
        // `high.select()`.
        let mut iter = result.1.clone();
        while let Some((mid, mid_pos)) = iter.next() {
            if mid_pos - mid <= rank {
                result = (mid + 1, iter.clone());
            } else {
                break;
            }
        }

        result
    }
}

//-----------------------------------------------------------------------------

/// Space-efficient [`SparseVector`] construction.
///
/// A `SparseBuilder` allocates the data structures based on universe size (bitvector length) and the number of set bits.
/// The set bits must then be indicated in order using [`SparseBuilder::set`] or the [`Extend`] trait.
/// Once the builder is full, it can be converted into a [`SparseVector`] using the [`TryFrom`] trait.
/// The conversion will not fail if the builder is full.
///
/// Setting a bit `i` fails if the builder is full or the index is too small (`i < self.next_index()`) or too large (`i > self.universe()`).
/// [`Extend::extend`] will panic in such situations.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::BitVec;
/// use simple_sds::sparse_vector::{SparseVector, SparseBuilder};
/// use std::convert::TryFrom;
///
/// let mut builder = SparseBuilder::new(300, 5).unwrap();
/// assert_eq!(builder.len(), 0);
/// assert_eq!(builder.capacity(), 5);
/// assert_eq!(builder.universe(), 300);
/// assert_eq!(builder.next_index(), 0);
/// assert!(!builder.is_full());
/// assert!(!builder.is_multiset());
///
/// builder.set(12);
/// assert_eq!(builder.len(), 1);
/// assert_eq!(builder.next_index(), 13);
///
/// // This will return an error because the index is too small.
/// let _ = builder.try_set(10);
/// assert_eq!(builder.len(), 1);
/// assert_eq!(builder.next_index(), 13);
///
/// let v: Vec<usize> = vec![24, 48, 96, 192];
/// builder.extend(v);
/// assert_eq!(builder.len(), 5);
/// assert!(builder.is_full());
///
/// let sv = SparseVector::try_from(builder).unwrap();
/// assert_eq!(sv.len(), 300);
/// assert_eq!(sv.count_ones(), 5);
/// ```
#[derive(Clone, Debug)]
pub struct SparseBuilder {
    data: SparseVector,
    // We need a mutable bitvector during construction.
    high: RawVector,
    // Number of bits already set.
    len: usize,
    // The first index that can be set.
    next: usize,
    // `0` if we are building a multiset, `1` if not.
    increment: usize,
}

impl SparseBuilder {
    /// Returns an empty `SparseBuilder` without multiset semantics.
    ///
    /// Returns [`Err`] if `ones > universe`.
    ///
    /// # Arguments
    ///
    /// * `universe`: Universe size or length of the bitvector.
    /// * `ones`: Number of bits that will be set in the bitvector.
    pub fn new(universe: usize, ones: usize) -> Result<SparseBuilder, &'static str> {
        if ones > universe {
            return Err("Number of set bits is greater than universe size");
        }

        let (width, high_len) = Self::get_params(universe, ones);
        let low = IntVector::with_len(ones, width, 0).unwrap();
        let data = SparseVector {
            len: universe,
            high: BitVector::from(RawVector::new()),
            low,
        };

        let high = RawVector::with_len(high_len, false);
        Ok(SparseBuilder {
            data,
            high,
            len: 0,
            next: 0,
            increment: 1,
        })
    }

    /// Returns an empty `SparseBuilder` with multiset semantics.
    ///
    /// # Arguments
    ///
    /// * `universe`: Universe size or length of the bitvector.
    /// * `ones`: Number of bits that will be set in the bitvector.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::ops::BitVec;
    /// use simple_sds::sparse_vector::{SparseVector, SparseBuilder};
    /// use std::convert::TryFrom;
    ///
    /// let mut builder = SparseBuilder::multiset(120, 3);
    /// assert_eq!(builder.capacity(), 3);
    /// assert_eq!(builder.universe(), 120);
    /// assert!(builder.is_multiset());
    ///
    /// builder.set(12);
    /// builder.set(24);
    /// builder.set(24);
    /// assert!(builder.is_full());
    ///
    /// let sv = SparseVector::try_from(builder).unwrap();
    /// assert_eq!(sv.len(), 120);
    /// assert_eq!(sv.count_ones(), 3);
    /// assert!(sv.is_multiset());
    /// ```
    pub fn multiset(universe: usize, ones: usize) -> SparseBuilder {
        let (width, high_len) = Self::get_params(universe, ones);
        let low = IntVector::with_len(ones, width, 0).unwrap();
        let data = SparseVector {
            len: universe,
            high: BitVector::from(RawVector::new()),
            low,
        };

        let high = RawVector::with_len(high_len, false);
        SparseBuilder {
            data,
            high,
            len: 0,
            next: 0,
            increment: 0,
        }
    }

    /// Returns `true` if the builder is using multiset semantics.
    pub fn is_multiset(&self) -> bool {
        self.increment == 0
    }

    // Returns `(low.width(), high.len())`. Now works with overfull multisets as well.
    fn get_params(universe: usize, ones: usize) -> (usize, usize) {
        let mut low_width: usize = 1;
        if ones > 0 && ones <= universe {
            let ideal_width = ((universe as f64 * 2.0_f64.ln()) / (ones as f64)).log2();
            low_width = ideal_width.max(1.0).round() as usize;
        }
        let buckets = Self::get_buckets(universe, low_width);
        (low_width, ones + buckets)
    }

    // Returns `high.len()` for the given `universe` and `low.width()`.
    fn get_buckets(universe: usize, low_width: usize) -> usize {
        let mut buckets = if low_width < bits::WORD_BITS { universe >> low_width } else { 0 };
        if universe & (bits::low_set(low_width) as usize) != 0 {
            buckets += 1;
        }
        buckets
    }

    /// Returns the number of bits that have already been set.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the number of bits that can be set.
    pub fn capacity(&self) -> usize {
        self.data.count_ones()
    }

    /// Returns the universe size or the length of the bitvector.
    pub fn universe(&self) -> usize {
        self.data.len()
    }

    /// Returns the smallest index in the bitvector that can still be set.
    pub fn next_index(&self) -> usize {
        self.next
    }

    /// Returns `true` if all bits that can be set have been set.
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    /// Returns `true` if no bits have been set.
    ///
    /// Keeps Clippy happy.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sets the specified bit in the bitvector.
    ///
    /// # Panics
    ///
    /// Panics if the builder is full, if `index < self.next_index()`, or if `index >= self.universe()`.
    pub fn set(&mut self, index: usize) {
        self.try_set(index).unwrap();
    }

    /// Unsafe version of [`SparseBuilder::set`] without sanity checks.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if the builder is full, if `index < self.next_index()`, or if `index >= self.universe()`.
    pub unsafe fn set_unchecked(&mut self, index: usize) {
        let parts = self.data.split(index);
        self.high.set_bit(parts.high + self.len, true);
        self.data.low.set(self.len, parts.low as u64);
        self.len += 1; self.next = index + self.increment;
    }

    /// Tries to set the specified bit in the bitvector.
    ///
    /// Returns [`Err`] if the builder is full, if `index < self.next_index()`, or if `index >= self.universe()`.
    pub fn try_set(&mut self, index: usize) -> Result<(), &'static str> {
        if self.is_full() {
            return Err("The builder is full");
        }
        if index < self.next_index() {
            if self.increment == 0 {
                return Err("Index must be >= previous set position");
            } else {
                return Err("Index must be > previous set position");
            }
        }
        if index >= self.universe() {
            return Err("Index is larger than universe size");
        }
        unsafe { self.set_unchecked(index); }
        Ok(())
    }
}

impl Extend<usize> for SparseBuilder {
    fn extend<T: IntoIterator<Item = usize>>(&mut self, iter: T) {
        for index in iter {
            self.set(index);
        }
    }
}

impl TryFrom<SparseBuilder> for SparseVector {
    type Error = &'static str;

    fn try_from(builder: SparseBuilder) -> Result<Self, Self::Error> {
        let mut builder = builder;
        if !builder.is_full() {
            return Err("The builder is not full");
        }
        builder.data.high = BitVector::from(builder.high);
        builder.data.high.enable_select();
        builder.data.high.enable_select_zero();
        Ok(builder.data)
    }
}

//-----------------------------------------------------------------------------

/// A read-only iterator over [`SparseVector`].
///
/// The type of `Item` is [`bool`].
///
/// # Examples
///
/// ```
/// use simple_sds::ops::BitVec;
/// use simple_sds::sparse_vector::{SparseVector, SparseBuilder};
/// use std::convert::TryFrom;
///
/// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
/// let ones = source.iter().filter(|&b| *b).count();
/// let mut builder = SparseBuilder::new(source.len(), ones).unwrap();
/// for (index, _) in source.iter().enumerate().filter(|v| *v.1) {
///     builder.set(index);
/// }
/// let sv = SparseVector::try_from(builder).unwrap();
///
/// assert_eq!(sv.iter().len(), source.len());
/// for (index, value) in sv.iter().enumerate() {
///     assert_eq!(source[index], value);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Iter<'a> {
    parent: OneIter<'a>,
    // The first index we have not visited.
    next: usize,
    // The first set bit we have not visited.
    next_set: Option<usize>,
    // The first index we should not visit.
    limit: usize,
    // The last set bit we have not visited.
    last_set: Option<usize>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.limit {
            return None;
        }
        match self.next_set {
            Some(value) => {
                if value == self.next {
                    // We have to find the next unvisited (unique) value, and `last_set` is the initial candidate.
                    self.next_set = self.last_set;
                    // Skip duplicates until we find a new value or run out of values.
                    for (_, index) in self.parent.by_ref() {
                        if index > self.next {
                            self.next_set = Some(index);
                            break;
                        }
                    }
                    self.next += 1;
                    Some(true)
                } else {
                    self.next += 1;
                    Some(false)
                }
            },
            None => {
                self.next += 1;
                Some(false)
            },
        }
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
            return None;
        }
        self.limit -= 1;
        match self.last_set {
            Some(value) => {
                if value == self.limit {
                    // We have to find the previous unvisited (unique) value, and `next_set` is the initial candidate.
                    self.last_set = self.next_set;
                    // Skip duplicates until we find a new value or run out of values.
                    while let Some((_, index)) = self.parent.next_back() {
                        if index < self.limit {
                            self.last_set = Some(index);
                            break;
                        }
                    }
                    Some(true)
                } else {
                    Some(false)
                }
            },
            None => Some(false),
        }
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {}

impl<'a> FusedIterator for Iter<'a> {}

//-----------------------------------------------------------------------------

impl<'a> BitVec<'a> for SparseVector {
    type Iter = Iter<'a>;

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn count_ones(&self) -> usize {
        self.low.len()
    }

    // Override the default implementation, because it may underflow with multisets.
    #[inline]
    fn count_zeros(&self) -> usize {
        if self.count_ones() >= self.len() {
            0
        } else {
            self.len() - self.count_ones()
        }
    }

    fn get(&self, index: usize) -> bool {
        // Find the first value with the same high part, if it exists.
        let parts = self.split(index);
        let mut pos = self.lower_bound(parts.high);

        // Iterate forward over the values with the same high part until we find
        // a value no less than `value` or we run out of such values.
        while pos.high < self.high.len() && self.high.get(pos.high) {
            let low = self.low.get(pos.low) as usize;
            if low >= parts.low {
                return low == parts.low;
            }
            pos.high += 1; pos.low += 1;
        }

        false
    }

    fn iter(&'a self) -> Self::Iter {
        let mut one_iter = self.one_iter();
        let next_set = if let Some((_, index)) = one_iter.next() {
            Some(index)
        } else {
            None
        };
        let last_set = if let Some((_, index)) = one_iter.next_back() {
            Some(index)
        } else {
            next_set
        };
        Self::Iter {
            parent: one_iter,
            next: 0,
            next_set,
            limit: self.len(),
            last_set,
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> Rank<'a> for SparseVector {
    fn supports_rank(&self) -> bool {
        true
    }

    fn enable_rank(&mut self) {}

    fn rank(&self, index: usize) -> usize {
        if index >= self.len() {
            return self.count_ones();
        }

        // Find the last value with the same high part, if it exists.
        let parts = self.split(index);
        let mut pos = self.upper_bound(parts.high);
        if pos.low == 0 {
            return 0;
        }
        pos.high -= 1; pos.low -= 1;

        // Iterate backward over the values with the same high part until we find
        // as value lower than `index` or we run out of such values.
        while self.high.get(pos.high) && (self.low.get(pos.low) as usize) >= parts.low {
            if pos.low == 0 {
                return 0;
            }
            pos.high -= 1; pos.low -= 1;
        }

        pos.low + 1
    }
}

//-----------------------------------------------------------------------------

/// An iterator over the set bits in [`SparseVector`].
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
/// # Examples
///
/// ```
/// use simple_sds::ops::{BitVec, Select};
/// use simple_sds::sparse_vector::{SparseVector, SparseBuilder};
/// use std::convert::TryFrom;
///
/// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
/// let ones = source.iter().filter(|&b| *b).count();
/// let mut builder = SparseBuilder::new(source.len(), ones).unwrap();
/// for (index, _) in source.iter().enumerate().filter(|v| *v.1) {
///     builder.set(index);
/// }
/// let sv = SparseVector::try_from(builder).unwrap();
///
/// let mut iter = sv.one_iter();
/// assert_eq!(iter.len(), ones);
/// assert_eq!(iter.next(), Some((0, 0)));
/// assert_eq!(iter.next(), Some((1, 2)));
/// assert_eq!(iter.next(), Some((2, 3)));
/// assert_eq!(iter.next(), Some((3, 5)));
/// assert_eq!(iter.next(), Some((4, 6)));
/// assert!(iter.next().is_none());
/// ```
#[derive(Clone, Debug)]
pub struct OneIter<'a> {
    parent: &'a SparseVector,
    // The first position we have not visited.
    next: Pos,
    // The first position we should not visit.
    limit: Pos,
}

impl<'a> OneIter<'a> {
    // Build an empty iterator for the parent bitvector.
    fn empty_iter(parent: &'a SparseVector) -> OneIter<'a> {
        OneIter {
            parent,
            next: Pos { high: parent.high.len(), low: parent.low.len(), },
            limit: Pos { high: parent.high.len(), low: parent.low.len(), },
        }
    }
}

impl<'a> Iterator for OneIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.low >= self.limit.low {
            None
        } else {
            while !self.parent.high.get(self.next.high) {
                self.next.high += 1;
            }
            let result = self.parent.combine(self.next);
            self.next.high += 1; self.next.low += 1;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.limit.low - self.next.low;
        (remaining, Some(remaining))
    }
}

impl<'a> DoubleEndedIterator for OneIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.next.low >= self.limit.low {
            None
        } else {
            self.limit.high -= 1; self.limit.low -= 1;
            while !self.parent.high.get(self.limit.high) {
                self.limit.high -= 1;
            }
            Some(self.parent.combine(self.limit))
        }
    }
}

impl<'a> ExactSizeIterator for OneIter<'a> {}

impl<'a> FusedIterator for OneIter<'a> {}

//-----------------------------------------------------------------------------

/// An iterator over the unset bits in [`SparseVector`].
///
/// The type of `Item` is `(`[`usize`]`, `[`usize`]`)`.
/// This can be interpreted as:
///
/// * `(index, value)` or `(i, select(i))` in the integer array of the complement; or
/// * `(rank(j), j)` in the bit array with `j` such that `self.get(j) == false`.
///
/// Note that `index` is not always the index provided by [`Iterator::enumerate`].
/// Queries may create iterators in the middle of the bitvector.
///
/// This iterator does not work correctly with multisets.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{BitVec, SelectZero};
/// use simple_sds::sparse_vector::{SparseVector, SparseBuilder};
/// use std::convert::TryFrom;
///
/// let source: Vec<bool> = vec![true, false, true, true, false, true, true, false];
/// let ones = source.iter().filter(|&b| *b).count();
/// let mut builder = SparseBuilder::new(source.len(), ones).unwrap();
/// for (index, _) in source.iter().enumerate().filter(|v| *v.1) {
///     builder.set(index);
/// }
/// let sv = SparseVector::try_from(builder).unwrap();
///
/// let mut iter = sv.zero_iter();
/// assert_eq!(iter.len(), source.len() - ones);
/// assert_eq!(iter.next(), Some((0, 1)));
/// assert_eq!(iter.next(), Some((1, 4)));
/// assert_eq!(iter.next(), Some((2, 7)));
/// assert!(iter.next().is_none());
/// ```
#[derive(Clone, Debug)]
pub struct ZeroIter<'a> {
    iter: OneIter<'a>,
    // The position of the next one, or the length of the bitvector.
    one_pos: usize,
    // The first position we have not visited.
    next: (usize, usize),
    // The first position we should not visit.
    limit: (usize, usize),
}

impl<'a> ZeroIter<'a> {
    // Build an empty iterator for the parent bitvector.
    fn empty_iter(parent: &'a SparseVector) -> ZeroIter<'a> {
        ZeroIter {
            iter: OneIter::empty_iter(parent),
            one_pos: 0,
            next: (0, 0),
            limit: (0, 0),
        }
    }

    // Go to the next run of zeros if necessary, assuming that we are not at the end.
    fn next_run(&mut self) {
        while self.next.1 >= self.one_pos {
            self.next.1 = self.one_pos + 1;
            self.one_pos = if let Some((_, pos)) = self.iter.next() { pos } else { self.limit.1 };
        }
    }
}

impl<'a> Iterator for ZeroIter<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.0 >= self.limit.0 {
            None
        } else {
            self.next_run();
            let result = self.next;
            self.next.0 += 1;
            self.next.1 += 1;
            Some(result)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.limit.0 - self.next.0;
        (remaining, Some(remaining))
    }
}

// TODO: DoubleEndedIterator?

impl<'a> ExactSizeIterator for ZeroIter<'a> {}

impl<'a> FusedIterator for ZeroIter<'a> {}

//-----------------------------------------------------------------------------

impl<'a> Select<'a> for SparseVector {
    type OneIter = OneIter<'a>;

    fn supports_select(&self) -> bool {
        true
    }

    fn enable_select(&mut self) {}

    fn one_iter(&'a self) -> Self::OneIter {
        Self::OneIter {
            parent: self,
            next: Pos { high: 0, low: 0, },
            limit: Pos { high: self.high.len(), low: self.low.len(), },
        }
    }

    fn select(&'a self, rank: usize) -> Option<usize> {
         if rank >= self.count_ones() {
             None
        } else {
            Some(self.combine(self.pos(rank)).1)
        }
    }

    fn select_iter(&'a self, rank: usize) -> Self::OneIter {
         if rank >= self.count_ones() {
             Self::OneIter::empty_iter(self)
        } else {
            Self::OneIter {
                parent: self,
                next: self.pos(rank),
                limit: Pos { high: self.high.len(), low: self.low.len(), },
            }
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> SelectZero<'a> for SparseVector {
    type ZeroIter = ZeroIter<'a>;

    fn supports_select_zero(&self) -> bool {
        true
    }

    fn enable_select_zero(&mut self) {}

    fn zero_iter(&'a self) -> Self::ZeroIter {
        let mut iter = self.one_iter();
        let one_pos = if let Some((_, pos)) = iter.next() { pos } else { self.len() };
        ZeroIter {
            iter,
            one_pos,
            next: (0, 0),
            limit: (self.count_zeros(), self.len()),
        }
    }

    fn select_zero(&'a self, rank: usize) -> Option<usize> {
        if rank >= self.count_zeros() {
            return None;
        }
        let (run_rank, _) = self.find_zero_run(rank);
        Some(run_rank + rank)
    }

    fn select_zero_iter(&'a self, rank: usize) -> Self::ZeroIter {
        if rank >= self.count_zeros() {
            return Self::ZeroIter::empty_iter(self);
        }
        let (run_rank, mut iter) = self.find_zero_run(rank);
        let one_pos = if let Some((_, pos)) = iter.next() { pos } else { self.len() };
        ZeroIter {
            iter,
            one_pos,
            next: (rank, run_rank + rank),
            limit: (self.count_zeros(), self.len()),
        }
    }
}

//-----------------------------------------------------------------------------

impl<'a> PredSucc<'a> for SparseVector {
    type OneIter = OneIter<'a>;

    fn supports_pred_succ(&self) -> bool {
        true
    }

    fn enable_pred_succ(&mut self) {}

    fn predecessor(&'a self, value: usize) -> Self::OneIter {
        if self.is_empty() {
            return Self::OneIter::empty_iter(self);
        }

        // Find the last value with the same high part, if it exists.
        let parts = self.split(cmp::min(value, self.len() - 1));
        let mut pos = self.upper_bound(parts.high);
        if pos.low == 0 {
            return Self::OneIter::empty_iter(self);
        }
        pos.high -= 1; pos.low -= 1;

        // Iterate backward over the values with the same high part until we find
        // a value no greater than `value` or we run out of such values.
        while self.high.get(pos.high) && (self.low.get(pos.low) as usize) > parts.low {
            if pos.low == 0 {
                return Self::OneIter::empty_iter(self);
            }
            pos.high -= 1; pos.low -= 1;
        }

        // The predecessor has a lower high part, so we continue iterating until we find it.
        while !self.high.get(pos.high) {
            pos.high -= 1;
        }

        Self::OneIter {
            parent: self,
            next: pos,
            limit: Pos { high: self.high.len(), low: self.low.len(), },
        }
    }

    fn successor(&'a self, value: usize) -> Self::OneIter {
        if value >= self.len() {
            return Self::OneIter::empty_iter(self);
        }

        // Find the first value with the same high part, if it exists.
        let parts = self.split(value);
        let mut pos = self.lower_bound(parts.high);

        // Iterate forward over the values with the same high part until we find
        // a value no less than `value` or we run out of such values.
        while pos.high < self.high.len() && self.high.get(pos.high) {
            if (self.low.get(pos.low) as usize) >= parts.low {
                return Self::OneIter {
                    parent: self,
                    next: pos,
                    limit: Pos { high: self.high.len(), low: self.low.len(), },
                };
            }
            pos.high += 1; pos.low += 1;
        }

        // The successor has a greater high part, so we continue iterating until we find it.
        while pos.high < self.high.len() {
            if self.high.get(pos.high) {
                return Self::OneIter {
                    parent: self,
                    next: pos,
                    limit: Pos { high: self.high.len(), low: self.low.len(), },
                };
            }
            pos.high += 1;
        }

        Self::OneIter::empty_iter(self)
    }
}

//-----------------------------------------------------------------------------

impl Serialize for SparseVector {
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.len.serialize(writer)?;
        Ok(())
    }

    fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.high.serialize(writer)?;
        self.low.serialize(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let len = usize::load(reader)?;
        let mut high = BitVector::load(reader)?;
        let low = IntVector::load(reader)?;

        // Enable support structures, because the data may be from a library that does not know
        // how to build them.
        high.enable_select();
        high.enable_select_zero();

        // Sanity checks.
        if low.len() != high.count_ones() {
            return Err(Error::new(ErrorKind::InvalidData, "Inconsistent number of set bits"));
        }
        if high.len() != low.len() + SparseBuilder::get_buckets(len, low.width()){
            return Err(Error::new(ErrorKind::InvalidData, "Invalid number of buckets"));
        }

        let result = SparseVector {
            len, high, low,
        };
        Ok(result)
    }

    fn size_in_elements(&self) -> usize {
        self.len.size_in_elements() +
        self.high.size_in_elements() +
        self.low.size_in_elements()
    }
}

//-----------------------------------------------------------------------------
