//! Operations common to various vectors.
//!
//! # Integer vectors
//!
//! * [`Vector`]: Basic operations.
//! * [`Resize`]: Resizable vectors.
//! * [`Pack`]: Space-efficiency by e.g. bit packing.
//! * [`Access`]: Random access.
//! * [`Push`], [`Pop`]: Stack operations.
//! * [`VectorIndex`]: Rank, select, predecessor, and successor queries.
//!
//! # Bitvectors
//!
//! * [`BitVec`]: Random access and iterators over all bits.
//! * [`Rank`]: Rank queries.
//! * [`Select`]: Select queries and iterators over set bits.
//! * [`SelectZero`]: Select queries on the complement and iterators over unset bits.
//! * [`PredSucc`]: Predecessor and successor queries.

use std::iter::FusedIterator;
use std::cmp;

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

/// A vector that contains items of a fixed type.
///
/// The type of the item must implement [`Eq`] and [`Copy`].
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{Vector, Resize, Pack, Access, AccessIter};
/// use simple_sds::bits;
///
/// #[derive(Clone, Debug, PartialEq, Eq)]
/// struct Example(Vec<u8>);
///
/// impl Example {
///     fn new() -> Example {
///         Example(Vec::new())
///     }
/// }
///
/// impl Vector for Example {
///     type Item = u8;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         8
///     }
///
///     fn max_len(&self) -> usize {
///         usize::MAX
///     }
/// } 
///
/// impl Resize for Example {
///     fn resize(&mut self, new_len: usize, value: Self::Item) {
///         self.0.resize(new_len, value);
///     }
///
///     fn clear(&mut self) {
///         self.0.clear();
///     }
///
///     fn capacity(&self) -> usize {
///         self.0.capacity()
///     }
///
///     fn reserve(&mut self, additional: usize) {
///         self.0.reserve(additional);
///     }
/// }
///
/// // Packing does not make much sense with the running example.
/// impl Pack for Example {
///     fn pack(&mut self) {}
/// }
///
/// impl<'a> Access<'a> for Example {
///     type Iter = AccessIter<'a, Self>;
///
///     fn get(&self, index: usize) -> Self::Item {
///         self.0[index]
///     }
///
///     fn iter(&'a self) -> Self::Iter {
///         Self::Iter::new(self)
///     }
///
///     fn is_mutable(&self) -> bool {
///         true
///     }
///
///     fn set(&mut self, index: usize, value: Self::Item) {
///         self.0[index] = value
///     }
/// }
///
/// // Vector
/// let mut v = Example::new();
/// assert!(v.is_empty());
/// assert_eq!(v.len(), 0);
/// assert_eq!(v.width(), bits::bit_len(u8::MAX as u64));
///
/// // Resize
/// v.reserve(4);
/// assert!(v.capacity() >= 4);
/// v.resize(4, 0);
/// assert_eq!(v.len(), 4);
/// v.clear();
/// assert!(v.is_empty());
///
/// // Pack
/// let mut v = Example(vec![1, 2, 3]);
/// v.pack();
/// assert_eq!(v.len(), 3);
///
/// // Access
/// assert!(v.is_mutable());
/// for i in 0..v.len() {
///     assert_eq!(v.get(i), (i + 1) as u8);
///     v.set(i, i as u8);
///     assert_eq!(v.get(i), i as u8);
/// }
/// let extracted: Vec<u8> = v.iter().collect();
/// assert_eq!(extracted, vec![0, 1, 2]);
/// ```
pub trait Vector {
    /// The type of the items in the vector.
    type Item: Eq + Copy;

    /// Returns the number of items in the vector.
    fn len(&self) -> usize;

    /// Returns `true` if the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the width of of an item in bits.
    fn width(&self) -> usize;

    /// Returns the maximum length of the vector.
    fn max_len(&self) -> usize;
}

/// A vector that can be resized.
///
/// See [`Vector`] for an example.
pub trait Resize: Vector {
    /// Resizes the vector to a specified length.
    ///
    /// If `new_len > self.len()`, the new `new_len - self.len()` values will be initialized.
    /// If `new_len < self.len()`, the vector is truncated.
    ///
    /// # Arguments
    ///
    /// * `new_len`: New length of the vector.
    /// * `value`: Initialization value.
    ///
    /// # Panics
    ///
    /// May panic if the length would exceed the maximum length.
    fn resize(&mut self, new_len: usize, value: <Self as Vector>::Item);

    /// Clears the vector without freeing the data.
    fn clear(&mut self);

    /// Returns the number of items that the vector can store without reallocations.
    fn capacity(&self) -> usize;

    /// Reserves space for storing at least `self.len() + additional` items in the vector.
    ///
    /// Does nothing if the capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// May panic if the capacity would exceed the maximum length.
    fn reserve(&mut self, additional: usize);
}

/// Store the vector more space-efficiently.
///
/// This may, for example, reduce the width of an item.
///
/// See [`Vector`] for an example.
pub trait Pack: Vector {
    /// Try to store the items of the vector more space-efficiently.
    fn pack(&mut self);
}

//-----------------------------------------------------------------------------

/// A vector that supports random access to its items.
///
/// The default implementations of [`Access::is_mutable`] and [`Access::set`] make the vector immutable.
///
/// See [`Vector`] for an example and [`AccessIter`] for a possible implementation of the iterator.
pub trait Access<'a>: Vector {
    /// Iterator over the items in the vector.
    type Iter: Iterator<Item = <Self as Vector>::Item> + ExactSizeIterator;

    /// Returns an item from the vector.
    ///
    /// # Panics
    ///
    /// May panic if `index` is not a valid index in the vector.
    /// May panic from I/O errors.
    fn get(&self, index: usize) -> <Self as Vector>::Item;

    /// Returns an item from the vector or the provided value if `index` is invalid.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn get_or(&self, index: usize, value: <Self as Vector>::Item) -> <Self as Vector>::Item {
        if index >= self.len() { value } else { self.get(index) }
    }

    /// Returns an iterator over the items in the vector.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reason.
    fn iter(&'a self) -> Self::Iter;

    /// Returns `true` if the underlying data is mutable.
    ///
    /// This is relevant, for example, with memory-mapped vectors, where the underlying file may be opened as read-only.
    #[inline]
    fn is_mutable(&self) -> bool {
        false
    }

    /// Sets an item in the vector.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `value`: New value of the item.
    ///
    /// # Panics
    ///
    /// May panic if `index` is not a valid index in the vector.
    /// May panic if the underlying data is not mutable.
    /// May panic from I/O errors.
    fn set(&mut self, _: usize, _: <Self as Vector>::Item) {
        panic!("Access::set(): The default implementation is immutable");
    }
}

//-----------------------------------------------------------------------------

/// A read-only iterator over a [`Vector`] that implements [`Access`].
///
/// The iterator uses [`Access::get`] and has efficient implementations of [`Iterator::nth`] and [`DoubleEndedIterator::nth_back`].
///
/// See [`Vector`] for an example.
#[derive(Clone, Debug)]
pub struct AccessIter<'a, VectorType: Access<'a>> {
    parent: &'a VectorType,
    // The first index we have not used.
    next: usize,
    // The first index we should not use.
    limit: usize,
}

impl<'a, VectorType: Access<'a>> AccessIter<'a, VectorType> {
    // Creates a new iterator over the vector.
    pub fn new(parent: &'a VectorType) -> Self {
        AccessIter {
            parent,
            next: 0,
            limit: parent.len(),
        }
    }
}

impl<'a, VectorType: Access<'a>> Iterator for AccessIter<'a, VectorType> {
    type Item = <VectorType as Vector>::Item;

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

impl<'a, VectorType: Access<'a>> DoubleEndedIterator for AccessIter<'a, VectorType> {
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

impl<'a, VectorType: Access<'a>> ExactSizeIterator for AccessIter<'a, VectorType> {}

impl<'a, VectorType: Access<'a>> FusedIterator for AccessIter<'a, VectorType> {}

//-----------------------------------------------------------------------------

/// Append items to a vector.
///
/// [`Pop`] is a separate trait, because a file writer may not implement it.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{Vector, Push, Pop};
///
/// struct Example(Vec<u8>);
///
/// impl Example {
///     fn new() -> Example {
///         Example(Vec::new())
///     }
/// }
///
/// impl Vector for Example {
///     type Item = u8;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         8
///     }
///
///     fn max_len(&self) -> usize {
///         usize::MAX
///     }
/// } 
///
/// impl Push for Example {
///     fn push(&mut self, value: Self::Item) {
///         self.0.push(value);
///     }
/// }
///
/// impl Pop for Example {
///     fn pop(&mut self) -> Option<Self::Item> {
///         self.0.pop()
///     }
/// }
///
/// // Push
/// let mut v = Example::new();
/// assert!(v.is_empty());
/// v.push(1);
/// v.push(2);
/// v.push(3);
/// assert_eq!(v.len(), 3);
///
/// // Pop
/// assert_eq!(v.pop(), Some(3));
/// assert_eq!(v.pop(), Some(2));
/// assert_eq!(v.pop(), Some(1));
/// assert!(v.pop().is_none());
/// assert!(v.is_empty());
/// ```
pub trait Push: Vector {
    /// Appends an item to the vector.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// May panic if the vector would exceed the maximum length.
    fn push(&mut self, value: <Self as Vector>::Item);
}

/// Remove and return top items from a vector.
///
/// [`Push`] is a separate trait, because a file writer may not implement `Pop`.
///
/// See [`Push`] for an example.
pub trait Pop: Vector {
    /// Removes and returns the last item from the vector.
    ///
    /// Returns [`None`] if there are no more items in the vector.
    fn pop(&mut self) -> Option<<Self as Vector>::Item>;
}

//-----------------------------------------------------------------------------

/// Rank, select, predecessor, and successor queries on a vector.
///
/// Position `index` of the vector is an occurrence of item `value` of rank `rank`, if `self.get(index) == value` and `self.rank(index) == rank`.
///
/// This generalizes [`Rank`], [`Select`], [`SelectZero`], and [`PredSucc`] from binary vectors to vectors with arbitrary items.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{Vector, Access, AccessIter, VectorIndex};
/// use std::cmp;
///
/// #[derive(Clone, Debug, PartialEq, Eq)]
/// struct Example(Vec<char>);
///
/// impl Vector for Example {
///     type Item = char;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         32
///     }
///
///     fn max_len(&self) -> usize {
///         usize::MAX
///     }
/// }
///
/// impl<'a> Access<'a> for Example {
///     type Iter = AccessIter<'a, Self>;
///
///     fn get(&self, index: usize) -> Self::Item {
///         self.0[index]
///     }
///
///     fn iter(&'a self) -> Self::Iter {
///         Self::Iter::new(self)
///     }
/// }
///
/// struct ValueIter<'a> {
///     parent: &'a Example,
///     value: char,
///     rank: usize,
///     index: usize,
/// }
///
/// impl<'a> Iterator for ValueIter<'a> {
///     type Item = (usize, usize);
///
///     fn next(&mut self) -> Option<Self::Item> {
///         while self.index < self.parent.len() {
///             if self.parent.get(self.index) == self.value {
///                 let result = Some((self.rank, self.index));
///                 self.rank += 1; self.index += 1;
///                 return result;
///             }
///             self.index += 1;
///         }
///         None
///     }
/// }
///
/// impl<'a> VectorIndex<'a> for Example {
///     type ValueIter = ValueIter<'a>;
///
///     fn rank(&self, index: usize, value: <Self as Vector>::Item) -> usize {
///         let index = cmp::min(index, self.len());
///         let mut result = 0;
///         for i in 0..index {
///             if self.get(i) == value {
///                 result += 1;
///             }
///         }
///         result
///     }
///
///     fn value_iter(&'a self, value: <Self as Vector>::Item) -> Self::ValueIter {
///         Self::ValueIter { parent: self, value, rank: 0, index: 0, }
///     }
///
///     fn value_of(iter: &Self::ValueIter) -> <Self as Vector>::Item {
///         iter.value
///     }
///
///     fn select(&self, rank: usize, value: <Self as Vector>::Item) -> Option<usize> {
///         let mut found = 0;
///         for index in 0..self.len() {
///             if self.get(index) == value {
///                 if found == rank {
///                     return Some(index);
///                 }
///                 found += 1;
///             }
///         }
///         None
///     }
///
///     fn select_iter(&'a self, rank: usize, value: <Self as Vector>::Item) -> Self::ValueIter {
///         let index = self.select(rank, value).unwrap_or(self.len());
///         Self::ValueIter { parent: self, value, rank, index, }
///     }
/// }
///
/// let vec = Example(vec!['a', 'b', 'c', 'b', 'a', 'b', 'c', 'c']);
///
/// // Has item
/// assert!(vec.contains('b'));
/// assert!(!vec.contains('d'));
///
/// // Rank
/// assert_eq!(vec.rank(5, 'b'), 2);
/// assert_eq!(vec.rank(6, 'b'), 3);
///
/// // Iterator
/// let a: Vec<(usize, usize)> = vec.value_iter('a').collect();
/// assert_eq!(a, vec![(0, 0), (1, 4)]);
/// assert_eq!(Example::value_of(&vec.value_iter('c')), 'c');
///
/// // Select
/// assert_eq!(vec.select(0, 'c'), Some(2));
/// let mut iter = vec.select_iter(1, 'c');
/// assert_eq!(iter.next(), Some((1, 6)));
/// assert_eq!(iter.next(), Some((2, 7)));
/// assert!(iter.next().is_none());
///
/// // Inverse select
/// let offset = 3;
/// let inverse = vec.inverse_select(offset).unwrap();
/// assert_eq!(inverse, (1, 'b'));
/// assert_eq!(vec.select(inverse.0, inverse.1), Some(offset));
///
/// // PredSucc
/// assert!(vec.predecessor(1, 'c').next().is_none());
/// assert_eq!(vec.predecessor(2, 'c').next(), Some((0, 2)));
/// assert_eq!(vec.predecessor(3, 'c').next(), Some((0, 2)));
/// assert_eq!(vec.successor(3, 'a').next(), Some((1, 4)));
/// assert_eq!(vec.successor(4, 'a').next(), Some((1, 4)));
/// assert!(vec.successor(5, 'a').next().is_none());
/// ```
pub trait VectorIndex<'a>: Access<'a> {
    /// Iterator type over the occurrences of a specific item.
    ///
    /// The `Item` in the iterator is a (rank, index) pair such that the value at position `index` is the occurrence of rank `rank`.
    type ValueIter: Iterator<Item = (usize, usize)>;

    /// Returns `true` if the vector contains an item with the given value.
    ///
    /// The default implementation uses [`VectorIndex::rank`].
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn contains(&self, value: <Self as Vector>::Item) -> bool {
        self.rank(self.len(), value) > 0
    }

    /// Returns the number of indexes `i < index` in vector such that `self.get(i) == value`.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn rank(&self, index: usize, value: <Self as Vector>::Item) -> usize;

    /// Computes the inverse of [`VectorIndex::select`], or returns [`None`] if `index` is invalid.
    ///
    /// Returns `(rank, value)` such that [`VectorIndex::select`]`(rank, value) == index`.
    /// The default implementation computes [`VectorIndex::rank`]`(index, `[`Access::get`]`(index))`.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn inverse_select(&self, index: usize) -> Option<(usize, <Self as Vector>::Item)> {
        if index >= self.len() {
            return None;
        }
        let value = self.get(index);
        Some((self.rank(index, value), value))
    }

    /// Returns an iterator over the occurrences of item `value`.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reason.
    fn value_iter(&'a self, value: <Self as Vector>::Item) -> Self::ValueIter;

    /// Returns the value of the items iterated over by the iterator.
    fn value_of(iter: &Self::ValueIter) -> <Self as Vector>::Item;

    /// Returns the index of the vector that contains the occurrence of item `value` of rank `rank`.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn select(&self, rank: usize, value: <Self as Vector>::Item) -> Option<usize>;

    /// Returns an iterator at the occurrence of item `value` of rank `rank`.
    ///
    /// The iterator will return [`None`] if the rank is out of bounds.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reason.
    fn select_iter(&'a self, rank: usize, value: <Self as Vector>::Item) -> Self::ValueIter;

    /// Returns an iterator at the last occurrence `i <= index` of item `value`.
    ///
    /// The iterator will return [`None`] if no such occurrence exists.
    /// The default implementation uses [`VectorIndex::rank`] and [`VectorIndex::select_iter`].
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reason.
    fn predecessor(&'a self, index: usize, value: <Self as Vector>::Item) -> Self::ValueIter {
        let rank = self.rank(index + 1, value);
        let rank = if rank > 0 { rank - 1 } else { self.len() };
        self.select_iter(rank, value)
    }

    /// Returns an iterator at the first occurrence `i >= index` of item `value`.
    ///
    /// The iterator will return [`None`] if no such occurrence exists.
    /// The default implementation uses [`VectorIndex::rank`] and [`VectorIndex::select_iter`].
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reason.
    fn successor(&'a self, index: usize, value: <Self as Vector>::Item) -> Self::ValueIter {
        let rank = self.rank(index, value);
        self.select_iter(rank, value)
    }
}

//-----------------------------------------------------------------------------

/// An immutable structure that can be seen as a bit array or a sorted array of distinct unsigned integers.
///
/// Let `A` be the integer array and `B` the bit array.
/// `B[i] == true` is then equivalent to value `i` being present in array `A`.
/// Indexes in both arrays are 0-based, and the ones in the integer array are often called ranks.
/// Because of the dual interpretations, bitvectors should not implement [`IntoIterator`].
///
/// Some operations deal with the complement of the bitvector.
/// In the bit array interpretation, all bits in the complement vector are flipped.
/// In the integer array interpretation, the complement contains the values in `0..self.len()` missing from the original.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{BitVec, Rank, Select, PredSucc};
/// use simple_sds::raw_vector::{RawVector, AccessRaw};
/// use simple_sds::bits;
/// use std::cmp;
///
/// struct NaiveBitVector {
///     ones: usize,
///     data: RawVector,
/// }
///
/// struct BitIter<'a> {
///     parent: &'a NaiveBitVector,
///     index: usize,
/// }
///
/// impl<'a> Iterator for BitIter<'a> {
///     type Item = bool;
///
///     fn next(&mut self) -> Option<Self::Item> {
///         if self.index < self.parent.len() {
///             let value = self.parent.get(self.index);
///             self.index += 1;
///             return Some(value);
///         }
///         None
///     }
///
///     fn size_hint(&self) -> (usize, Option<usize>) {
///         let remaining = self.parent.len() - self.index;
///         (remaining, Some(remaining))
///     }
/// }
///
/// impl<'a> ExactSizeIterator for BitIter<'a> {}
///
/// impl From<RawVector> for NaiveBitVector {
///     fn from(data: RawVector) -> Self {
///         let ones = data.count_ones();
///         NaiveBitVector {
///             ones: ones,
///             data: data,
///         }
///     }
/// }
///
/// impl<'a> BitVec<'a> for NaiveBitVector {
///     type Iter = BitIter<'a>;
///
///     fn len(&self) -> usize {
///         self.data.len()
///     }
///
///     fn count_ones(&self) -> usize {
///         self.ones
///     }
///
///     fn get(&self, index: usize) -> bool {
///         self.data.bit(index)
///     }
///
///     fn iter(&'a self) -> Self::Iter {
///         Self::Iter {
///             parent: self,
///             index: 0,
///         }
///     }
/// }
///
/// impl<'a> Rank<'a> for NaiveBitVector {
///     fn supports_rank(&self) -> bool {
///         true
///     }
///
///     fn enable_rank(&mut self) {}
///
///     fn rank(&self, index: usize) -> usize {
///         let mut result: usize = 0;
///         let index = cmp::min(self.len(), index);
///         for i in 0..index {
///             result += self.get(i) as usize;
///         }
///         result
///     }
/// }
///
/// struct SelectIter<'a> {
///     parent: &'a NaiveBitVector,
///     rank: usize,
///     index: usize,
/// }
///
/// impl<'a> Iterator for SelectIter<'a> {
///     type Item = (usize, usize);
///
///     fn next(&mut self) -> Option<Self::Item> {
///         while self.index < self.parent.len() {
///             if self.parent.get(self.index) {
///                 let result = Some((self.rank, self.index));
///                 self.rank += 1; self.index += 1;
///                 return result;
///             }
///             self.index += 1;
///         }
///         None
///     }
///
///     fn size_hint(&self) -> (usize, Option<usize>) {
///         let remaining = self.parent.count_ones() - self.rank;
///         (remaining, Some(remaining))
///     }
/// }
///
/// impl<'a> ExactSizeIterator for SelectIter<'a> {}
///
/// impl<'a> Select<'a> for NaiveBitVector {
///     type OneIter = SelectIter<'a>;
///
///     fn supports_select(&self) -> bool {
///         true
///     }
///
///     fn enable_select(&mut self) {}
///
///     fn one_iter(&'a self) -> Self::OneIter {
///         Self::OneIter {
///             parent: self,
///             rank: 0,
///             index: 0,
///         }
///     }
///
///     fn select(&'a self, rank: usize) -> Option<usize> {
///         let mut found: usize = 0;
///         for index in 0..self.len() {
///             if self.get(index) {
///                 if found == rank {
///                     return Some(index);
///                 }
///                 found += 1;
///             }
///         }
///         None
///     }
///
///     fn select_iter(&'a self, rank: usize) -> Self::OneIter {
///         let index = self.select(rank).unwrap_or(self.len());
///         Self::OneIter { parent: self, rank, index, }
///     }
/// }
///
/// impl<'a> PredSucc<'a> for NaiveBitVector {
///     type OneIter = SelectIter<'a>;
///
///     fn supports_pred_succ(&self) -> bool {
///         true
///     }
///
///     fn enable_pred_succ(&mut self) {}
///
///     fn predecessor(&'a self, value: usize) -> Self::OneIter {
///         let mut rank = self.rank(value + 1);
///         let index = if rank > 0 { rank = rank - 1; self.select(rank).unwrap() } else { self.len() };
///         Self::OneIter { parent: self, rank, index, }
///     }
///
///     fn successor(&'a self, value: usize) -> Self::OneIter {
///         let rank = self.rank(value);
///         let index = self.select(rank).unwrap_or(self.len());
///         Self::OneIter { parent: self, rank, index, }
///     }
/// }
///
/// let mut data = RawVector::with_len(137, false);
/// data.set_bit(1, true); data.set_bit(33, true); data.set_bit(95, true); data.set_bit(123, true);
/// let mut bv = NaiveBitVector::from(data);
///
/// // BitVec
/// assert_eq!(bv.len(), 137);
/// assert!(!bv.is_empty());
/// assert_eq!(bv.count_ones(), 4);
/// assert_eq!(bv.count_zeros(), 133);
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
pub trait BitVec<'a> {
    /// Iterator type over the bit array.
    type Iter: Iterator<Item = bool> + ExactSizeIterator;

    /// Returns the length of the bit array or the universe size of the integer array.
    fn len(&self) -> usize;

    /// Returns `true` if the bitvector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of the integer array or the number of ones in the bit array.
    ///
    /// Because the vector is immutable, the implementation should cache the value during construction.
    fn count_ones(&self) -> usize;

    /// Returns the number of zeros in the bit array.
    #[inline]
    fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }

    /// Reads a bit from the bit array.
    ///
    /// In the integer array interpretation, returns `true` if value `index` is in the array.
    ///
    /// # Panics
    ///
    /// May panic if `index` is not a valid index in the bit array.
    /// May panic from I/O errors.
    fn get(&self, index: usize) -> bool;

    /// Returns an iterator over the bit array.
    ///
    /// See traits [`Select`] and [`SelectZero`] for other iterators.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reason.
    fn iter(&'a self) -> Self::Iter;

    // TODO: add `copy_bit_vec`?
}

//-----------------------------------------------------------------------------

/// Rank queries on a bitvector.
///
/// Some bitvector types do not build rank/select support structures by default.
/// After the vector has been built, rank support can be enabled with `self.enable_rank()`.
///
/// See [`BitVec`] for an example.
pub trait Rank<'a>: BitVec<'a> {
    /// Returns `true` if rank support has been enabled.
    fn supports_rank(&self) -> bool;

    /// Enables rank support for the vector.
    ///
    /// No effect if rank support has already been enabled.
    fn enable_rank(&mut self);

    /// Returns the number of indexes `i < index` in the bit array such that `self.get(i) == true`.
    ///
    /// In the integer array interpretation, returns the number of values smaller than `index`.
    /// The semantics of the query are the same as in [SDSL](https://github.com/simongog/sdsl-lite).
    ///
    /// # Panics
    ///
    /// May panic if rank support has not been enabled.
    /// May panic from I/O errors.
    fn rank(&self, index: usize) -> usize;

    /// Returns the number of indexes `i < index` in the bit array such that `self.get(i) == false`.
    ///
    /// In the integer array interpretation, returns the number of missing values smaller than `index`.
    /// The semantics of the query are the same as in [SDSL](https://github.com/simongog/sdsl-lite).
    ///
    /// # Panics
    ///
    /// May panic if rank support has not been enabled.
    /// May panic from I/O errors.
    #[inline]
    fn rank_zero(&self, index: usize) -> usize {
        index - self.rank(index)
    }
}

//-----------------------------------------------------------------------------

/// Select queries on a bitvector.
///
/// Some bitvector types do not build rank/select support structures by default.
/// After the vector has been built, select support can be enabled with `self.enable_select()`.
///
/// See [`BitVec`] for an example.
pub trait Select<'a>: BitVec<'a> {
    /// Iterator type over (index, value) pairs in the integer array.
    ///
    /// The `Item` in the iterator is an (index, value) pair in the integer array.
    /// This can be interpreted as `(i, select(i))` or `(rank(j), j)`.
    type OneIter: Iterator<Item = (usize, usize)> + ExactSizeIterator;

    /// Returns `true` if select support has been enabled.
    fn supports_select(&self) -> bool;

    /// Enables select support for the vector.
    ///
    /// No effect if select support has already been enabled.
    fn enable_select(&mut self);

    /// Returns an iterator over the integer array.
    ///
    /// In the bit array interpretation, the iterator will visit all set bits.
    /// The iterator must not require enabling select support.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reason.
    fn one_iter(&'a self) -> Self::OneIter;

    /// Returns the specified value in the integer array or [`None`] if no such value exists.
    ///
    /// In the bit array interpretation, the return value is an index `i` such that `self.get(i) == true` and `self.rank(i) == rank`.
    /// This trait uses 0-based indexing, while the [SDSL](https://github.com/simongog/sdsl-lite) select uses 1-based indexing.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled.
    /// May panic from I/O errors.
    fn select(&'a self, rank: usize) -> Option<usize>;

    /// Returns an iterator at the specified rank in the integer array.
    ///
    /// The iterator will return [`None`] if the rank is out of bounds.
    /// In the bit array interpretation, the iterator points to the set bit of the specified rank.
    /// This means a bit array index `i` such that `self.get(i) == true` and `self.rank(i) == rank`.
    /// This trait uses 0-based indexing, while the [SDSL](https://github.com/simongog/sdsl-lite) select uses 1-based indexing.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled.
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reasons.
    fn select_iter(&'a self, rank: usize) -> Self::OneIter;
}

//-----------------------------------------------------------------------------

/// Select successor queries on the complement of a bitvector.
///
/// Some bitvector types do not build rank/select support structures by default.
/// After the vector has been built, select support for the complement can be enabled with `self.enable_complement()`.
///
/// This trait is analogous to [`Select`].
pub trait SelectZero<'a>: BitVec<'a> {
    /// Iterator type over (index, value) pairs in the complement of the integer array.
    ///
    /// The `Item` in the iterator is an (index, value) pair in the complement of the integer array.
    /// This can be interpreted as `(i, select_zero(i))` or `(rank_zero(j), j)`.
    type ZeroIter: Iterator<Item = (usize, usize)> + ExactSizeIterator;

    /// Returns `true` if select support has been enabled for the complement.
    fn supports_select_zero(&self) -> bool;

    /// Enables select support for the complement vector.
    ///
    /// No effect if select support has already been enabled for the complement.
    fn enable_select_zero(&mut self);

    /// Returns an iterator over the integer array of the complement vector.
    ///
    /// In the bit array interpretation, the iterator will visit all unset bits.
    /// The iterator must not require enabling select support for the complement.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reason.
    fn zero_iter(&'a self) -> Self::ZeroIter;

    /// Returns the specified value in the complement of the integer array or [`None`] if no such value exists.
    ///
    /// In the bit array interpretation, the return value is an index `i` such that `self.get(i) == false` and `self.rank_zero(i) == rank`.
    /// This trait uses 0-based indexing, while the [SDSL](https://github.com/simongog/sdsl-lite) select uses 1-based indexing.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled for the complement.
    /// May panic from I/O errors.
    fn select_zero(&'a self, rank: usize) -> Option<usize>;

    /// Returns an iterator at the specified rank in the complement of the integer array.
    ///
    /// The iterator will return [`None`] if the rank is out of bounds.
    /// In the bit array interpretation, the iterator points to an index `i` such that `self.get(i) == false` and `self.rank_zero(i) == rank`.
    /// This trait uses 0-based indexing, while the [SDSL](https://github.com/simongog/sdsl-lite) select uses 1-based indexing.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled for the complement.
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reasons.
    fn select_zero_iter(&'a self, rank: usize) -> Self::ZeroIter;
}

//-----------------------------------------------------------------------------

/// Predecessor / successor queries on a bitvector.
///
/// Some bitvector types do not build rank/select support structures by default.
/// After the vector has been built, predecessor/successor support can be enabled with `self.enable_pred_succ()`.
/// This may also enable other support structures.
///
/// See [`BitVec`] for an example.
pub trait PredSucc<'a>: BitVec<'a> {
    /// Iterator type over (index, value) pairs in the integer array.
    ///
    /// The `Item` in the iterator is an (index, value) pair in the integer array.
    /// This can be interpreted as `(i, select(i))` or `(rank(j), j)`.
    type OneIter: Iterator<Item = (usize, usize)> + ExactSizeIterator;

    /// Returns `true` if predecessor/successor support has been enabled.
    fn supports_pred_succ(&self) -> bool;

    /// Enables predecessor/successor support for the vector.
    ///
    /// No effect if predecessor/successor support has already been enabled.
    fn enable_pred_succ(&mut self);

    /// Returns an iterator at the largest `v <= value` in the integer array.
    ///
    /// The iterator will return [`None`] if no such value exists.
    /// In the bit array interpretation, the iterator points to the largest `i <= value` such that `self.get(i) == true`.
    ///
    /// # Panics
    ///
    /// May panic if predecessor/successor support has not been enabled.
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reasons.
    fn predecessor(&'a self, value: usize) -> Self::OneIter;

    /// Returns an iterator at the smallest `v >= value` in the integer array.
    ///
    /// The iterator will return [`None`] if no such value exists.
    /// In the bit array interpretation, the iterator points to the smallest `i >= value` such that `self.get(i) == true`.
    ///
    /// # Panics
    ///
    /// May panic if predecessor/successor support has not been enabled.
    /// May panic from I/O errors.
    /// The iterator may also panic for the same reasons.
    fn successor(&'a self, value: usize) -> Self::OneIter;
}

//-----------------------------------------------------------------------------
