//! Operations common to various vectors.
//!
//! # Integer vectors
//!
//! * [`Vector`]: Basic operations.
//! * [`Resize`]: Resizable vectors.
//! * [`Pack`]: Space-efficiency by e.g. bit packing.
//! * [`Access`]: Random access.
//! * [`Push`], [`Pop`]: Stack operations.
//! * [`SubItem`]: The items are tuples of fixed type.
//! * [`AccessSub`]: Access to individual fields in the tuples.
//!
//! # Bitvectors
//!
//! * [`BitVec`]: Random access and iterators over all bits.
//! * [`Rank`]: Rank queries.
//! * [`Select`]: Select queries and iterators over set bits.
//! * [`SelectZero`]: Select queries on the complement and iterators over unset bits.
//! * [`PredSucc`]: Predecessor and successor queries.

//-----------------------------------------------------------------------------

/// A vector that contains items of a fixed type.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{Vector, Resize, Pack, Access};
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
/// impl Access for Example {
///     fn get(&self, index: usize) -> Self::Item {
///         self.0[index]
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
/// let mut v = Example(Vec::from([1, 2, 3]));
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
/// ```
pub trait Vector {
    /// The type of the items in the vector.
    type Item;

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
/// See [`Vector`] for an example.
pub trait Access: Vector {
    /// Gets an item from the vector.
    ///
    /// # Panics
    ///
    /// May panic if `index` is not a valid index in the vector.
    /// May panic from I/O errors.
    fn get(&self, index: usize) -> <Self as Vector>::Item;

    /// Returns `true` if the underlying data is mutable.
    ///
    /// This is relevant, for example, with memory-mapped vectors, where the underlying file may be opened as read-only.
    fn is_mutable(&self) -> bool;

    /// Sets an item in the vector.
    ///
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
    fn set(&mut self, index: usize, value: <Self as Vector>::Item);
}

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

/// A vector that contains items with a fixed number of subitems of a fixed type in each item.
///
/// Term *index* refers to the location of an item within a vector, while *offset* refers to the location of a subitem within an item.
/// Every subitem at the same offset has the same width in bits.
///
/// # Examples
///
/// ```
/// use simple_sds::ops::{Vector, SubItem, AccessSub};
/// use simple_sds::bits;
/// use std::mem;
///
/// struct Example(Vec<[u8; 8]>);
///
/// impl Example {
///     fn new() -> Example {
///         Example(Vec::new())
///     }
/// }
///
/// impl Vector for Example {
///     type Item = [u8; 8];
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         64
///     }
///
///     fn max_len(&self) -> usize {
///         usize::MAX
///     }
/// } 
///
/// impl SubItem for Example {
///     type SubItem = u8;
///
///     fn item_len(&self) -> usize {
///         8
///     }
///
///     fn sub_width(&self, _: usize) -> usize {
///         8
///     }
/// }
///
/// impl AccessSub for Example {
///     fn sub(&self, index: usize, offset: usize) -> Self::SubItem {
///         self.0[index][offset]
///     }
///
///     fn set_sub(&mut self, index: usize, offset: usize, value: Self::SubItem) {
///         self.0[index][offset] = value;
///     }
/// }
///
/// // SubItem
/// let v = Example::new();
/// assert!(v.is_empty());
/// assert_eq!(v.item_len(), mem::size_of::<<Example as Vector>::Item>());
/// for i in 0..v.len() {
///     assert_eq!(v.sub_width(i), bits::bit_len(u8::MAX as u64));
/// }
///
/// // AccessSub
/// let mut v = Example(Vec::from([
///     [0, 1, 2, 3, 4, 5, 6, 7],
///     [7, 6, 5, 4, 3, 2, 1, 0],
/// ]));
/// assert_eq!(v.sub(1, 3), 4u8);
/// v.set_sub(1, 3, 55u8);
/// assert_eq!(v.sub(1, 3), 55u8);
/// ```
pub trait SubItem: Vector {
    /// The type of the subitems of an item.
    type SubItem;

    /// Returns the number of subitems in an item.
    fn item_len(&self) -> usize;

    /// Returns the width of the specified subitem in bits.
    ///
    /// # Panics
    ///
    /// May panic if `offset >= self.item_len()`.
    fn sub_width(&self, offset: usize) -> usize;
}

/// A vector that supports random access to the subitems of its items.
///
/// See [`SubItem`] for an example.
pub trait AccessSub: SubItem {
    /// Gets a subitem from the vector.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `offset`: Offset in the item.
    ///
    /// # Panics
    ///
    /// May panic if `index` is not a valid index in the vector or `offset` is not a valid offset in the item.
    /// May panic from I/O errors.
    fn sub(&self, index: usize, offset: usize) -> <Self as SubItem>::SubItem;

    /// Sets a subitem in the vector.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `offset`: Offset in the item.
    /// * `value`: New value of the subitem.
    ///
    /// # Panics
    ///
    /// May panic if `index` is not a valid index in the vector or `offset` is not a valid offset in the item.
    /// May panic if the underlying data is not mutable.
    /// May panic from I/O errors.
    fn set_sub(&mut self, index: usize, offset: usize, value: <Self as SubItem>::SubItem);
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
/// }
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
/// }
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
///         let mut found: usize = 0;
///         let mut index: usize = 0;
///         while index < self.len() {
///             if self.get(index) {
///                 if found == rank {
///                     break;
///                 }
///                 found += 1;
///             }
///             index += 1;
///         }
///         Self::OneIter {
///             parent: self,
///             rank: rank,
///             index: index,
///         }
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
///         let mut result = Self::OneIter {
///             parent: self,
///             rank: self.count_ones(),
///             index: self.len(),
///         };
///         let mut rank: usize = 0;
///         let mut index: usize = 0;
///         while index <= value && index < self.len() {
///             if self.get(index) {
///                 result.rank = rank;
///                 result.index = index;
///                 rank += 1;
///             }
///             index += 1;
///         }
///         result
///     }
///
///     fn successor(&'a self, value: usize) -> Self::OneIter {
///         let mut result = Self::OneIter {
///             parent: self,
///             rank: self.count_ones(),
///             index: self.len(),
///         };
///         let mut rank: usize = 0;
///         let mut index: usize = 0;
///         while index < value && index < self.len() {
///             rank += self.get(index) as usize;
///             index += 1;
///         }
///         while index < self.len() {
///             if self.get(index) {
///                 result.rank = rank;
///                 result.index = index;
///                 return result;
///             }
///             index += 1;
///         }
///         result
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
    type Iter: Iterator<Item = bool>;

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
    /// The iterator may also panic for the same reasons.
    fn iter(&'a self) -> Self::Iter;
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
    type OneIter: Iterator<Item = (usize, usize)>;

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
    type ZeroIter: Iterator<Item = (usize, usize)>;

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
    type OneIter: Iterator<Item = (usize, usize)>;

    /// Returns `true` if predecessor/successor support has been enabled.
    fn supports_pred_succ(&self) -> bool;

    /// Enables predecessor/successor support for the vector.
    ///
    /// No effect if predecessor/successor support has already been enabled.
    fn enable_pred_succ(&mut self);

    /// Returns an iterator at the largest `v <= value` in the integer array.
    ///
    /// The iterator will return `None` if no such value exists.
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
