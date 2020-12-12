//! Operations common to various bit-packed vectors.

//-----------------------------------------------------------------------------

/// A vector that contains elements of a fixed type.
///
/// # Example
///
/// ```
/// use simple_sds::ops::Element;
/// use simple_sds::bits;
///
/// struct Example(Vec<u8>);
///
/// impl Example {
///     fn new() -> Example {
///         Example(Vec::new())
///     }
/// }
///
/// impl Element for Example {
///     type Item = u8;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         8
///     }
/// } 
///
/// let v = Example::new();
/// assert_eq!(v.len(), 0);
/// assert_eq!(v.width(), bits::bit_len(u8::MAX as u64));
/// ```
pub trait Element {
    /// The type of the elements in the vector.
    type Item;

    /// Returns the number of elements in the vector.
    fn len(&self) -> usize;

    /// Returns the width of of an element in bits.
    fn width(&self) -> usize;
}

/// A vector that can be resized.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{Element, Resize};
///
/// struct Example(Vec<u8>);
///
/// impl Example {
///     fn new() -> Example {
///         Example(Vec::new())
///     }
/// }
///
/// impl Element for Example {
///     type Item = u8;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         8
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
/// let mut v = Example::new();
/// assert_eq!(v.len(), 0);
/// v.reserve(4);
/// assert!(v.capacity() >= 4);
/// v.resize(4, 0);
/// assert_eq!(v.len(), 4);
/// v.clear();
/// assert_eq!(v.len(), 0);
/// ```
pub trait Resize: Element {
    /// Resizes the vector to a specified length.
    ///
    /// If `new_len > self.len()`, the new `new_len - self.len()` values will be initialized.
    /// If `new_len < self.len()`, the vector is truncated.
    ///
    /// Behavior is undefined if the length would exceed the maximum length.
    ///
    /// # Arguments
    ///
    /// * `new_len`: New length of the vector.
    /// * `value`: Initialization value.
    fn resize(&mut self, new_len: usize, value: <Self as Element>::Item);

    /// Clears the vector without freeing the data.
    fn clear(&mut self);

    /// Returns the number of elements that the vector can store without reallocations.
    fn capacity(&self) -> usize;

    /// Reserves space for storing at least `self.len() + additional` elements in the vector.
    ///
    /// Does nothing if the capacity is already sufficient.
    /// Behavior is undefined if the capacity would exceed the maximum length.
    fn reserve(&mut self, additional: usize);
}

/// Conversion into a more space-efficient representation of the same data in a vector of the same type.
///
/// This may, for example, reduce the width of an element.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{Element, Pack};
///
/// #[derive(Clone)]
/// struct Example(Vec<u8>);
///
/// impl Element for Example {
///     type Item = u8;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         8
///     }
/// } 
///
/// // Packing does not make much sense with the running example.
/// impl Pack for Example {
///     fn pack(self) -> Self {
///         self
///     }
/// }
///
/// let v = Example(Vec::from([1, 2, 3]));
/// let w = v.clone().pack();
/// assert_eq!(w.len(), v.len());
/// ```
pub trait Pack: Element {
    /// Converts the vector into a more space-efficient vector of the same type.
    fn pack(self) -> Self;
}

//-----------------------------------------------------------------------------

/// A vector that supports random access to its elements.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{Element, Access};
///
/// struct Example(Vec<u8>);
///
/// impl Example {
///     fn new() -> Example {
///         Example(Vec::new())
///     }
/// }
///
/// impl Element for Example {
///     type Item = u8;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         8
///     }
/// } 
///
/// impl Access for Example {
///     fn get(&self, index: usize) -> Self::Item {
///         self.0[index]
///     }
///
///     fn mutable(&self) -> bool {
///         true
///     }
///
///     fn set(&mut self, index: usize, value: Self::Item) {
///         self.0[index] = value
///     }
/// }
///
/// let mut v = Example(Vec::from([1, 2, 3]));
/// assert!(v.mutable());
/// for i in 0..v.len() {
///     assert_eq!(v.get(i), (i + 1) as u8);
///     v.set(i, i as u8);
///     assert_eq!(v.get(i), i as u8);
/// }
/// ```
pub trait Access: Element {
    /// Gets an element from the vector.
    ///
    /// Behavior is undefined if `index` is not a valid index in the vector.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn get(&self, index: usize) -> <Self as Element>::Item;

    /// Returns `true` if the underlying data is mutable.
    ///
    /// This is relevant, for example, with memory-mapped vectors, where the underlying file may be opened as read-only.
    fn mutable(&self) -> bool;

    /// Sets an element in the vector.
    ///
    /// Behavior is undefined if `index` is not a valid index in the vector or if the underlying data is not mutable.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `value`: New value of the element.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn set(&mut self, index: usize, value: <Self as Element>::Item);
}

//-----------------------------------------------------------------------------

/// Append elements to a vector.
///
/// [`Pop`] is a separate trait, because a file writer may not implement it.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{Element, Push};
///
/// struct Example(Vec<u8>);
///
/// impl Example {
///     fn new() -> Example {
///         Example(Vec::new())
///     }
/// }
///
/// impl Element for Example {
///     type Item = u8;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         8
///     }
/// } 
///
/// impl Push for Example {
///     fn push(&mut self, value: Self::Item) {
///         self.0.push(value);
///     }
/// }
///
/// let mut v = Example::new();
/// assert_eq!(v.len(), 0);
/// v.push(1);
/// v.push(2);
/// v.push(3);
/// assert_eq!(v.len(), 3);
/// ```
pub trait Push: Element {
    /// Appends an element to the vector.
    ///
    /// Behavior is undefined if the vector would exceed the maximum length.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn push(&mut self, value: <Self as Element>::Item);
}

/// Remove and return top elements from a vector.
///
/// [`Push`] is a separate trait, because a file writer may not implement `Pop`.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{Element, Pop};
///
/// struct Example(Vec<u8>);
///
/// impl Example {
///     fn new() -> Example {
///         Example(Vec::new())
///     }
/// }
///
/// impl Element for Example {
///     type Item = u8;
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         8
///     }
/// } 
///
/// impl Pop for Example {
///     fn pop(&mut self) -> Option<Self::Item> {
///         self.0.pop()
///     }
/// }
///
/// let mut v = Example(Vec::from([1, 2, 3]));
/// assert_eq!(v.len(), 3);
/// assert_eq!(v.pop(), Some(3));
/// assert_eq!(v.pop(), Some(2));
/// assert_eq!(v.pop(), Some(1));
/// assert_eq!(v.len(), 0);
/// ```
pub trait Pop: Element {
    /// Removes and returns the last element from the vector.
    /// Returns `None` if there are no more elements in the vector.
    fn pop(&mut self) -> Option<<Self as Element>::Item>;
}

//-----------------------------------------------------------------------------

/// A vector that contains elements with a fixed number of subelements of a fixed type in each element.
///
/// Term *index* refers to the location of an element within a vector, while *offset* refers to the location of a subelement within an element.
/// Every subelement at the same offset has the same width in bits.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{Element, SubElement};
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
/// impl Element for Example {
///     type Item = [u8; 8];
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         64
///     }
/// } 
///
/// impl SubElement for Example {
///     type SubItem = u8;
///
///     fn element_len(&self) -> usize {
///         8
///     }
///
///     fn sub_width(&self, _: usize) -> usize {
///         8
///     }
/// }
///
/// let v = Example::new();
/// assert_eq!(v.len(), 0);
/// assert_eq!(v.element_len(), mem::size_of::<<Example as Element>::Item>());
/// for i in 0..v.len() {
///     assert_eq!(v.sub_width(i), bits::bit_len(u8::MAX as u64));
/// }
/// ```
pub trait SubElement: Element {
    /// The type of the subelements of an element.
    type SubItem;

    /// Returns the number of subelements in an element.
    fn element_len(&self) -> usize;

    /// Returns the width of the specified subelement in bits.
    ///
    /// Behavior is undefined if `offset >= self.element_len()`.
    fn sub_width(&self, offset: usize) -> usize;
}

/// A vector that supports random access to the subelements of its elements.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{Element, SubElement, AccessSub};
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
/// impl Element for Example {
///     type Item = [u8; 8];
///
///     fn len(&self) -> usize {
///         self.0.len()
///     }
///
///     fn width(&self) -> usize {
///         64
///     }
/// } 
///
/// impl SubElement for Example {
///     type SubItem = u8;
///
///     fn element_len(&self) -> usize {
///         8
///     }
///
///     fn sub_width(&self, _: usize) -> usize {
///         8
///     }
/// }
///
/// impl AccessSub for Example {
///     fn get_sub(&self, index: usize, offset: usize) -> Self::SubItem {
///         self.0[index][offset]
///     }
///
///     fn set_sub(&mut self, index: usize, offset: usize, value: Self::SubItem) {
///         self.0[index][offset] = value;
///     }
/// }
///
/// let mut v = Example(Vec::from([
///     [0, 1, 2, 3, 4, 5, 6, 7],
///     [7, 6, 5, 4, 3, 2, 1, 0],
/// ]));
/// assert_eq!(v.get_sub(1, 3), 4u8);
/// v.set_sub(1, 3, 55u8);
/// assert_eq!(v.get_sub(1, 3), 55u8);
/// ```
pub trait AccessSub: SubElement {
    /// Gets a subelement from the vector.
    ///
    /// Behavior is undefined if `index` is not a valid index in the vector or `offset` is not a valid offset in the element.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `offset`: Offset in the element.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn get_sub(&self, index: usize, offset: usize) -> <Self as SubElement>::SubItem;

    /// Sets a subelement in the vector.
    ///
    /// Behavior is undefined if:
    /// * `index` is not a valid index in the vector;
    /// * `offset` is not a valid offset in the element; or
    /// * the underlying data is not mutable.
    ///
    /// # Arguments
    ///
    /// * `index`: Index in the vector.
    /// * `offset`: Offset in the element.
    /// * `value`: New value of the subelement.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn set_sub(&mut self, index: usize, offset: usize, value: <Self as SubElement>::SubItem);
}

//-----------------------------------------------------------------------------

/// An immutable structure that can be seen as a binary array or a sorted array of distinct integers.
///
/// Let `A` be the integer array and `B` the binary array.
/// `B[i] == true` is then equivalent to value `i` being present in array `A`.
/// Indexes in both arrays are 0-based.
///
/// Some operations deal with the complement of the bitvector.
/// In the binary array interpretation, all bits in the complement vector are flipped.
/// In the integer array interpretation, the complement contains the values in `0..self.len()` missing from the original.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{BitVec};
/// use simple_sds::raw_vector::{RawVector, GetRaw, SetRaw};
/// use simple_sds::bits;
///
/// struct NaiveBitVector {
///     ones: usize,
///     data: RawVector,
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
/// impl BitVec for NaiveBitVector {
///     fn len(&self) -> usize {
///         self.data.len()
///     }
///
///     fn count_ones(&self) -> usize {
///         self.ones
///     }
///
///     fn get(&self, index: usize) -> bool {
///         self.data.get_bit(index)
///     }
/// }
///
/// let mut data = RawVector::with_len(137, false);
/// data.set_bit(1, true); data.set_bit(33, true); data.set_bit(95, true); data.set_bit(123, true);
/// let bv = NaiveBitVector::from(data);
/// assert_eq!(bv.len(), 137);
/// assert_eq!(bv.count_ones(), 4);
/// assert!(bv.get(33));
/// assert!(!bv.get(34));
/// ```
pub trait BitVec {
    /// Returns the length of the binary array or the universe size of the integer array.
    fn len(&self) -> usize;

    /// Returns the length of the integer array or the number of ones in the binary array.
    ///
    /// Because the vector is immutable, the implementation should cache the value during construction.
    fn count_ones(&self) -> usize;

    /// Reads a bit from the binary array.
    ///
    /// In the integer array interpretation, returns `true` if value `index` is in the array.
    /// Behavior is undefined if `index` is not a valid index in the binary array.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn get(&self, index: usize) -> bool;
}

/// Rank queries on a bitvector.
///
/// Some bitvector types do not build rank/select support structures by default.
/// After the vector has been built, rank support can be enabled with `self.enable_rank()`.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{BitVec, Rank};
/// use simple_sds::raw_vector::{RawVector, GetRaw, SetRaw};
/// use simple_sds::bits;
/// use std::cmp;
///
/// struct NaiveBitVector {
///     ones: usize,
///     data: RawVector,
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
/// impl BitVec for NaiveBitVector {
///     fn len(&self) -> usize {
///         self.data.len()
///     }
///
///     fn count_ones(&self) -> usize {
///         self.ones
///     }
///
///     fn get(&self, index: usize) -> bool {
///         self.data.get_bit(index)
///     }
/// }
///
/// impl Rank for NaiveBitVector {
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
///             if self.get(i) {
///                 result += 1;
///             }
///         }
///         result
///     }
/// }
///
/// let mut data = RawVector::with_len(137, false);
/// data.set_bit(1, true); data.set_bit(33, true); data.set_bit(95, true); data.set_bit(123, true);
/// let mut bv = NaiveBitVector::from(data);
/// bv.enable_rank();
/// assert!(bv.supports_rank());
/// assert_eq!(bv.rank(33), 1);
/// assert_eq!(bv.rank(34), 2);
/// assert_eq!(bv.complement_rank(65), 63);
/// ```
pub trait Rank: BitVec {
    /// Returns `true` if rank support has been enabled.
    fn supports_rank(&self) -> bool;

    /// Enables rank support for the vector.
    ///
    /// No effect if rank support has already been enabled.
    fn enable_rank(&mut self);

    /// Returns the number of indexes `i < index` in the binary array such that `self.get(i) == true`.
    ///
    /// In the integer array interpretation, returns the number of values smaller than `index`.
    /// The semantics of the query are the same as in [SDSL](https://github.com/simongog/sdsl-lite).
    ///
    /// # Panics
    ///
    /// May panic if rank support has not been enabled.
    /// May panic from I/O errors.
    fn rank(&self, index: usize) -> usize;

    /// Returns the number of indexes `i < index` in the binary array such that `self.get(i) == false`.
    ///
    /// In the integer array interpretation, returns the number of missing values smaller than `index`.
    /// The semantics of the query are the same as in [SDSL](https://github.com/simongog/sdsl-lite).
    ///
    /// # Panics
    ///
    /// May panic if rank support has not been enabled.
    /// May panic from I/O errors.
    fn complement_rank(&self, index: usize) -> usize {
        index - self.rank(index)
    }
}

/// Select / predecessor / successor queries on a bitvector.
///
/// Some bitvector types do not build rank/select support structures by default.
/// After the vector has been built, select support can be enabled with `self.enable_select()`.
///
/// # Example
///
/// ```
/// use simple_sds::ops::{BitVec, Select};
/// use simple_sds::raw_vector::{RawVector, GetRaw, SetRaw};
/// use simple_sds::bits;
/// use std::cmp;
///
/// struct NaiveBitVector {
///     ones: usize,
///     data: RawVector,
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
/// impl BitVec for NaiveBitVector {
///     fn len(&self) -> usize {
///         self.data.len()
///     }
///
///     fn count_ones(&self) -> usize {
///         self.ones
///     }
///
///     fn get(&self, index: usize) -> bool {
///         self.data.get_bit(index)
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
///     type Iter = SelectIter<'a>;
///
///     fn supports_select(&self) -> bool {
///         true
///     }
///
///     fn enable_select(&mut self) {}
///
///     fn select(&'a self, index: usize) -> Self::Iter {
///         let mut rank: usize = 0;
///         let mut bv_index: usize = 0;
///         while bv_index < self.len() {
///             if self.get(bv_index) {
///                 if rank == index {
///                     break;
///                 }
///                 rank += 1;
///             }
///             bv_index += 1;
///         }
///         Self::Iter {
///             parent: self,
///             rank: rank,
///             index: bv_index,
///         }
///     }
///
///     fn predecessor(&'a self, index: usize) -> Self::Iter {
///         let mut result = Self::Iter {
///             parent: self,
///             rank: self.count_ones(),
///             index: self.len(),
///         };
///         let mut rank: usize = 0;
///         let mut bv_index: usize = 0;
///         while bv_index <= index && bv_index < self.len() {
///             if self.get(bv_index) {
///                 result.rank = rank;
///                 result.index = bv_index;
///                 rank += 1;
///             }
///             bv_index += 1;
///         }
///         result
///     }
///
///     fn successor(&'a self, index: usize) -> Self::Iter {
///         let mut result = Self::Iter {
///             parent: self,
///             rank: self.count_ones(),
///             index: self.len(),
///         };
///         let mut rank: usize = 0;
///         let mut bv_index: usize = 0;
///         while bv_index < index && bv_index < self.len() {
///             rank += self.get(bv_index) as usize;
///             bv_index += 1;
///         }
///         while bv_index < self.len() {
///             if self.get(bv_index) {
///                 result.rank = rank;
///                 result.index = bv_index;
///                 return result;
///             }
///             bv_index += 1;
///         }
///         result
///     }
/// }
///
/// let mut data = RawVector::with_len(137, false);
/// data.set_bit(1, true); data.set_bit(33, true); data.set_bit(95, true); data.set_bit(123, true);
/// let mut bv = NaiveBitVector::from(data);
/// bv.enable_select();
/// assert!(bv.supports_select());
/// assert_eq!(bv.select(2).next(), Some((2, 95)));
/// assert_eq!(bv.predecessor(0).next(), None);
/// assert_eq!(bv.predecessor(1).next(), Some((0, 1)));
/// assert_eq!(bv.predecessor(2).next(), Some((0, 1)));
/// assert_eq!(bv.successor(122).next(), Some((3, 123)));
/// assert_eq!(bv.successor(123).next(), Some((3, 123)));
/// assert_eq!(bv.successor(124).next(), None);
/// ```
pub trait Select<'a>: BitVec {
    /// Iterator type over (index, value) pairs in the integer array.
    ///
    /// The `Item` in the iterator is an (index, value) pair in the integer array.
    /// This can be interpreted as `(i, select(i))` or `(rank(j), j)`.
    type Iter: Iterator<Item = (usize, usize)>;

    /// Returns `true` if select support has been enabled.
    fn supports_select(&self) -> bool;

    /// Enables select support for the vector.
    ///
    /// No effect if select support has already been enabled.
    fn enable_select(&mut self);

    /// Returns an iterator at the specified index in the integer array.
    ///
    /// The iterator will return `None` if the index is out of bounds.
    /// In the bit array interpretation, the iterator points to an index `i` such that `self.get(i) == true` and `self.rank(i) == index`.
    /// This trait uses 0-based indexing, while the [SDSL](https://github.com/simongog/sdsl-lite) select uses 1-based indexing.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled.
    /// May panic from I/O errors.
    fn select(&'a self, index: usize) -> Self::Iter;

    /// Returns an iterator at the largest `v <= value` in the integer array.
    ///
    /// The iterator will return `None` if no such value exists.
    /// In the bit array interpretation, the iterator points to the largest `i <= value` such that `self.get(i) == true`.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled.
    /// May panic from I/O errors.
    fn predecessor(&'a self, value: usize) -> Self::Iter;

    /// Returns an iterator at the smallest `v >= value` in the integer array.
    ///
    /// The iterator will return `None` if no such value exists.
    /// In the bit array interpretation, the iterator points to the smallest `i >= value` such that `self.get(i) == true`.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled.
    /// May panic from I/O errors.
    fn successor(&'a self, value: usize) -> Self::Iter;
}


/// Select / predecessor / successor queries on the complement of a bitvector.
///
/// Some bitvector types do not build rank/select support structures by default.
/// After the vector has been built, select support for the complement can be enabled with `self.enable_complement()`.
///
/// This trait is analogous to [`Select`].
/// See that trait for an example.
pub trait Complement<'a>: BitVec {
    /// Iterator type over (index, value) pairs in the complement of the integer array.
    ///
    /// The `Item` in the iterator is an (index, value) pair in the complement of the integer array.
    /// This can be interpreted as `(i, complement_select(i))` or `(complement_rank(j), j)`.
    type Iter: Iterator<Item = (usize, usize)>;

    /// Returns `true` if select support has been enabled for the complement.
    fn supports_complement(&self) -> bool;

    /// Enables select support for the complement vector.
    ///
    /// No effect if select support has already been enabled for the complement.
    fn enable_complement(&mut self);

    /// Returns an iterator at the specified index in the complement of the integer array.
    ///
    /// The iterator will return `None` if the index is out of bounds.
    /// In the bit array interpretation, the iterator points to an index `i` such that `self.get(i) == false` and `self.complement_rank(i) == index`.
    /// This trait uses 0-based indexing, while the [SDSL](https://github.com/simongog/sdsl-lite) select uses 1-based indexing.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled for the complement.
    /// May panic from I/O errors.
    fn complement_select(&'a self, index: usize) -> Self::Iter;

    /// Returns an iterator at the largest `v <= value` missing from the integer array.
    ///
    /// The iterator will return `None` if no such value exists.
    /// In the bit array interpretation, the iterator points to the largest `i <= value` such that `self.get(i) == false`.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled for the complement.
    /// May panic from I/O errors.
    fn complement_predecessor(&'a self, value: usize) -> Self::Iter;

    /// Returns an iterator at the smallest `v >= value` missing from the integer array.
    ///
    /// The iterator will return `None` if no such value exists.
    /// In the bit array interpretation, the iterator points to the smallest `i >= value` such that `self.get(i) == false`.
    ///
    /// # Panics
    ///
    /// May panic if select support has not been enabled for the complement.
    /// May panic from I/O errors.
    fn complement_successor(&'a self, value: usize) -> Self::Iter;
}

//-----------------------------------------------------------------------------
