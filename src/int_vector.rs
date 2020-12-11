//! A bit-packed integer vector storing fixed-width integers.

use crate::ops::{Element, Resize, Pack, Access, Push, Pop};
use crate::raw_vector::{RawVector, SetRaw, GetRaw, PushRaw, PopRaw};
use crate::serialize::Serialize;
use crate::bits;

use std::iter::FromIterator;
use std::io;

//-----------------------------------------------------------------------------

/// A contiguous growable bit-packed array of fixed-width integers.
///
/// This structure contains [`RawVector`], which is in turn contains [`Vec`].
/// Each element consists of the lowest 1 to 64 bits of an `u64` value, as specified by parameter `width`.
/// The maximum length of the vector is `usize::MAX / width` elements.
///
/// A default constructed `IntVector` has `width == 64`.
/// `IntVector` can be built from an iterator over `u8`, `u16`, `u32`, `u64`, or `usize` values.
///
/// `IntVector` implements the following `simple_sds` traits:
/// * Basic functionality: [`Element`], [`Resize`], [`Pack`]
/// * Queries and operations: [`Access`], [`Push`], [`Pop`]
/// * Serialization: [`Serialize`]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntVector {
    len: usize,
    width: usize,
    data: RawVector,
}

impl IntVector {
    /// Creates an empty vector with specified width.
    /// 
    /// Returns `None` if the width is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::int_vector::IntVector;
    /// use simple_sds::ops::Element;
    ///
    /// let v = IntVector::new(13).unwrap();
    /// assert_eq!(v.len(), 0);
    /// assert_eq!(v.width(), 13);
    /// ```
    pub fn new(width: usize) -> Option<IntVector> {
        if width == 0 || width > bits::WORD_BITS {
            return None;
        }
        Some(IntVector {
            len: 0,
            width: width,
            data: RawVector::new(),
        })
    }

    /// Creates an initialized vector of specified length and width.
    /// 
    /// Returns `None` if the width is invalid.
    /// Behavior is undefined if `len * width > usize::MAX`.
    ///
    /// # Arguments
    ///
    /// * `len`: Number of elements in the vector.
    /// * `width`: Width of each element in bits.
    /// * `value`: Initialization value.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::int_vector::IntVector;
    /// use simple_sds::ops::{Element, Access};
    ///
    /// let v = IntVector::with_len(4, 13, 1234).unwrap();
    /// assert_eq!(v.len(), 4);
    /// assert_eq!(v.width(), 13);
    /// for i in 0..v.len() {
    ///     assert_eq!(v.get(i), 1234);
    /// }
    /// ```
    pub fn with_len(len: usize, width: usize, value: u64) -> Option<IntVector> {
        if width == 0 || width > bits::WORD_BITS {
            return None;
        }
        let mut data = RawVector::with_capacity(len * width);
        for _ in 0..len {
            data.push_int(value, width);
        }
        Some(IntVector {
            len: len,
            width: width,
            data: data,
        })
    }

    /// Creates an empty vector with enough capacity for at least the specified number of elements of specified width.
    ///
    /// Returns `None` if the width is invalid.
    /// Behavior is undefined if `capacity * width > usize::MAX`.
    ///
    /// # Arguments
    ///
    /// * `capacity`: Minimun capacity of the vector in elements.
    /// * `width`: Width of each element in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::int_vector::IntVector;
    /// use simple_sds::ops::{Element, Resize};
    ///
    /// let v = IntVector::with_capacity(4, 13).unwrap();
    /// assert_eq!(v.len(), 0);
    /// assert_eq!(v.width(), 13);
    /// assert!(v.capacity() >= 4);
    /// ```
    pub fn with_capacity(capacity: usize, width: usize) -> Option<IntVector> {
        if width == 0 || width > bits::WORD_BITS {
            None
        } else {
            Some(IntVector {
                len: 0,
                width: width,
                data: RawVector::with_capacity(capacity * width),
            })
        }
    }

    /// Returns an iterator visiting all elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::int_vector::IntVector;
    ///
    /// let source: Vec<u16> = vec![123, 456, 789, 10];
    /// let mut v: IntVector = source.iter().cloned().collect();
    /// for (index, value) in v.iter().enumerate() {
    ///     assert_eq!(source[index] as u64, value);
    /// }
    /// ```
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            parent: self,
            index: 0,
        }
    }
}

//-----------------------------------------------------------------------------

impl Element for IntVector {
    type Item = u64;

    fn len(&self) -> usize {
        self.len
    }

    fn width(&self) -> usize {
        self.width
    }
}

impl Resize for IntVector {
    fn resize(&mut self, new_len: usize, value: <Self as Element>::Item) {
        if new_len > self.len() {
            self.reserve(new_len - self.len());
            while self.len() < new_len {
                self.push(value);
            }
        } else if new_len < self.len() {
            self.data.resize(new_len * self.width(), false);
            self.len = new_len;
        }
    }

    fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    fn capacity(&self) -> usize {
        self.data.capacity() / self.width()
    }

    fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.width());
    }
}

impl Pack for IntVector {
    fn pack(self) -> Self {
        if self.len() == 0 {
            return self;
        }
        let new_width = bits::bit_len(self.iter().max().unwrap());
        if new_width == self.width() {
            return self;
        }
        let mut result = IntVector::with_capacity(self.len(), new_width).unwrap();
        for value in self.iter() {
            result.push(value);
        }
        result
    }
}

impl Access for IntVector {
    fn get(&self, index: usize) -> <Self as Element>::Item {
        self.data.get_int(index * self.width(), self.width)
    }

    fn mutable(&self) -> bool {
        true
    }

    fn set(&mut self, index: usize, value: <Self as Element>::Item) {
        self.data.set_int(index * self.width(), value, self.width());
    }
}

impl Push for IntVector {
    fn push(&mut self, value: <Self as Element>::Item) {
        self.data.push_int(value, self.width());
        self.len += 1;
    }    
}

impl Pop for IntVector {
    fn pop(&mut self) -> Option<<Self as Element>::Item> {
        if self.len() > 0 {
            self.len -= 1;
        }
        self.data.pop_int(self.width())
    }    
}

impl Serialize for IntVector {
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.len.serialize(writer)?;
        self.width.serialize(writer)?;
        self.data.serialize_header(writer)?;
        Ok(())
    }

    fn serialize_data<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.data.serialize_data(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let len = usize::load(reader)?;
        let width = usize::load(reader)?;
        let data = RawVector::load(reader)?;
        Ok(IntVector {
            len: len,
            width: width,
            data: data,
        })
    }

    fn size_in_bytes(&self) -> usize {
        self.len.size_in_bytes() + self.width.size_in_bytes() + self.data.size_in_bytes()
    }
}

//-----------------------------------------------------------------------------

impl Default for IntVector {
    fn default() -> Self {
        IntVector {
            len: 0,
            width: bits::WORD_BITS,
            data: RawVector::new(),
        }
    }
}

//-----------------------------------------------------------------------------

macro_rules! iter_to_int_vector {
    ($t:ident, $w:expr) => {
        impl FromIterator<$t> for IntVector {
            fn from_iter<I: IntoIterator<Item=$t>>(iter: I) -> Self {
                let mut result = IntVector::new($w).unwrap();
                for value in iter {
                    result.push(value as <Self as Element>::Item);
                }
                result
            }
        }
    }
}

iter_to_int_vector!(u8, 8);
iter_to_int_vector!(u16, 16);
iter_to_int_vector!(u32, 32);
iter_to_int_vector!(u64, 64);
iter_to_int_vector!(usize, 64);

//-----------------------------------------------------------------------------

/// A read-only iterator over [`IntVector`].
///
/// The type of `Item` is `u64`.
///
/// # Examples
///
/// ```
/// use simple_sds::int_vector::IntVector;
///
/// let source: Vec<u64> = vec![123, 456, 789, 10];
/// let mut v: IntVector = source.iter().cloned().collect();
/// for (index, value) in v.iter().enumerate() {
///     assert_eq!(source[index], value);
/// }
/// ```
pub struct Iter<'a> {
    parent: &'a IntVector,
    index: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = <IntVector as Element>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.parent.len() {
            None
        } else {
            let result = Some(self.parent.get(self.index));
            self.index += 1;
            result
        }
    }
}

/// [`IntVector`] transformed into an iterator.
///
/// The type of `Item` is `u64`.
///
/// # Example
///
/// ```
/// use simple_sds::int_vector::IntVector;
///
/// let source: Vec<u64> = vec![1, 3, 15, 255, 65535];
/// let mut v: IntVector = source.iter().cloned().collect();
/// let target: Vec<u64> = v.into_iter().collect();
/// assert_eq!(target, source);
/// ```
pub struct IntoIter {
    parent: IntVector,
    index: usize,
}

impl Iterator for IntoIter {
    type Item = <IntVector as Element>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.parent.len() {
            None
        } else {
            let result = Some(self.parent.get(self.index));
            self.index += 1;
            result
        }
    }
}

impl IntoIterator for IntVector {
    type Item = <Self as Element>::Item;
    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            parent: self,
            index: 0,
        }
    }
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Element, Resize, Pack, Access, Push, Pop};
    use crate::serialize::Serialize;
    use crate::serialize;
    use std::fs;

    #[test]
    fn empty_int_vector() {
        let empty = IntVector::default();
        assert_eq!(empty.len(), 0, "Created a non-empty empty vector");
        assert_eq!(empty.width(), 64, "Invalid width for an empty vector");
        assert_eq!(empty.capacity(), 0, "Reserved unnecessary memory for an empty vector");

        let with_width = IntVector::new(13).unwrap();
        assert_eq!(with_width.len(), 0, "Created a non-empty empty vector with a specified width");
        assert_eq!(with_width.width(), 13, "Invalid width for an empty vector with a specified width");
        assert_eq!(with_width.capacity(), 0, "Reserved unnecessary memory for an empty vector with a specified width");

        let with_capacity = IntVector::with_capacity(137, 13).unwrap();
        assert_eq!(with_capacity.len(), 0, "Created a non-empty vector by specifying capacity");
        assert_eq!(with_width.width(), 13, "Invalid width for an empty vector with a specified capacity");
        assert!(with_capacity.capacity() >= 137, "Vector capacity is lower than specified");
    }

    #[test]
    fn with_len_and_clear() {
        let mut v = IntVector::with_len(137, 13, 123).unwrap();
        assert_eq!(v.len(), 137, "Vector length is not as specified");
        assert_eq!(v.width(), 13, "Vector width is not as specified");
        v.clear();
        assert_eq!(v.len(), 0, "Could not clear the vector");
        assert_eq!(v.width(), 13, "Clearing the vector changed its width");
    }

    #[test]
    fn initialization_vs_push() {
        let with_len = IntVector::with_len(137, 13, 123).unwrap();
        let mut pushed = IntVector::new(13).unwrap();
        for _ in 0..137 {
            pushed.push(123);
        }
        assert_eq!(with_len, pushed, "Initializing with and pushing values yield different vectors");
    }

    #[test]
    fn initialization_vs_resize() {
        let initialized = IntVector::with_len(137, 13, 123).unwrap();

        let mut extended = IntVector::with_len(66, 13, 123).unwrap();
        extended.resize(137, 123);
        assert_eq!(extended, initialized, "Extended vector is invalid");

        let mut truncated = IntVector::with_len(212, 13, 123).unwrap();
        truncated.resize(137, 123);
        assert_eq!(truncated, initialized, "Truncated vector is invalid");

        let mut popped = IntVector::with_len(97, 13, 123).unwrap();
        for _ in 0..82 {
            popped.pop();
        }
        popped.resize(137, 123);
        assert_eq!(popped, initialized, "Popped vector is invalid after extension");
    }

    #[test]
    fn reserving_capacity() {
        let mut original = IntVector::with_len(137, 13, 123).unwrap();
        let copy = original.clone();
        original.reserve(31 + original.capacity() - original.len());

        assert!(original.capacity() >= 137 + 31, "Reserving additional capacity failed");
        assert_eq!(original, copy, "Reserving additional capacity changed the vector");
    }

    #[test]
    fn push_pop() {
        let mut correct: Vec<u16> = Vec::new();
        for i in 0..64 {
            correct.push(i); correct.push(i * (i + 1));
        }

        let mut v = IntVector::new(16).unwrap();
        for value in correct.iter() {
            v.push(*value as u64);
        }
        assert_eq!(v.len(), 128, "Invalid vector length");

        let from_iter: IntVector = correct.iter().cloned().collect();
        assert_eq!(from_iter, v, "Vector built from an iterator is invalid");

        correct.reverse();
        let mut popped: Vec<u16> = Vec::new();
        for _ in 0..correct.len() {
            if let Some(value) = v.pop() {
                popped.push(value as u16);
            }
        }
        assert_eq!(popped.len(), correct.len(), "Invalid number of popped ints");
        assert_eq!(v.len(), 0, "Non-empty vector after popping all ints");
        assert_eq!(popped, correct, "Invalid popped ints");
    }

    #[test]
    fn set_get() {
        let mut v = IntVector::with_len(128, 13, 0).unwrap();
        for i in 0..64 {
            v.set(2 * i, i as u64); v.set(2 * i + 1, (i * (i + 1)) as u64);
        }
        for i in 0..64 {
            assert_eq!(v.get(2 * i), i as u64, "Invalid integer [{}].0", i);
            assert_eq!(v.get(2 * i + 1), (i * (i + 1)) as u64, "Invalid integer [{}].1", i);
        }
    }

    #[test]
    fn iterators_and_pack() {
        let correct: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];

        let packed: IntVector = correct.iter().cloned().collect();
        let packed = packed.pack();
        assert_eq!(packed.width(), 7, "Incorrect width after pack()");
        for (index, value) in packed.iter().enumerate() {
            assert_eq!(value, correct[index], "Invalid value in the packed vector");
        }

        let from_iter: Vec<u64> = packed.into_iter().collect();
        assert_eq!(from_iter, correct, "Invalid vector built from into_iter()");
    }

    #[test]
    fn serialize_int_vector() {
        let mut original = IntVector::new(16).unwrap();
        for i in 0..64 {
            original.push(i * (i + 1) * (i + 2));
        }
        assert_eq!(original.size_in_bytes(), 160, "Invalid IntVector size in bytes");

        let filename = serialize::temp_file_name("int-vector");
        serialize::serialize_to(&original, &filename).unwrap();

        let copy: IntVector = serialize::load_from(&filename).unwrap();
        assert_eq!(copy, original, "Serialization changed the IntVector");

        fs::remove_file(&filename).unwrap();
    }
}

//-----------------------------------------------------------------------------
