//! A bit-packed integer vector storing fixed-width integers.

//use crate::ops::{Element, Resize, Pack, Push, Pop, Access};
use crate::ops::{Element, Resize, Push, Pop, Access};
use crate::raw_vector::{RawVector, PushRaw, PopRaw, SetRaw, GetRaw};
//use crate::serialize::Serialize;
use crate::bits;

//-----------------------------------------------------------------------------

/// A contiguous growable bit-packed array of fixed-width integers.
///
/// This is a wrapper over [`RawVector`], which is in turn a wrapper over [`Vec`].
/// Each element consists of the lowest 1 to 64 bits of an `u64` value, as specified by parameter `width`.
/// The *maximum length* of the vector is `usize::MAX` bits or `usize::MAX / width` elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntVector {
    len: usize,
    width: usize,
    data: RawVector,
}

impl IntVector {
    /// Creates an empty vector with specified width.
    ///
    /// If `width < 1` or `width > 64`, it is silently adjusted to the closest value in the accepted range.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::int_vector::IntVector;
    /// use simple_sds::ops::Element;
    ///
    /// let v = IntVector::new(13);
    /// assert_eq!(v.len(), 0);
    /// assert_eq!(v.width(), 13);
    /// ```
    pub fn new(width: usize) -> IntVector {
        IntVector {
            len: 0,
            width: Self::adjust_width(width),
            data: RawVector::new(),
        }
    }

    /// Creates an initialized vector of specified length and width.
    ///
    /// Behavior is undefined if `len * width > usize::MAX`.
    /// If `width < 1` or `width > 64`, it is silently adjusted to the closest value in the accepted range.
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
    /// let v = IntVector::with_len(4, 13, 1234);
    /// assert_eq!(v.len(), 4);
    /// for i in 0..v.len() {
    ///     assert_eq!(v.get(i), 1234);
    /// }
    /// ```
    pub fn with_len(len: usize, width: usize, value: <Self as Element>::Item) -> IntVector {
        let width = Self::adjust_width(width);
        let mut data = RawVector::with_capacity(len * width);
        for _ in 0..len {
            data.push_int(value, width);
        }
        IntVector {
            len: len,
            width: width,
            data: data,
        }
    }

    /// Creates an empty vector with enough capacity for at least the specified number of elements of specified width.
    ///
    /// Behavior is undefined if `capacity * width > usize::MAX`.
    /// If `width < 1` or `width > 64`, it is silently adjusted to the closest value in the accepted range.
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
    /// let v = IntVector::with_capacity(4, 13);
    /// assert_eq!(v.len(), 0);
    /// assert_eq!(v.width(), 13);
    /// assert!(v.capacity() >= 4);
    /// ```
    pub fn with_capacity(capacity: usize, width: usize) -> IntVector {
        let width = Self::adjust_width(width);
        IntVector {
            len: 0,
            width: width,
            data: RawVector::with_capacity(capacity * width),
        }
    }

    fn adjust_width(width: usize) -> usize {
        match width {
            0 => 1,
            1..=bits::WORD_BITS => width,
            _ => bits::WORD_BITS
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

// FIXME Pack

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

// FIXME Serialize

//-----------------------------------------------------------------------------

// FIXME FromIterator for all unsigned integer types

// FIXME iterators

//-----------------------------------------------------------------------------

// FIXME tests: construction, element, stack, random access, serialize, iterators

//-----------------------------------------------------------------------------
