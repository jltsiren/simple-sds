//! A bit-packed integer vector storing fixed-width integers.

use crate::ops::{Element, Resize, Pack, Access, Push, Pop};
use crate::raw_vector::{RawVector, RawVectorWriter, AccessRaw, PushRaw, PopRaw};
use crate::serialize::{Serialize, Writer, FlushMode};
use crate::bits;

use std::fs::File;
use std::io::{Error, ErrorKind};
use std::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, FromIterator, Extend};
use std::path::Path;
use std::io;

//-----------------------------------------------------------------------------

/// A contiguous growable bit-packed array of fixed-width integers.
///
/// This structure contains [`RawVector`], which is in turn contains [`Vec`].
/// Each element consists of the lowest 1 to 64 bits of a `u64` value, as specified by parameter `width`.
/// The maximum length of the vector is `usize::MAX / width` elements.
///
/// A default constructed `IntVector` has `width == 64`.
/// `IntVector` can be built from an iterator over `u8`, `u16`, `u32`, `u64`, or `usize` values.
///
/// `IntVector` implements the following `simple_sds` traits:
/// * Basic functionality: [`Element`], [`Resize`], [`Pack`]
/// * Queries and operations: [`Access`], [`Push`], [`Pop`]
/// * Serialization: [`Serialize`]
///
/// # Notes
///
/// * `IntVector` never panics from I/O errors.
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
    /// assert!(v.is_empty());
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
    /// assert!(v.is_empty());
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
            next: 0,
            limit: self.len(),
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
        if self.is_empty() {
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
        self.data.int(index * self.width(), self.width)
    }

    fn is_mutable(&self) -> bool {
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

    fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.data.serialize_body(writer)?;
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

impl AsRef<RawVector> for IntVector {
    fn as_ref(&self) -> &RawVector {
        &(self.data)
    }
}

impl From<IntVector> for RawVector {
    fn from(source: IntVector) -> Self {
        source.data
    }
}

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
/// let v: IntVector = source.iter().cloned().collect();
/// for (index, value) in v.iter().enumerate() {
///     assert_eq!(source[index], value);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Iter<'a> {
    parent: &'a IntVector,
    // The first index we have not used.
    next: usize,
    // The first index we should not use.
    limit: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = <IntVector as Element>::Item;

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

/// [`IntVector`] transformed into an iterator.
///
/// The type of `Item` is `u64`.
///
/// # Examples
///
/// ```
/// use simple_sds::int_vector::IntVector;
///
/// let source: Vec<u64> = vec![1, 3, 15, 255, 65535];
/// let v: IntVector = source.iter().cloned().collect();
/// let target: Vec<u64> = v.into_iter().collect();
/// assert_eq!(target, source);
/// ```
#[derive(Clone, Debug)]
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.parent.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for IntoIter {}

impl FusedIterator for IntoIter {}

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

/// A buffered file writer compatible with the serialization format of [`IntVector`].
///
/// When the writer goes out of scope, the internal buffer is flushed, the file is closed, and all errors are ignored.
/// Call [`IntVectorWriter::close`] explicitly to handle the errors.
#[derive(Debug)]
pub struct IntVectorWriter {
    len: usize,
    width: usize,
    writer: RawVectorWriter,
}

impl IntVectorWriter {
    /// Creates an empty vector stored in the specified file with the default buffer size.
    ///
    /// If the file already exists, it will be overwritten.
    ///
    /// # Arguments
    ///
    /// * `filename`: Name of the file.
    /// * `width`: Width of each element in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::int_vector::IntVectorWriter;
    /// use simple_sds::ops::Element;
    /// use simple_sds::serialize;
    /// use std::{fs, mem};
    ///
    /// let filename = serialize::temp_file_name("int-vector-writer-new");
    /// let mut v = IntVectorWriter::new(&filename, 13).unwrap();
    /// assert!(v.is_empty());
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error of the kind [`ErrorKind::Other`] if the width is invalid.
    /// Any I/O errors will be passed through.
    pub fn new<P: AsRef<Path>>(filename: P, width: usize) -> io::Result<IntVectorWriter> {
        if width == 0 || width > bits::WORD_BITS {
            return Err(Error::new(ErrorKind::Other, format!("invalid element width: {}", width)));
        }
        let writer = RawVectorWriter::new(filename)?;
        let mut result = IntVectorWriter {
            len: 0,
            width: width,
            writer: writer,
        };
        // TODO: This is an ugly hack. The writer already wrote the header, so we have to
        // go back and write the right header.
        result.seek_to_start()?;
        result.write_header()?;
        Ok(result)
    }

    /// Creates an empty vector stored in the specified file with user-defined buffer size.
    ///
    /// If the file already exists, it will be overwritten.
    /// Actual buffer size may be slightly higher than requested.
    /// Behavior is undefined if `buf_len * width > usize::MAX`.
    ///
    /// # Arguments
    ///
    /// * `filename`: Name of the file.
    /// * `width`: Width of each element in bits.
    /// * `buf_len`: Buffer size in elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::int_vector::IntVectorWriter;
    /// use simple_sds::ops::Element;
    /// use simple_sds::serialize;
    /// use std::{fs, mem};
    ///
    /// let filename = serialize::temp_file_name("int-vector-writer-with-buf-len");
    /// let mut v = IntVectorWriter::with_buf_len(&filename, 13, 1024).unwrap();
    /// assert!(v.is_empty());
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error of the kind [`ErrorKind::Other`] if the width is invalid.
    /// Any I/O errors will be passed through.
    pub fn with_buf_len<P: AsRef<Path>>(filename: P, width: usize, buf_len: usize) -> io::Result<IntVectorWriter> {
        if width == 0 || width > bits::WORD_BITS {
            return Err(Error::new(ErrorKind::Other, format!("invalid element width: {}", width)));
        }
        let writer = RawVectorWriter::with_buf_len(filename, buf_len * width)?;
        let mut result = IntVectorWriter {
            len: 0,
            width: width,
            writer: writer,
        };
        // TODO: This is an ugly hack. The writer already wrote the header, so we have to
        // go back and write the right header.
        result.seek_to_start()?;
        result.write_header()?;
        Ok(result)
    }
}

//-----------------------------------------------------------------------------

impl Element for IntVectorWriter {
    type Item = u64;

    fn len(&self) -> usize {
        self.len
    }

    fn width(&self) -> usize {
        self.width
    }
}

impl Push for IntVectorWriter {
    fn push(&mut self, value: <Self as Element>::Item) {
        self.writer.push_int(value, self.width());
        self.len += 1;
    }
}

impl Writer for IntVectorWriter {
    fn file(&mut self) -> Option<&mut File> {
        self.writer.file()
    }

    fn flush(&mut self, mode: FlushMode) -> io::Result<()> {
        self.writer.flush(mode)
    }

    fn write_header(&mut self) -> io::Result<()> {
        if let Some(f) = self.writer.file() {
            self.len.serialize(f)?;
            self.width.serialize(f)?;
        }
        self.writer.write_header()?;
        Ok(())
    }

    fn close_file(&mut self) {
        self.writer.close_file()
    }
}

impl Drop for IntVectorWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

//-----------------------------------------------------------------------------

macro_rules! iter_to_int_vector {
    ($t:ident, $w:expr) => {
        impl FromIterator<$t> for IntVector {
            fn from_iter<I: IntoIterator<Item = $t>>(iter: I) -> Self {
                let mut result = IntVector::new($w).unwrap();
                result.extend(iter);
                result
            }
        }

        impl Extend<$t> for IntVector {
            fn extend<I: IntoIterator<Item = $t>>(&mut self, iter: I) {
                let mut iter = iter.into_iter();
                let (lower_bound, _) = iter.size_hint();
                self.reserve(lower_bound);
                while let Some(value) = iter.next() {
                    self.push(value as <Self as Element>::Item);
                }
            }
        }

        impl Extend<$t> for IntVectorWriter {
            fn extend<I: IntoIterator<Item = $t>>(&mut self, iter: I) {
                for value in iter {
                    self.push(value as <Self as Element>::Item);
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{Element, Resize, Pack, Access, Push, Pop};
    use crate::serialize::{Serialize, Writer};
    use crate::serialize;
    use std::fs;
    use rand::Rng;

    #[test]
    fn empty_vector() {
        let empty = IntVector::default();
        assert!(empty.is_empty(), "Created a non-empty empty vector");
        assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
        assert_eq!(empty.width(), 64, "Invalid width for an empty vector");
        assert_eq!(empty.capacity(), 0, "Reserved unnecessary memory for an empty vector");

        let with_width = IntVector::new(13).unwrap();
        assert!(with_width.is_empty(), "Created a non-empty empty vector with a specified width");
        assert_eq!(with_width.width(), 13, "Invalid width for an empty vector with a specified width");
        assert_eq!(with_width.capacity(), 0, "Reserved unnecessary memory for an empty vector with a specified width");

        let with_capacity = IntVector::with_capacity(137, 13).unwrap();
        assert!(with_capacity.is_empty(), "Created a non-empty vector by specifying capacity");
        assert_eq!(with_width.width(), 13, "Invalid width for an empty vector with a specified capacity");
        assert!(with_capacity.capacity() >= 137, "Vector capacity is lower than specified");
    }

    #[test]
    fn with_len_and_clear() {
        let mut v = IntVector::with_len(137, 13, 123).unwrap();
        assert_eq!(v.len(), 137, "Vector length is not as specified");
        assert_eq!(v.width(), 13, "Vector width is not as specified");
        v.clear();
        assert!(v.is_empty(), "Could not clear the vector");
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
    fn push_pop_from_iter() {
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
        assert!(v.is_empty(), "Non-empty vector after popping all ints");
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

        let raw = RawVector::from(v.clone());
        assert_eq!(raw.len(), v.len() * v.width(), "Invalid length for the extracted RawVector");
        for i in 0..v.len() {
            assert_eq!(raw.int(i * v.width(), v.width()), v.get(i), "Invalid value {} in the RawVector", i);
        }
    }

    #[test]
    fn extend() {
        let first: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        let second: Vec<u64> = vec![1, 2, 4, 8, 16, 32, 64, 128];
        let mut correct: Vec<u64> = Vec::new();
        correct.extend(first.iter().cloned()); correct.extend(second.iter().cloned());

        let mut int_vec = IntVector::new(8).unwrap();
        int_vec.extend(first); int_vec.extend(second);
        assert_eq!(int_vec.len(), correct.len(), "Invalid length for the extended IntVector");

        let collected: Vec<u64> = int_vec.into_iter().collect();
        assert_eq!(collected, correct, "Invalid values collected from the IntVector");
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

        assert_eq!(packed.iter().len(), packed.len(), "Invalid length from the iterator");

        let from_iter: Vec<u64> = packed.into_iter().collect();
        assert_eq!(from_iter, correct, "Invalid vector built from into_iter()");
    }

    #[test]
    fn double_ended_iterator() {
        let correct: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];

        let v: IntVector = correct.iter().cloned().collect();
        let mut index = correct.len();
        let mut iter = v.iter();
        while let Some(value) = iter.next_back() {
            index -= 1;
            assert_eq!(value, correct[index], "Invalid value {} when iterating backwards", index);
        }

        let mut next = 0;
        let mut limit = correct.len();
        let mut iter = v.iter();
        while iter.len() > 0 {
            assert_eq!(iter.next(), Some(correct[next]), "Invalid value {} (forward)", next);
            next += 1;
            if iter.len() == 0 {
                break;
            }
            limit -= 1;
            assert_eq!(iter.next_back(), Some(correct[limit]), "Invalid value {} (backward)", limit);
        }
        assert_eq!(next, limit, "Iterator did not visit all values");
    }

    #[test]
    fn serialize() {
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

    #[test]
    fn empty_writer() {
        let first = serialize::temp_file_name("empty-int-vector-writer");
        let second = serialize::temp_file_name("empty-int-vector-writer");

        let mut v = IntVectorWriter::new(&first, 13).unwrap();
        assert!(v.is_empty(), "Created a non-empty empty writer");
        assert!(v.is_open(), "Newly created writer is not open");
        v.close().unwrap();

        let mut w = IntVectorWriter::with_buf_len(&second, 13, 1024).unwrap();
        assert!(w.is_empty(), "Created a non-empty empty writer with custom buffer size");
        assert!(w.is_open(), "Newly created writer is not open with custom buffer size");
        w.close().unwrap();

        fs::remove_file(&first).unwrap();
        fs::remove_file(&second).unwrap();
    }

    #[test]
    fn push_to_writer() {
        let filename = serialize::temp_file_name("push-to-int-vector-writer");

        let mut correct: Vec<u64> = Vec::new();
        let mut rng = rand::thread_rng();
        let width = 31;
        for _ in 0..71 {
            let value: u64 = rng.gen();
            correct.push(value & bits::low_set(width));
        }

        let mut v = IntVectorWriter::with_buf_len(&filename, width, 32).unwrap();
        for value in correct.iter() {
            v.push(*value);
        }
        assert_eq!(v.len(), correct.len(), "Invalid size for the writer");
        v.close().unwrap();
        assert!(!v.is_open(), "Could not close the writer");

        let w: IntVector = serialize::load_from(&filename).unwrap();
        assert_eq!(w.len(), correct.len(), "Invalid size for the loaded vector");
        for i in 0..correct.len() {
            assert_eq!(w.get(i), correct[i], "Invalid integer {}", i);
        }

        fs::remove_file(&filename).unwrap();
    }

    #[test]
    fn extend_writer() {
        let filename = serialize::temp_file_name("extend-int-vector-writer");

        let first: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        let second: Vec<u64> = vec![1, 2, 4, 8, 16, 32, 64, 128];
        let mut correct: Vec<u64> = Vec::new();
        correct.extend(first.iter().cloned()); correct.extend(second.iter().cloned());

        let mut writer = IntVectorWriter::with_buf_len(&filename, 8, 32).unwrap();
        writer.extend(first); writer.extend(second);
        assert_eq!(writer.len(), correct.len(), "Invalid length for the extended writer");
        writer.close().unwrap();

        let int_vec: IntVector = serialize::load_from(&filename).unwrap();
        let collected: Vec<u64> = int_vec.into_iter().collect();
        assert_eq!(collected, correct, "Invalid values collected from the writer");

        fs::remove_file(&filename).unwrap();
    }

    #[test]
    #[ignore]
    fn large_writer() {
        let filename = serialize::temp_file_name("large-int-vector-writer");

        let mut correct: Vec<u64> = Vec::new();
        let mut rng = rand::thread_rng();
        let width = 31;
        for _ in 0..620001 {
            let value: u64 = rng.gen();
            correct.push(value & bits::low_set(width));
        }

        let mut v = IntVectorWriter::new(&filename, width).unwrap();
        for value in correct.iter() {
            v.push(*value);
        }
        assert_eq!(v.len(), correct.len(), "Invalid size for the writer");
        v.close().unwrap();
        assert!(!v.is_open(), "Could not close the writer");

        let w: IntVector = serialize::load_from(&filename).unwrap();
        assert_eq!(w.len(), correct.len(), "Invalid size for the loaded vector");
        for i in 0..correct.len() {
            assert_eq!(w.get(i), correct[i], "Invalid integer {}", i);
        }

        fs::remove_file(&filename).unwrap();
    }
}

//-----------------------------------------------------------------------------
