//! A bit-packed integer vector storing fixed-width integers.

use crate::ops::{Element, Resize, Pack, Access, Push, Pop};
use crate::raw_vector::{RawVector, RawVectorMapper, RawVectorWriter, AccessRaw, PushRaw, PopRaw};
use crate::serialize::{MemoryMap, MemoryMapped, Serialize, Writer, FlushMode};
use crate::bits;

use std::fs::File;
use std::io::{Error, ErrorKind};
use std::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator, FromIterator, Extend};
use std::path::Path;
use std::io;

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

/// A contiguous growable bit-packed array of fixed-width integers.
///
/// This structure contains [`RawVector`], which is in turn contains [`Vec`].
/// Each element consists of the lowest 1 to 64 bits of a [`u64`] value, as specified by parameter `width`.
/// The maximum length of the vector is `usize::MAX / width` elements.
///
/// A default constructed `IntVector` has `width == 64`.
/// `IntVector` can be built from an iterator over [`u8`], [`u16`], [`u32`], [`u64`], or [`usize`] values.
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
    /// Returns [`Err`] if the width is invalid.
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
    pub fn new(width: usize) -> Result<IntVector, &'static str> {
        if width == 0 || width > bits::WORD_BITS {
            Err("Integer width must be 1 to 64 bits")
        }
        else {
            Ok(IntVector {
                len: 0,
                width: width,
                data: RawVector::new(),
            })
        }
    }

    /// Creates an initialized vector of specified length and width.
    /// 
    /// Returns [`Err`] if the width is invalid.
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
    ///
    /// # Panics
    ///
    /// May panic if the vector would exceed the maximum length.
    pub fn with_len(len: usize, width: usize, value: u64) -> Result<IntVector, &'static str> {
        if width == 0 || width > bits::WORD_BITS {
            return Err("Integer width must be 1 to 64 bits");
        }
        let mut data = RawVector::with_capacity(len * width);
        for _ in 0..len {
            unsafe { data.push_int(value, width); }
        }
        Ok(IntVector {
            len: len,
            width: width,
            data: data,
        })
    }

    /// Creates an empty vector with enough capacity for at least the specified number of elements of specified width.
    ///
    /// Returns [`Err`] if the width is invalid.
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
    ///
    /// # Panics
    ///
    /// May panic if the capacity would exceed the maximum length.
    pub fn with_capacity(capacity: usize, width: usize) -> Result<IntVector, &'static str> {
        if width == 0 || width > bits::WORD_BITS {
            Err("Integer width must be 1 to 64 bits")
        } else {
            Ok(IntVector {
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

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn max_len(&self) -> usize {
        usize::MAX / self.width()
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

    #[inline]
    fn capacity(&self) -> usize {
        self.data.capacity() / self.width()
    }

    fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional * self.width());
    }
}

impl Pack for IntVector {
    fn pack(&mut self) {
        if self.is_empty() {
            return;
        }
        let new_width = bits::bit_len(self.iter().max().unwrap());
        if new_width == self.width() {
            return;
        }
        let mut new_data = RawVector::with_capacity(self.len() * new_width);
        for value in self.iter() {
            unsafe { new_data.push_int(value, new_width); }
        }
        self.width = new_width;
        self.data = new_data;
    }
}

impl Access for IntVector {
    #[inline]
    fn get(&self, index: usize) -> <Self as Element>::Item {
        assert!(index < self.len(), "Index is out of bounds");
        unsafe { self.data.int(index * self.width(), self.width()) }
    }

    #[inline]
    fn is_mutable(&self) -> bool {
        true
    }

    #[inline]
    fn set(&mut self, index: usize, value: <Self as Element>::Item) {
        assert!(index < self.len(), "Index is out of bounds");
        unsafe { self.data.set_int(index * self.width(), value, self.width()); }
    }
}

impl Push for IntVector {
    #[inline]
    fn push(&mut self, value: <Self as Element>::Item) {
        unsafe { self.data.push_int(value, self.width()); }
        self.len += 1;
    }    
}

impl Pop for IntVector {
    #[inline]
    fn pop(&mut self) -> Option<<Self as Element>::Item> {
        if self.len() > 0 {
            self.len -= 1;
        }
        unsafe { self.data.pop_int(self.width()) }
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
        if len * width != data.len() {
            Err(Error::new(ErrorKind::InvalidData, "Data length does not match len * width"))
        }
        else {
            Ok(IntVector {
                len: len,
                width: width,
                data: data,
            })
        }
    }

    fn size_in_elements(&self) -> usize {
        self.len.size_in_elements() + self.width.size_in_elements() + self.data.size_in_elements()
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
    #[inline]
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
/// The type of `Item` is [`u64`].
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
}

impl<'a> ExactSizeIterator for Iter<'a> {}

impl<'a> FusedIterator for Iter<'a> {}

//-----------------------------------------------------------------------------

/// [`IntVector`] transformed into an iterator.
///
/// The type of `Item` is [`u64`].
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

    #[inline]
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
///
/// # Examples
///
/// ```
/// use simple_sds::int_vector::{IntVector, IntVectorWriter};
/// use simple_sds::ops::{Element, Access, Push};
/// use simple_sds::serialize::Writer;
/// use simple_sds::serialize;
/// use std::fs;
///
/// let filename = serialize::temp_file_name("int-vector-writer");
/// let mut writer = IntVectorWriter::new(&filename, 13).unwrap();
/// assert!(writer.is_empty());
/// writer.push(123); writer.push(456); writer.push(789);
/// assert_eq!(writer.len(), 3);
/// writer.close().unwrap();
///
/// let v: IntVector = serialize::load_from(&filename).unwrap();
/// assert_eq!(v.len(), 3);
/// assert_eq!(v.get(0), 123); assert_eq!(v.get(1), 456); assert_eq!(v.get(2), 789);
/// fs::remove_file(&filename).unwrap();
/// ```
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
    /// # Errors
    ///
    /// Returns an error of the kind [`ErrorKind::Other`] if the width is invalid.
    /// Any I/O errors will be passed through.
    pub fn new<P: AsRef<Path>>(filename: P, width: usize) -> io::Result<IntVectorWriter> {
        if width == 0 || width > bits::WORD_BITS {
            return Err(Error::new(ErrorKind::Other, "Integer width must be 1 to 64 bits"));
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
    ///
    /// # Arguments
    ///
    /// * `filename`: Name of the file.
    /// * `width`: Width of each element in bits.
    /// * `buf_len`: Buffer size in elements.
    ///
    /// # Errors
    ///
    /// Returns an error of the kind [`ErrorKind::Other`] if the width is invalid.
    /// Any I/O errors will be passed through.
    ///
    /// # Panics
    ///
    /// May panic if buffer length would exceed the maximum length.
    pub fn with_buf_len<P: AsRef<Path>>(filename: P, width: usize, buf_len: usize) -> io::Result<IntVectorWriter> {
        if width == 0 || width > bits::WORD_BITS {
            return Err(Error::new(ErrorKind::Other, "Integer width must be 1 to 64 bits"));
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

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn max_len(&self) -> usize {
        usize::MAX / self.width()
    }
}

impl Push for IntVectorWriter {
    #[inline]
    fn push(&mut self, value: <Self as Element>::Item) {
        unsafe { self.writer.push_int(value, self.width()); }
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

/// An immutable memory-mapped [`IntVector`].
///
/// This is compatible with the serialization format of [`IntVector`].
///
/// # Examples
///
/// ```
/// use simple_sds::int_vector::{IntVector, IntVectorMapper};
/// use simple_sds::ops::{Element, Access};
/// use simple_sds::serialize::{MemoryMap, MemoryMapped, MappingMode};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let filename = serialize::temp_file_name("int-vector-mapper");
/// let mut v = IntVector::with_len(3, 13, 0).unwrap();
/// v.set(0, 123); v.set(1, 456); v.set(2, 789);
/// serialize::serialize_to(&v, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// let mapper = IntVectorMapper::new(&map, 0).unwrap();
/// assert_eq!(mapper.len(), v.len());
/// for i in 0..mapper.len() {
///     assert_eq!(mapper.get(i), v.get(i));
/// }
///
/// drop(mapper); drop(map);
/// fs::remove_file(&filename).unwrap();
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct IntVectorMapper<'a> {
    len: usize,
    width: usize,
    data: RawVectorMapper<'a>,
}

impl<'a> IntVectorMapper<'a> {
    /// Returns an iterator visiting all elements of the vector in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::int_vector::{IntVector, IntVectorMapper};
    /// use simple_sds::serialize::{MemoryMap, MemoryMapped, MappingMode};
    /// use simple_sds::serialize;
    /// use std::fs;
    ///
    /// let filename = serialize::temp_file_name("int-vector-mapper-iter");
    /// let v = IntVector::from(vec![123u32, 456u32, 789u32, 10u32]);
    /// serialize::serialize_to(&v, &filename);
    ///
    /// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
    /// let mapper = IntVectorMapper::new(&map, 0).unwrap();
    /// assert!(mapper.iter().eq(v.iter()));
    ///
    /// drop(mapper); drop(map);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    pub fn iter(&self) -> MappedIter<'_> {
        MappedIter {
            parent: self,
            next: 0,
            limit: self.len(),
        }
    }
}

impl<'a> Element for IntVectorMapper<'a> {
    type Item = u64;

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn width(&self) -> usize {
        self.width
    }

    #[inline]
    fn max_len(&self) -> usize {
        usize::MAX / self.width()
    }
}

impl<'a> Access for IntVectorMapper<'a> {
    #[inline]
    fn get(&self, index: usize) -> <Self as Element>::Item {
        assert!(index < self.len(), "Index is out of bounds");
        unsafe { self.data.int(index * self.width(), self.width()) }
    }

    #[inline]
    fn is_mutable(&self) -> bool {
        false
    }

    #[inline]
    fn set(&mut self, _: usize, _: <Self as Element>::Item) {
        panic!("Not implemented");
    }
}

impl<'a> MemoryMapped<'a> for IntVectorMapper<'a> {
    fn new(map: &'a MemoryMap, offset: usize) -> io::Result<Self> {
        if offset + 1 >= map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The starting offset is out of range"));
        }
        let slice: &[u64] = map.as_ref();
        let len = slice[offset] as usize;
        let width = slice[offset + 1] as usize;
        let data = RawVectorMapper::new(map, offset + 2)?;
        Ok(IntVectorMapper {
            len: len,
            width: width,
            data: data,
        })
    }

    fn map_offset(&self) -> usize {
        self.data.map_offset() - 2
    }

    fn map_len(&self) -> usize {
        self.data.map_len() + 2
    }
}

impl<'a> AsRef<RawVectorMapper<'a>> for IntVectorMapper<'a> {
    #[inline]
    fn as_ref(&self) -> &RawVectorMapper<'a> {
        &(self.data)
    }
}

//-----------------------------------------------------------------------------

/// A read-only iterator over [`IntVectorMapper`].
///
/// The type of `Item` is [`u64`].
///
/// # Examples
///
/// ```
/// use simple_sds::int_vector::{IntVector, IntVectorMapper};
/// use simple_sds::serialize::{MemoryMap, MemoryMapped, MappingMode};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let filename = serialize::temp_file_name("int-vector-mapped-iter");
/// let source: Vec<u16> = vec![12, 34, 56, 78, 90, 11, 12, 13];
/// let v: IntVector = source.iter().cloned().collect();
/// serialize::serialize_to(&v, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// let mapper = IntVectorMapper::new(&map, 0).unwrap();
/// for (index, value) in mapper.iter().enumerate() {
///     assert_eq!(value, source[index] as u64);
/// }
///
/// drop(mapper); drop(map);
/// fs::remove_file(&filename).unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct MappedIter<'a> {
    parent: &'a IntVectorMapper<'a>,
    // The first index we have not used.
    next: usize,
    // The first index we should not use.
    limit: usize,
}

impl<'a> Iterator for MappedIter<'a> {
    type Item = <IntVectorMapper<'a> as Element>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.limit {
            None
        } else {
            let result = Some(self.parent.get(self.next));
            self.next += 1;
            result
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.limit - self.next;
        (remaining, Some(remaining))
    }
}

impl<'a> DoubleEndedIterator for MappedIter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.next >= self.limit {
            None
        } else {
            self.limit -= 1;
            Some(self.parent.get(self.limit))
        }
    }
}

impl<'a> ExactSizeIterator for MappedIter<'a> {}

impl<'a> FusedIterator for MappedIter<'a> {}

//-----------------------------------------------------------------------------

macro_rules! from_extend_int_vector {
    ($t:ident, $w:expr) => {
        impl From<Vec<$t>> for IntVector {
            fn from(v: Vec<$t>) -> Self {
                let mut result = IntVector::with_capacity(v.len(), $w).unwrap();
                result.extend(v);
                result
            }
        }

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

from_extend_int_vector!(u8, 8);
from_extend_int_vector!(u16, 16);
from_extend_int_vector!(u32, 32);
from_extend_int_vector!(u64, 64);
from_extend_int_vector!(usize, 64);

//-----------------------------------------------------------------------------
