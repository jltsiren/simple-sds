//! The basic vector implementing the low-level functionality used by other vectors in the crate.

use crate::serialize::{Serialize, Writer, FlushMode};
use crate::bits;

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::io;

//-----------------------------------------------------------------------------

/// Random access to bits and variable-width integers in a bit array.
///
/// # Examples
///
/// ```
/// use simple_sds::raw_vector::AccessRaw;
/// use simple_sds::bits;
///
/// struct Example(Vec<u64>);
///
/// impl AccessRaw for Example {
///     fn bit(&self, bit_offset: usize) -> bool {
///         let (index, offset) = bits::split_offset(bit_offset);
///         (self.0[index] & (1u64 << offset)) != 0
///     }
///
///     fn int(&self, bit_offset: usize, width: usize) -> u64 {
///         bits::read_int(&self.0, bit_offset, width)
///     }
///
///     fn word(&self, index: usize) -> u64 {
///         self.0[index]
///     }
///
///     unsafe fn word_unchecked(&self, index: usize) -> u64 {
///         *self.0.get_unchecked(index)
///     }
///
///     fn is_mutable(&self) -> bool {
///         true
///     }
///
///     fn set_bit(&mut self, bit_offset: usize, value: bool) {
///         let (index, offset) = bits::split_offset(bit_offset);
///         self.0[index] &= !(1u64 << offset);
///         self.0[index] |= (value as u64) << offset;
///     }
///
///     fn set_int(&mut self, bit_offset: usize, value: u64, width: usize) {
///         bits::write_int(&mut self.0, bit_offset, value, width);
///     }
/// }
///
/// let mut example = Example(vec![0u64; 2]);
/// assert!(example.is_mutable());
///
/// example.set_int(4, 0x33, 8);
/// example.set_int(63, 2, 2);
/// example.set_bit(72, true);
/// assert_eq!(example.0[0], 0x330);
/// assert_eq!(example.0[1], 0x101);
///
/// assert!(example.bit(72));
/// assert!(!example.bit(68));
/// assert_eq!(example.int(4, 8), 0x33);
/// assert_eq!(example.int(63, 2), 2);
/// assert_eq!(example.word(1), 0x101);
/// ```
pub trait AccessRaw {
    /// Reads a bit from the array.
    ///
    /// # Panics
    ///
    /// May panic if `bit_offset` is not a valid offset in the bit array.
    /// May panic from I/O errors.
    fn bit(&self, bit_offset: usize) -> bool;

    /// Reads an integer from the container.
    ///
    /// Behavior is undefined if `width > 64`.
    ///
    /// # Arguments
    ///
    /// * `bit_offset`: Starting offset in the bit array.
    /// * `width`: The width of the integer in bits.
    ///
    /// # Panics
    ///
    /// May panic if `bit_offset + width - 1` is not a valid offset in the bit array.
    /// May panic from I/O errors.
    fn int(&self, bit_offset: usize, width: usize) -> u64;

    /// Reads a 64-bit word from the container.
    ///
    /// This may be faster than calling `self.int(index * 64, 64)`.
    ///
    /// # Panics
    ///
    /// May panic if `index * 64` is not a valid offset in the bit array.
    /// May panic from I/O errors.
    fn word(&self, index: usize) -> u64;

    /// Unsafe version of [`AccessRaw::word`] without bounds checks.
    ///
    /// Behavior is undefined in situations where the safe versions may panic.
    unsafe fn word_unchecked(&self, index: usize) -> u64;

    /// Returns `true` if the underlying data is mutable.
    ///
    /// This is relevant, for example, with memory-mapped vectors, where the underlying file may be opened as read-only.
    fn is_mutable(&self) -> bool;

    /// Writes a bit to the container.
    ///
    /// # Arguments
    ///
    /// * `bit_offset`: Offset in the bit array.
    /// * `value`: The value of the bit.
    ///
    /// # Panics
    ///
    /// May panic if `bit_offset` is not a valid offset in the bit array.
    /// May panic if the underlying data is not mutable.
    /// May panic from I/O errors.
    fn set_bit(&mut self, bit_offset: usize, value: bool);

    /// Writes an integer to the container.
    ///
    /// Behavior is undefined if `width > 64`.
    ///
    /// # Arguments
    ///
    /// * `bit_offset`: Starting offset in the bit array.
    /// * `value`: The integer to be written.
    /// * `width`: The width of the integer in bits.
    ///
    /// # Panics
    ///
    /// May panic if `bit_offset + width - 1` is not a valid offset in the bit array.
    /// May panic if the underlying data is not mutable.
    /// May panic from I/O errors.
    fn set_int(&mut self, bit_offset: usize, value: u64, width: usize);
}

//-----------------------------------------------------------------------------

/// Append bits and variable-width integers to a container.
///
/// The container is not required to remember the types of the pushed items.
///
/// # Examples
/// ```
/// use simple_sds::raw_vector::PushRaw;
/// use simple_sds::bits;
///
/// struct Example(Vec<bool>, Vec<u64>);
///
/// impl Example{
///     fn new() -> Example {
///         Example(Vec::new(), Vec::new())
///     }
/// }
///
/// impl PushRaw for Example {
///     fn push_bit(&mut self, value: bool) {
///         self.0.push(value);
///     }
///
///     fn push_int(&mut self, value: u64, width: usize) {
///         self.1.push(value & bits::low_set(width));
///     }
/// }
///
/// let mut example = Example::new();
/// example.push_bit(false);
/// example.push_int(123, 8);
/// example.push_int(456, 9);
/// example.push_bit(true);
///
/// assert_eq!(example.0.len(), 2);
/// assert_eq!(example.1.len(), 2);
/// ```
pub trait PushRaw {
    /// Appends a bit to the container.
    ///
    /// Behavior is undefined if there is an integer overflow.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn push_bit(&mut self, value: bool);

    /// Appends an integer to the container.
    ///
    /// Behavior is undefined if there is an integer overflow.
    ///
    /// # Arguments
    ///
    /// * `value`: The integer to be appended.
    /// * `width`: The width of the integer in bits.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    fn push_int(&mut self, value: u64, width: usize);
}

/// Remove and return bits and variable-width integers from a container.
///
/// Behavior is implementation-dependent if the sequence of pop operations is not the reverse of push operations.
///
/// # Examples
/// ```
/// use simple_sds::raw_vector::PopRaw;
///
/// struct Example(Vec<bool>, Vec<u64>);
///
/// impl Example{
///     fn new() -> Example {
///         Example(Vec::new(), Vec::new())
///     }
/// }
///
/// impl PopRaw for Example {
///     fn pop_bit(&mut self) -> Option<bool> {
///         self.0.pop()
///     }
///
///     fn pop_int(&mut self, _: usize) -> Option<u64> {
///         self.1.pop()
///     }
/// }
///
/// let mut example = Example::new();
/// example.0.push(false);
/// example.1.push(123);
/// example.1.push(456);
/// example.0.push(true);
///
/// assert_eq!(example.pop_bit().unwrap(), true);
/// assert_eq!(example.pop_int(9).unwrap(), 456);
/// assert_eq!(example.pop_int(8).unwrap(), 123);
/// assert_eq!(example.pop_bit().unwrap(), false);
/// assert_eq!(example.pop_bit(), None);
/// assert_eq!(example.pop_int(1), None);
/// ```
pub trait PopRaw {
    /// Removes and returns the last bit from the container.
    /// Returns `None` the container does not have more bits.
    fn pop_bit(&mut self) -> Option<bool>;

    /// Removes and returns the last `width` bits from the container as an integer.
    /// Returns `None` if the container does not have more integers of that width.
    ///
    /// Behavior is undefined if `width > 64`.
    fn pop_int(&mut self, width: usize) -> Option<u64>;
}

//-----------------------------------------------------------------------------

/// A contiguous growable array of bits and up to 64-bit integers based on [`Vec`] of `u64` values.
///
/// There are no iterators over the vector, because it may contain items of varying widths.
///
/// # Notes
/// * The unused part of the last integer is always set to `0`.
/// * The underlying vector may allocate but not use more integers than are strictly necessary.
/// * `RawVector` never panics from I/O errors.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RawVector {
    len: usize,
    data: Vec<u64>,
}

impl RawVector {
    /// Returns the length of the vector in bits.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the capacity of the vector in bits.
    #[inline]
    pub fn capacity(&self) -> usize {
        bits::words_to_bits(self.data.capacity())
    }

    /// Counts the number of ones in the bit array.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::{RawVector, AccessRaw};
    ///
    /// let mut v = RawVector::with_len(137, false);
    /// assert_eq!(v.count_ones(), 0);
    /// v.set_bit(1, true); v.set_bit(33, true); v.set_bit(95, true); v.set_bit(123, true);
    /// assert_eq!(v.count_ones(), 4);
    /// ```
    pub fn count_ones(&self) -> usize {
        let mut result: usize = 0;
        for value in self.data.iter() {
            result += (*value).count_ones() as usize;
        }
        result
    }

    /// Creates an empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let v = RawVector::new();
    /// assert!(v.is_empty());
    /// assert_eq!(v.capacity(), 0);
    /// ```
    pub fn new() -> RawVector {
        RawVector::default()
    }

    /// Creates an initialized vector of specified length.
    ///
    /// # Arguments
    ///
    /// * `len`: Length of the vector in bits.
    /// * `value`: Initialization value.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let v = RawVector::with_len(137, false);
    /// assert_eq!(v.len(), 137);
    /// ```
    pub fn with_len(len: usize, value: bool) -> RawVector {
        let val = bits::filler_value(value);
        let data: Vec<u64> = vec![val; bits::bits_to_words(len)];
        let mut result = RawVector {
            len: len,
            data: data,
        };
        result.set_unused_bits(false);
        result
    }

    /// Creates an empty vector with enough capacity for at least `capacity` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let v = RawVector::with_capacity(137);
    /// assert!(v.capacity() >= 137);
    /// ```
    pub fn with_capacity(capacity: usize) -> RawVector {
        RawVector {
            len: 0,
            data: Vec::with_capacity(bits::bits_to_words(capacity)),
        }
    }

    /// Returns a copy of the vector with each bit flipped.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::{RawVector, AccessRaw};
    ///
    /// let mut original = RawVector::with_len(137, false);
    /// original.set_bit(1, true); original.set_bit(33, true);
    /// original.set_int(95, 456, 9); original.set_bit(123, true);
    /// let complement = original.complement();
    /// for i in 0..137 {
    ///     assert_eq!(!(complement.bit(i)), original.bit(i));
    /// }
    /// ```
    pub fn complement(&self) -> RawVector {
        let mut result = self.clone();
        for word in result.data.iter_mut() {
            *word = !*word;
        }
        result.set_unused_bits(false);
        result
    }

    /// Resizes the vector to a specified length.
    ///
    /// If `new_len > self.len()`, the new `new_len - self.len()` bits will be initialized.
    /// If `new_len < self.len()`, the vector is truncated.
    ///
    /// # Arguments
    ///
    /// * `new_len`: New length of the vector in bits.
    /// * `value`: Initialization value.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::new();
    /// v.resize(137, true);
    /// let w = RawVector::with_len(137, true);
    /// assert_eq!(v, w);
    /// ```
    pub fn resize(&mut self, new_len: usize, value: bool) {

        // Fill the unused bits if necessary.
        if new_len > self.len() {
            self.set_unused_bits(value);
        }

        // Use more space if necessary.
        self.data.resize(bits::bits_to_words(new_len), bits::filler_value(value));
        self.len = new_len;
        self.set_unused_bits(false);
    }

    /// Clears the vector without freeing the data.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::with_len(137, true);
    /// assert_eq!(v.len(), 137);
    /// v.clear();
    /// assert!(v.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    /// Reserves space for storing at least `self.len() + additional` bits in the vector.
    ///
    /// Does nothing if the capacity is already sufficient.
    /// Behavior is undefined if `self.len() + additional > usize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::with_len(137, false);
    /// v.reserve(318);
    /// assert!(v.capacity() >= 137 + 318);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        let words_needed = bits::bits_to_words(self.len() + additional);
        if words_needed > self.data.capacity() {
            self.data.reserve(words_needed - self.data.capacity());
        }
    }

    // Set the unused bits in the last integer to the specified value.
    fn set_unused_bits(&mut self, value: bool) {
        let (index, width) = bits::split_offset(self.len());
        if width > 0 {
            if value {
                self.data[index] |= !bits::low_set(width);
            }
            else {
                self.data[index] &= bits::low_set(width);
            }
        }
    }
}

//-----------------------------------------------------------------------------

impl AccessRaw for RawVector {
    #[inline]
    fn bit(&self, bit_offset: usize) -> bool {
        let (index, offset) = bits::split_offset(bit_offset);
        ((self.data[index] >> offset) & 1) == 1
    }

    #[inline]
    fn int(&self, bit_offset: usize, width: usize) -> u64 {
        bits::read_int(&self.data, bit_offset, width)
    }

    #[inline]
    fn word(&self, index: usize) -> u64 {
        self.data[index]
    }

    #[inline]
    unsafe fn word_unchecked(&self, index: usize) -> u64 {
        *self.data.get_unchecked(index)
    }

    #[inline]
    fn is_mutable(&self) -> bool {
        true
    }

    #[inline]
    fn set_bit(&mut self, bit_offset: usize, value: bool) {
        let (index, offset) = bits::split_offset(bit_offset);
        self.data[index] &= !(1u64 << offset);
        self.data[index] |= (value as u64) << offset;
    }

    #[inline]
    fn set_int(&mut self, bit_offset: usize, value: u64, width: usize) {
        bits::write_int(&mut self.data, bit_offset, value, width);
    }
}

impl PushRaw for RawVector {
    fn push_bit(&mut self, value: bool) {
        let (index, offset) = bits::split_offset(self.len);
        if index == self.data.len() {
            self.data.push(0);
        }
        self.data[index] |= (value as u64) << offset;
        self.len += 1;
    }

    fn push_int(&mut self, value: u64, width: usize) {
        if self.len + width > bits::words_to_bits(self.data.len()) {
            self.data.push(0);
        }
        bits::write_int(&mut self.data, self.len, value, width);
        self.len += width;
    }
}

impl PopRaw for RawVector {
    fn pop_bit(&mut self) -> Option<bool> {
        if self.len() > 0 {
            let result = self.bit(self.len - 1);
            self.len -= 1;
            self.data.resize(bits::bits_to_words(self.len()), 0); // Avoid using unnecessary words.
            self.set_unused_bits(false);
            Some(result)
        } else {
            None
        }
    }

    fn pop_int(&mut self, width: usize) -> Option<u64> {
        if self.len() >= width {
            let result = self.int(self.len - width, width);
            self.len -= width;
            self.data.resize(bits::bits_to_words(self.len()), 0); // Avoid using unnecessary words.
            self.set_unused_bits(false);
            Some(result)
        } else {
            None
        }
    }
}

impl Serialize for RawVector {
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.len.serialize(writer)?;
        self.data.serialize_header(writer)?;
        Ok(())
    }

    fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.data.serialize_body(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let len = usize::load(reader)?;
        let data = <Vec<u64> as Serialize>::load(reader)?;
        Ok(RawVector {
            len: len,
            data: data,
        })
    }

    fn size_in_bytes(&self) -> usize {
        self.len.size_in_bytes() + self.data.size_in_bytes()
    }
}

//-----------------------------------------------------------------------------

impl AsRef<Vec<u64>> for RawVector {
    #[inline]
    fn as_ref(&self) -> &Vec<u64> {
        &(self.data)
    }
}

//-----------------------------------------------------------------------------

/// A buffered file writer compatible with the serialization format of [`RawVector`].
///
/// When the writer goes out of scope, the internal buffer is flushed, the file is closed, and all errors are ignored.
/// Call [`RawVectorWriter::close`] explicitly to handle the errors.
#[derive(Debug)]
pub struct RawVectorWriter {
    len: usize,
    buf_len: usize,
    file: Option<File>,
    buf: RawVector,
}

impl RawVectorWriter {
    /// Default buffer size in bits.
    pub const DEFAULT_BUFFER_SIZE: usize = 8 * 1024 * 1024;

    /// Returns the length of the vector in bits.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates an empty vector stored in the specified file with the default buffer size.
    ///
    /// If the file already exists, it will be overwritten.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVectorWriter;
    /// use simple_sds::serialize;
    /// use std::{fs, mem};
    ///
    /// let filename = serialize::temp_file_name("raw-vector-writer-new");
    /// let mut v = RawVectorWriter::new(&filename).unwrap();
    /// assert!(v.is_empty());
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Any I/O errors will be passed through.
    pub fn new<P: AsRef<Path>>(filename: P) -> io::Result<RawVectorWriter> {
        let mut options = OpenOptions::new();
        let file = options.create(true).write(true).truncate(true).open(filename)?;
        // Allocate one extra word for overflow.
        let buf = RawVector::with_capacity(Self::DEFAULT_BUFFER_SIZE + bits::WORD_BITS);
        let mut result = RawVectorWriter {
            len: 0,
            buf_len: Self::DEFAULT_BUFFER_SIZE,
            file: Some(file),
            buf: buf,
        };
        result.write_header()?;
        Ok(result)
    }

    /// Creates an empty vector stored in the specified file with user-defined buffer size.
    ///
    /// If the file already exists, it will be overwritten.
    /// The buffer size will be rounded up to the next multiple of [`bits::WORD_BITS`].
    ///
    /// # Arguments
    ///
    /// * `filename`: Name of the file.
    /// * `buf_len`: Buffer size in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVectorWriter;
    /// use simple_sds::serialize;
    /// use std::{fs, mem};
    ///
    /// let filename = serialize::temp_file_name("raw-vector-writer-with-buf-len");
    /// let mut v = RawVectorWriter::with_buf_len(&filename, 1024).unwrap();
    /// assert!(v.is_empty());
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Any I/O errors will be passed through.
    pub fn with_buf_len<P: AsRef<Path>>(filename: P, buf_len: usize) -> io::Result<RawVectorWriter> {
        let buf_len = bits::round_up_to_word_size(buf_len);
        let mut options = OpenOptions::new();
        let file = options.create(true).write(true).truncate(true).open(filename)?;
        // Allocate one extra word for overflow.
        let buf = RawVector::with_capacity(buf_len + bits::WORD_BITS);
        let mut result = RawVectorWriter {
            len: 0,
            buf_len: buf_len,
            file: Some(file),
            buf: buf,
        };
        result.write_header()?;
        Ok(result)
    }
}

//-----------------------------------------------------------------------------

impl PushRaw for RawVectorWriter {
    fn push_bit(&mut self, value: bool) {
        self.buf.push_bit(value); self.len += 1;
        if self.buf.len() >= self.buf_len {
            self.flush(FlushMode::Safe).unwrap();
        }
    }

    fn push_int(&mut self, value: u64, width: usize) {
        self.buf.push_int(value, width); self.len += width;
        if self.buf.len() >= self.buf_len {
            self.flush(FlushMode::Safe).unwrap();
        }
    }
}

impl Writer for RawVectorWriter {
    fn file(&mut self) -> Option<&mut File> {
        self.file.as_mut()
    }

    fn flush(&mut self, mode: FlushMode) -> io::Result<()> {
        if let Some(f) = self.file.as_mut() {
            // Handle the overflow if not serializing the entire buffer.
            let mut overflow: (u64, usize) = (0, 0);
            if let FlushMode::Safe = mode {
                if self.buf.len() > self.buf_len {
                    overflow = (self.buf.int(self.buf_len, self.buf.len() - self.buf_len), self.buf.len() - self.buf_len);
                    self.buf.resize(self.buf_len, false);
                }
            }

            // Serialize and clear the buffer.
            self.buf.serialize_body(f)?;
            self.buf.clear();

            // Push the overflow back to the buffer.
            if let FlushMode::Safe = mode {
                if overflow.1 > 0 {
                    self.buf.push_int(overflow.0, overflow.1);
                }
            }
        }
        Ok(())
    }

    fn write_header(&mut self) -> io::Result<()> {
        if let Some(f) = self.file.as_mut() {
            self.len.serialize(f)?;
            let words: usize = bits::bits_to_words(self.len);
            words.serialize(f)?;
        }
        Ok(())
    }

    fn close_file(&mut self) {
        self.file = None;
    }
}

impl Drop for RawVectorWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize;
    use std::fs;
    use rand::Rng;

    #[test]
    fn empty_vector() {
        let empty = RawVector::new();
        assert!(empty.is_empty(), "Created a non-empty empty vector");
        assert_eq!(empty.len(), 0, "Nonzero length for an empty vector");
        assert_eq!(empty.capacity(), 0, "Reserved unnecessary memory for an empty vector");

        let with_capacity = RawVector::with_capacity(137);
        assert!(with_capacity.is_empty(), "Created a non-empty vector by specifying capacity");
        assert!(with_capacity.capacity() >= 137, "Vector capacity is lower than specified");
    }

    #[test]
    fn with_len_and_clear() {
        let mut v = RawVector::with_len(137, true);
        assert_eq!(v.len(), 137, "Vector length is not as specified");
        v.clear();
        assert!(v.is_empty(), "Could not clear the vector");
    }

    #[test]
    fn initialization_vs_push() {
        let ones_set = RawVector::with_len(137, true);
        let mut ones_pushed = RawVector::new();
        for _ in 0..137 {
            ones_pushed.push_bit(true);
        }
        assert_eq!(ones_set, ones_pushed, "Initializing with and pushing ones yield different vectors");

        let zeros_set = RawVector::with_len(137, false);
        let mut zeros_pushed = RawVector::new();
        for _ in 0..137 {
            zeros_pushed.push_bit(false);
        }
        assert_eq!(zeros_set, zeros_pushed, "Initializing with and pushing zeros yield different vectors");
    }

    #[test]
    fn initialization_vs_resize() {
        let initialized = RawVector::with_len(137, true);

        let mut extended = RawVector::with_len(66, true);
        extended.resize(137, true);
        assert_eq!(extended, initialized, "Extended vector is invalid");

        let mut truncated = RawVector::with_len(212, true);
        truncated.resize(137, true);
        assert_eq!(truncated, initialized, "Truncated vector is invalid");

        let mut popped = RawVector::with_len(97, true);
        for _ in 0..82 {
            popped.pop_bit();
        }
        popped.resize(137, true);
        assert_eq!(popped, initialized, "Popped vector is invalid after extension");
    }

    #[test]
    fn reserving_capacity() {
        let mut original = RawVector::with_len(137, true);
        let copy = original.clone();
        original.reserve(31 + original.capacity() - original.len());

        assert!(original.capacity() >= 137 + 31, "Reserving additional capacity failed");
        assert_eq!(original, copy, "Reserving additional capacity changed the vector");
    }

    #[test]
    fn push_bits() {
        let mut source: Vec<bool> = Vec::new();
        for i in 0..137 {
            source.push(i & 1 == 1);
        }

        let mut v = RawVector::new();
        for i in 0..137 {
            v.push_bit(source[i]);
        }
        assert_eq!(v.len(), 137, "Invalid vector length");

        let mut popped: Vec<bool> = Vec::new();
        while let Some(bit) = v.pop_bit() {
            popped.push(bit);
        }
        popped.reverse();
        assert!(v.is_empty(), "Non-empty vector after popping all bits");
        assert_eq!(popped, source, "Invalid sequence of bits popped from RawVector");
    }

    #[test]
    fn set_bits() {
        let mut v = RawVector::with_len(137, false);
        let mut w = RawVector::with_len(137, true);
        assert_eq!(v.count_ones(), 0, "Non-zero bits in a zero-initialized vector");
        assert_eq!(w.count_ones(), 137, "Zero bits in an one-initialized vector");
        for i in 0..137 {
            v.set_bit(i, i & 1 == 1);
            w.set_bit(i, i & 1 == 1);
        }
        assert_eq!(v.len(), 137, "Invalid vector length");
        assert_eq!(v.count_ones(), 68, "Invalid number of ones in the vector");

        for i in 0..137 {
            assert_eq!(v.bit(i), i & 1 == 1, "Invalid bit {}", i);
        }
        assert_eq!(v, w, "Fully overwritten vector still depends on the initialization value");

        let complement = v.complement();
        for i in 0..137 {
            assert_eq!(!(complement.bit(i)), v.bit(i), "Invalid bit {} in the complement", i);
        }
    }

    #[test]
    fn push_ints() {
        let mut correct: Vec<(u64, usize)> = Vec::new();
        for i in 0..64 {
            correct.push((i, 63)); correct.push((i * (i + 1), 64));
        }

        let mut v = RawVector::new();
        for (value, width) in correct.iter() {
            v.push_int(*value, *width);
        }
        assert_eq!(v.len(), 64 * (63 + 64), "Invalid vector length");

        correct.reverse();
        let mut popped: Vec<(u64, usize)> = Vec::new();
        for i in 0..correct.len() {
            let width = correct[i].1;
            if let Some(value) = v.pop_int(width) {
                popped.push((value, width));
            }
        }
        assert_eq!(popped.len(), correct.len(), "Invalid number of popped ints");
        assert!(v.is_empty(), "Non-empty vector after popping all ints");
        assert_eq!(popped, correct, "Invalid popped ints");
    }

    #[test]
    fn set_ints() {
        let mut v = RawVector::with_len(64 * (63 + 64), false);
        let mut w = RawVector::with_len(64 * (63 + 64), true);
        let mut bit_offset = 0;
        for i in 0..64 {
            v.set_int(bit_offset, i, 63); w.set_int(bit_offset, i, 63); bit_offset += 63;
            v.set_int(bit_offset, i * (i + 1), 64); w.set_int(bit_offset, i * (i + 1), 64); bit_offset += 64;
        }
        assert_eq!(v.len(), 64 * (63 + 64), "Invalid vector length");

        bit_offset = 0;
        for i in 0..64 {
            assert_eq!(v.int(bit_offset, 63), i, "Invalid integer [{}].0", i); bit_offset += 63;
            assert_eq!(v.int(bit_offset, 64), i * (i + 1), "Invalid integer [{}].1", i); bit_offset += 64;
        }
        assert_eq!(v, w, "Fully overwritten vector still depends on the initialization value");
    }

    #[test]
    fn get_words() {
        let correct: Vec<u64> = vec![0x123456, 0x789ABC, 0xFEDCBA, 0x987654];
        let mut v = RawVector::with_len(correct.len() * 64, false);
        for (index, value) in correct.iter().enumerate() {
            v.set_int(index * 64, *value, 64);
        }
        for (index, value) in correct.iter().enumerate() {
            assert_eq!(v.word(index), *value, "Invalid integer {}", index);
        }

        unsafe {
            for (index, value) in correct.iter().enumerate() {
                assert_eq!(v.word_unchecked(index), *value, "Invalid integer {} (unchecked)", index);
            }
        }
    }

    #[test]
    fn serialize() {
        let mut original = RawVector::new();
        for i in 0..64 {
            original.push_int(i * (i + 1) * (i + 2), 16);
        }
        assert_eq!(original.size_in_bytes(), 144, "Invalid RawVector size in bytes");

        let filename = serialize::temp_file_name("raw-vector");
        serialize::serialize_to(&original, &filename).unwrap();

        let copy: RawVector = serialize::load_from(&filename).unwrap();
        assert_eq!(copy, original, "Serialization changed the RawVector");

        fs::remove_file(&filename).unwrap();
    }

    #[test]
    fn empty_writer() {
        let first = serialize::temp_file_name("empty-raw-vector-writer");
        let second = serialize::temp_file_name("empty-raw-vector-writer");

        let mut v = RawVectorWriter::new(&first).unwrap();
        assert!(v.is_empty(), "Created a non-empty empty writer");
        assert_eq!(v.len(), 0, "Nonzero length for an empty writer");
        assert!(v.is_open(), "Newly created writer is not open");
        v.close().unwrap();

        let mut w = RawVectorWriter::with_buf_len(&second, 1024).unwrap();
        assert!(w.is_empty(), "Created a non-empty empty writer with custom buffer size");
        assert!(w.is_open(), "Newly created writer is not open with custom buffer size");
        w.close().unwrap();

        fs::remove_file(&first).unwrap();
        fs::remove_file(&second).unwrap();
    }

    #[test]
    fn push_bits_to_writer() {
        let filename = serialize::temp_file_name("push-bits-to-raw-vector-writer");

        let mut correct: Vec<bool> = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..3523 {
            correct.push(rng.gen());
        }

        let mut v = RawVectorWriter::with_buf_len(&filename, 1024).unwrap();
        for bit in correct.iter() {
            v.push_bit(*bit);
        }
        assert_eq!(v.len(), correct.len(), "Invalid size for the writer after push_bit");
        v.close().unwrap();
        assert!(!v.is_open(), "Could not close the writer");

        let w: RawVector = serialize::load_from(&filename).unwrap();
        assert_eq!(w.len(), correct.len(), "Invalid size for the loaded vector");
        for i in 0..correct.len() {
            assert_eq!(w.bit(i), correct[i], "Invalid bit {}", i);
        }

        fs::remove_file(&filename).unwrap();
    }

    #[test]
    fn push_ints_to_writer() {
        let filename = serialize::temp_file_name("push-ints-to-raw-vector-writer");

        let mut correct: Vec<u64> = Vec::new();
        let mut rng = rand::thread_rng();
        let width = 31;
        for _ in 0..71 {
            let value: u64 = rng.gen();
            correct.push(value & bits::low_set(width));
        }

        let mut v = RawVectorWriter::with_buf_len(&filename, 1024).unwrap();
        for value in correct.iter() {
            v.push_int(*value, width);
        }
        assert_eq!(v.len(), correct.len() * width, "Invalid size for the writer");
        v.close().unwrap();
        assert!(!v.is_open(), "Could not close the writer");

        let w: RawVector = serialize::load_from(&filename).unwrap();
        assert_eq!(w.len(), correct.len() * width, "Invalid size for the loaded vector");
        for i in 0..correct.len() {
            assert_eq!(w.int(i * width, width), correct[i], "Invalid integer {}", i);
        }

        fs::remove_file(&filename).unwrap();
    }

    #[test]
    #[ignore]
    fn large_writer() {
        let filename = serialize::temp_file_name("large-raw-vector-writer");

        let mut correct: Vec<u64> = Vec::new();
        let mut rng = rand::thread_rng();
        let width = 31;
        for _ in 0..620001 {
            let value: u64 = rng.gen();
            correct.push(value & bits::low_set(width));
        }

        let mut v = RawVectorWriter::new(&filename).unwrap();
        for value in correct.iter() {
            v.push_int(*value, width);
        }
        assert_eq!(v.len(), correct.len() * width, "Invalid size for the writer");
        v.close().unwrap();
        assert!(!v.is_open(), "Could not close the writer");

        let w: RawVector = serialize::load_from(&filename).unwrap();
        assert_eq!(w.len(), correct.len() * width, "Invalid size for the loaded vector");
        for i in 0..correct.len() {
            assert_eq!(w.int(i * width, width), correct[i], "Invalid integer {}", i);
        }

        fs::remove_file(&filename).unwrap();
    }
}

//-----------------------------------------------------------------------------
