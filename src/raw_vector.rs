//! The basic vector implementing the low-level functionality used by other vectors in the crate.

use crate::serialize::{MappedSlice, MemoryMap, MemoryMapped, Serialize};
use crate::bits;

use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::{cmp, io};

#[cfg(test)]
mod tests;

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
///     unsafe fn int(&self, bit_offset: usize, width: usize) -> u64 {
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
///     unsafe fn set_int(&mut self, bit_offset: usize, value: u64, width: usize) {
///         bits::write_int(&mut self.0, bit_offset, value, width);
///     }
/// }
///
/// let mut example = Example(vec![0u64; 2]);
/// assert!(example.is_mutable());
///
/// unsafe {
///    example.set_int(4, 0x33, 8);
///    example.set_int(63, 2, 2);
/// }
/// example.set_bit(72, true);
/// assert_eq!(example.0[0], 0x330);
/// assert_eq!(example.0[1], 0x101);
///
/// assert!(example.bit(72));
/// assert!(!example.bit(68));
/// unsafe {
///     assert_eq!(example.int(4, 8), 0x33);
///     assert_eq!(example.int(63, 2), 2);
/// }
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
    /// # Arguments
    ///
    /// * `bit_offset`: Starting offset in the bit array.
    /// * `width`: The width of the integer in bits.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if `width > 64`.
    ///
    /// # Panics
    ///
    /// May panic if `bit_offset + width - 1` is not a valid offset in the bit array.
    /// May panic from I/O errors.
    unsafe fn int(&self, bit_offset: usize, width: usize) -> u64;

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
    /// # Safety
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
    /// # Arguments
    ///
    /// * `bit_offset`: Starting offset in the bit array.
    /// * `value`: The integer to be written.
    /// * `width`: The width of the integer in bits.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if `width > 64`.
    ///
    /// # Panics
    ///
    /// May panic if `bit_offset + width - 1` is not a valid offset in the bit array.
    /// May panic if the underlying data is not mutable.
    /// May panic from I/O errors.
    unsafe fn set_int(&mut self, bit_offset: usize, value: u64, width: usize);
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
///     unsafe fn push_int(&mut self, value: u64, width: usize) {
///         self.1.push(value & bits::low_set(width));
///     }
/// }
///
/// let mut example = Example::new();
/// example.push_bit(false);
/// unsafe {
///     example.push_int(123, 8);
///     example.push_int(456, 9);
/// }
/// example.push_bit(true);
///
/// assert_eq!(example.0.len(), 2);
/// assert_eq!(example.1.len(), 2);
/// ```
pub trait PushRaw {
    /// Appends a bit to the container.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// May panic if there is an integer overflow.
    fn push_bit(&mut self, value: bool);

    /// Appends an integer to the container.
    ///
    /// # Arguments
    ///
    /// * `value`: The integer to be appended.
    /// * `width`: The width of the integer in bits.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if `width > 64`.
    ///
    /// # Panics
    ///
    /// May panic from I/O errors.
    /// May panic if there is an integer overflow.
    unsafe fn push_int(&mut self, value: u64, width: usize);
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
///     unsafe fn pop_int(&mut self, _: usize) -> Option<u64> {
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
/// unsafe {
///     assert_eq!(example.pop_int(9).unwrap(), 456);
///     assert_eq!(example.pop_int(8).unwrap(), 123);
/// }
/// assert_eq!(example.pop_bit().unwrap(), false);
/// assert_eq!(example.pop_bit(), None);
/// unsafe { assert_eq!(example.pop_int(1), None); }
/// ```
pub trait PopRaw {
    /// Removes and returns the last bit from the container.
    ///
    /// Returns [`None`] the container does not have more bits.
    fn pop_bit(&mut self) -> Option<bool>;

    /// Removes and returns the last `width` bits from the container as an integer.
    ///
    /// Returns [`None`] if the container does not have more integers of that width.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if `width > 64`.
    unsafe fn pop_int(&mut self, width: usize) -> Option<u64>;
}

//-----------------------------------------------------------------------------

/// A contiguous growable array of bits and up to 64-bit integers based on [`Vec`] of [`u64`] values.
///
/// There are no iterators over the vector, because it may contain items of varying widths.
///
/// # Notes
///
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
            len, data,
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

    /// Returns the size of a serialized vector with the given capacity in [`u64`] elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// assert_eq!(RawVector::size_by_params(247), 6);
    /// ```
    pub fn size_by_params(capacity: usize) -> usize {
        2 + bits::bits_to_words(capacity)
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
    /// unsafe { original.set_int(95, 456, 9); } original.set_bit(123, true);
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
    ///
    /// # Panics
    ///
    /// May panic if `self.len() + additional + 63 > usize::MAX`.
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
    unsafe fn int(&self, bit_offset: usize, width: usize) -> u64 {
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
    unsafe fn set_int(&mut self, bit_offset: usize, value: u64, width: usize) {
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

    unsafe fn push_int(&mut self, value: u64, width: usize) {
        if self.len + width > bits::words_to_bits(self.data.len()) {
            self.data.push(0);
        }
        bits::write_int(&mut self.data, self.len, value, width);
        self.len += width;
    }
}

impl PopRaw for RawVector {
    fn pop_bit(&mut self) -> Option<bool> {
        if !self.is_empty() {
            let result = self.bit(self.len - 1);
            self.len -= 1;
            self.data.resize(bits::bits_to_words(self.len()), 0); // Avoid using unnecessary words.
            self.set_unused_bits(false);
            Some(result)
        } else {
            None
        }
    }

    unsafe fn pop_int(&mut self, width: usize) -> Option<u64> {
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
        if bits::bits_to_words(len) != data.len() {
            Err(Error::new(ErrorKind::InvalidData, "Bit length / word length mismatch"))
        } else {
            Ok(RawVector {
                len, data,
            })
        }
    }

    fn size_in_elements(&self) -> usize {
        self.len.size_in_elements() + self.data.size_in_elements()
    }
}

//-----------------------------------------------------------------------------

impl AsRef<[u64]> for RawVector {
    #[inline]
    fn as_ref(&self) -> &[u64] {
        self.data.as_ref()
    }
}

//-----------------------------------------------------------------------------

/// A buffered file writer compatible with the serialization format of [`RawVector`].
///
/// When the writer goes out of scope, the internal buffer is flushed, the file is closed, and all errors are ignored.
/// Call [`RawVectorWriter::close`] explicitly to handle the errors.
///
/// # Examples
///
/// ```
/// use simple_sds::raw_vector::{RawVector, RawVectorWriter, AccessRaw, PushRaw};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let filename = serialize::temp_file_name("raw-vector-writer");
/// let width = 29;
/// let mut header: Vec<u64> = Vec::new();
/// let mut writer = RawVectorWriter::new(&filename, &mut header).unwrap();
/// unsafe {
///     writer.push_int(123, width);
///     writer.push_int(456, width);
///     writer.push_int(789, width);
/// }
/// writer.close();
///
/// let v: RawVector = serialize::load_from(&filename).unwrap();
/// assert_eq!(v.len(), 3 * width);
/// unsafe {
///     assert_eq!(v.int(0, width), 123);
///     assert_eq!(v.int(width, width), 456);
///     assert_eq!(v.int(2 * width, width), 789);
/// }
///
/// fs::remove_file(&filename);
/// ```
#[derive(Debug)]
pub struct RawVectorWriter {
    len: usize,
    buf_len: usize,
    buf: RawVector,
    file: Option<File>,
    filename: PathBuf,
}

// Ways of flushing a write buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FlushMode {
    // Only flush the part of the buffer that can be flushed safely.
    Safe,
    // Flush the entire buffer.
    // Subsequent writes to the buffer may leave it in an invalid state.
    Final,
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
    /// # Arguments
    ///
    /// * `filename`: Name of the file.
    /// * `header`: Header of the parent structure (may be empty).
    pub fn new<P: AsRef<Path>>(filename: P, header: &mut Vec<u64>) -> io::Result<RawVectorWriter> {
        let mut options = OpenOptions::new();
        let file = options.create(true).write(true).truncate(true).open(&filename)?;
        // Allocate one extra word for overflow.
        let buf = RawVector::with_capacity(Self::DEFAULT_BUFFER_SIZE + bits::WORD_BITS);
        let mut name = PathBuf::new();
        name.push(&filename);
        let mut result = RawVectorWriter {
            len: 0,
            buf_len: Self::DEFAULT_BUFFER_SIZE,
            buf,
            file: Some(file),
            filename: name,
        };
        result.write_header(header)?;
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
    /// * `header`: Header of the parent structure (may be empty).
    /// * `buf_len`: Buffer size in bits.
    pub fn with_buf_len<P: AsRef<Path>>(filename: P, header: &mut Vec<u64>, buf_len: usize) -> io::Result<RawVectorWriter> {
        // Buffer length must be a positive multiple of `bits::WORD_BITS`.
        let buf_len = cmp::max(bits::round_up_to_word_bits(buf_len), bits::WORD_BITS);
        let mut options = OpenOptions::new();
        let file = options.create(true).write(true).truncate(true).open(&filename)?;
        // Allocate one extra word for overflow.
        let buf = RawVector::with_capacity(buf_len + bits::WORD_BITS);
        let mut name = PathBuf::new();
        name.push(&filename);
        let mut result = RawVectorWriter {
            len: 0,
            buf_len,
            buf,
            file: Some(file),
            filename: name,
        };
        result.write_header(header)?;
        Ok(result)
    }

    /// Returns the name of the file.
    pub fn filename(&self) -> &Path {
        self.filename.as_path()
    }

    /// Returns `true` if the file is open for writing.
    pub fn is_open(&self) -> bool {
        self.file.is_some()
    }

    // Flushes the buffer.
    fn flush(&mut self, mode: FlushMode) -> io::Result<()> {
        if let Some(f) = self.file.as_mut() {
            // Handle the overflow if not serializing the entire buffer.
            let mut overflow: (u64, usize) = (0, 0);
            if let FlushMode::Safe = mode {
                if self.buf.len() > self.buf_len {
                    unsafe { overflow = (self.buf.int(self.buf_len, self.buf.len() - self.buf_len), self.buf.len() - self.buf_len); }
                    self.buf.resize(self.buf_len, false);
                }
            }

            // Serialize and clear the buffer.
            self.buf.serialize_body(f)?;
            self.buf.clear();

            // Push the overflow back to the buffer.
            if let FlushMode::Safe = mode {
                if overflow.1 > 0 {
                    unsafe { self.buf.push_int(overflow.0, overflow.1); }
                }
            }
        }
        Ok(())
    }

    // Seeks to the start of the file, appends its own header to `header`, and writes it into the file.
    fn write_header(&mut self, header: &mut Vec<u64>) -> io::Result<()> {
        if let Some(f) = self.file.as_mut() {
            f.seek(SeekFrom::Start(0))?;
            header.push(self.len as u64);
            header.push(bits::bits_to_words(self.len) as u64);
            header.serialize_body(f)?;
        }
        Ok(())
    }

    /// Flushes the buffer, writes the header, and closes the file.
    ///
    /// No effect if the file is closed.
    ///
    /// # Errors
    ///
    /// Any I/O errors will be passed through.
    pub fn close(&mut self) -> io::Result<()> {
        let mut header: Vec<u64> = Vec::new();
        self.close_with_header(&mut header)
    }

    /// Flushes the buffer, writes the header, and closes the file.
    ///
    /// No effect if the file is closed.
    /// This method should only be called by the `close` method of a parent writer.
    ///
    /// # Errors
    ///
    /// Any I/O errors will be passed through.
    pub fn close_with_header(&mut self, header: &mut Vec<u64>) -> io::Result<()> {
        if self.is_open() {
            self.flush(FlushMode::Final)?;
            self.write_header(header)?;
            self.file = None
        }
        Ok(())
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

    unsafe fn push_int(&mut self, value: u64, width: usize) {
        self.buf.push_int(value, width); self.len += width;
        if self.buf.len() >= self.buf_len {
            self.flush(FlushMode::Safe).unwrap();
        }
    }
}

impl Drop for RawVectorWriter {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

//-----------------------------------------------------------------------------

/// An immutable memory-mapped [`RawVector`].
///
/// This is compatible with the serialization format of [`RawVector`].
///
/// # Examples
///
/// ```
/// use simple_sds::raw_vector::{RawVector, RawVectorMapper, AccessRaw, PushRaw};
/// use simple_sds::serialize::{MemoryMap, MemoryMapped, MappingMode};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let filename = serialize::temp_file_name("raw-vector-mapper");
/// let width = 29;
/// let mut original = RawVector::new();
/// unsafe {
///     original.push_int(123, width);
///     original.push_int(456, width);
///     original.push_int(789, width);
/// }
/// serialize::serialize_to(&original, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// let mapper = RawVectorMapper::new(&map, 0).unwrap();
/// assert_eq!(mapper.len(), 3 * width);
/// unsafe {
///     assert_eq!(mapper.int(0, width), 123);
///     assert_eq!(mapper.int(width, width), 456);
///     assert_eq!(mapper.int(2 * width, width), 789);
/// }
///
/// drop(mapper); drop(map);
/// fs::remove_file(&filename);
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct RawVectorMapper<'a> {
    len: usize,
    data: MappedSlice<'a, u64>,
}

impl<'a> RawVectorMapper<'a> {
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

    /// Counts the number of ones in the bit array.
    pub fn count_ones(&self) -> usize {
        let mut result: usize = 0;
        for value in self.data.iter() {
            result += (*value).count_ones() as usize;
        }
        result
    }
}

impl<'a> AccessRaw for RawVectorMapper<'a> {
    #[inline]
    fn bit(&self, bit_offset: usize) -> bool {
        let (index, offset) = bits::split_offset(bit_offset);
        ((self.data[index] >> offset) & 1) == 1
    }

    #[inline]
    unsafe fn int(&self, bit_offset: usize, width: usize) -> u64 {
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
        false
    }

    #[inline]
    fn set_bit(&mut self, _: usize, _: bool) {
        panic!("RawVectorMapper::set_bit(): Not implemented");
    }

    #[inline]
    unsafe fn set_int(&mut self, _: usize, _: u64, _: usize) {
        panic!("RawVectorMapper::set_int(): Not implemented");
    }
}

impl<'a> MemoryMapped<'a> for RawVectorMapper<'a> {
    fn new(map: &'a MemoryMap, offset: usize) -> io::Result<Self> {
        if offset >= map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The starting offset is out of range"));
        }
        let slice: &[u64] = map.as_ref();
        let len = slice[offset] as usize;
        let data = MappedSlice::new(map, offset + 1)?;
        Ok(RawVectorMapper {
            len, data,
        })
    }

    fn map_offset(&self) -> usize {
        self.data.map_offset() - 1
    }

    fn map_len(&self) -> usize {
        self.data.map_len() + 1
    }
}

impl<'a> AsRef<MappedSlice<'a, u64>> for RawVectorMapper<'a> {
    #[inline]
    fn as_ref(&self) -> &MappedSlice<'a, u64> {
        &(self.data)
    }
}

//-----------------------------------------------------------------------------
