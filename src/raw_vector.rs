//! The basic vector implementing the low-level functionality used by other vectors in the crate.

use crate::serialize::Serialize;
use crate::bits;

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom};
use std::iter::FromIterator;
use std::path::Path;
use std::io;

//-----------------------------------------------------------------------------

/// A contiguous growable array of bits and up to 64-bit integers based on [`Vec`] of `u64` values.
///
/// There are no iterators over the vector, because it may contain items of varying widths.
///
/// # Notes
/// * The unused part of the last integer is always set to `0`.
/// * The underlying vector may allocate but not use more integers than are strictly needed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawVector {
    bit_len: usize,
    data: Vec<u64>,
}

impl RawVector {
    /// Returns the length of the vector in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let v = RawVector::with_len(137, false);
    /// assert_eq!(v.len(), 137);
    /// ```
    pub fn len(&self) -> usize {
        self.bit_len
    }

    /// Returns the capacity of the vector in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let v = RawVector::with_capacity(137);
    /// assert!(v.capacity() >= 137);
    /// ```
    pub fn capacity(&self) -> usize {
        bits::words_to_bits(self.data.capacity())
    }

    /// Creates an empty vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let v = RawVector::new();
    /// assert_eq!(v.len(), 0);
    /// ```
    pub fn new() -> RawVector {
        RawVector {
            bit_len: 0,
            data: Vec::new(),
        }
    }

    /// Creates an initialized vector of specified length.
    ///
    /// # Arguments
    ///
    /// * `len`: Length of the vector in bits.
    /// * `bit`: Initialization value.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let v = RawVector::with_len(137, false);
    /// assert_eq!(v.len(), 137);
    /// ```
    pub fn with_len(len: usize, bit: bool) -> RawVector {
        let val = bits::filler_value(bit);
        let mut data: Vec<u64> = vec![val; bits::bits_to_words(len)];
        Self::set_unused_bits(&mut data, len, false);

        RawVector {
            bit_len: len,
            data: data,
        }
    }

    /// Creates a vector with enough capacity for at least `n` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let v = RawVector::with_capacity(137);
    /// assert!(v.capacity() >= 137);
    /// ```
    pub fn with_capacity(n: usize) -> RawVector {
        RawVector {
            bit_len: 0,
            data: Vec::with_capacity(bits::bits_to_words(n)),
        }
    }

    /// Resizes the vector to a specified length.
    ///
    /// If `new_len > self.len()`, the new `new_len - self.len()` will be initialized.
    /// If `new_len < self.len()`, the vector is truncated.
    ///
    /// # Arguments
    ///
    /// * `new_len`: New length of the vector in bits.
    /// * `bit`: Initialization value.
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
    pub fn resize(&mut self, new_len: usize, bit: bool) {

        // Fill the unused bits if necessary.
        if new_len > self.len() {
            let old_len = self.len();
            Self::set_unused_bits(&mut self.data, old_len, bit);
        }

        // Use more space if necessary.
        self.data.resize(bits::bits_to_words(new_len), bits::filler_value(bit));
        self.bit_len = new_len;
        Self::set_unused_bits(&mut self.data, new_len, false);
    }

    /// Clear the vector without freeing the data.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::with_len(137, true);
    /// assert_eq!(v.len(), 137);
    /// v.clear();
    /// assert_eq!(v.len(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.data.clear();
        self.bit_len = 0;
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

    /// Appends a bit to the vector.
    ///
    /// Behavior is undefined if `self.len() + 1 > usize::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::new();
    /// v.push_bit(true);
    /// assert_eq!(v.get_bit(0), true);
    /// ```
    pub fn push_bit(&mut self, bit: bool) {
        let (index, offset) = bits::split_offset(self.bit_len);
        if index == self.data.len() {
            self.data.push(0);
        }
        self.data[index] |= (bit as u64) << offset;
        self.bit_len += 1;
    }

    /// Removes and returns the last bit from the vector, or returns `None` if the vector is empty.
    ///
    /// # Examples
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::new();
    /// v.push_bit(true);
    /// assert_eq!(v.pop_bit(), Some(true));
    /// ```
    pub fn pop_bit(&mut self) -> Option<bool> {
        if self.len() > 0 {
            let result = self.get_bit(self.bit_len - 1);
            self.bit_len -= 1;
            self.data.resize(bits::bits_to_words(self.len()), 0); // Avoid using unnecessary words.
            let len_copy = self.len();
            Self::set_unused_bits(&mut self.data, len_copy, false);
            Some(result)
        } else {
            None
        }
    }

    /// Sets a bit in the vector.
    ///
    /// Behavior is undefined if `bit_offset >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::with_len(137, false);
    /// v.set_bit(66, true);
    /// assert_eq!(v.get_bit(66), true);
    /// ```
    pub fn set_bit(&mut self, bit_offset: usize, bit: bool) {
        let (index, offset) = bits::split_offset(bit_offset);
        self.data[index] &= !(1u64 << offset);
        self.data[index] |= (bit as u64) << offset;
    }

    /// Reads a bit from the vector.
    ///
    /// Behavior is undefined if `bit_offset >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::new();
    /// v.push_bit(true);
    /// assert_eq!(v.get_bit(0), true);
    /// ```
    pub fn get_bit(&self, bit_offset: usize) -> bool {
        let (index, offset) = bits::split_offset(bit_offset);
        (self.data[index] & (1u64 << offset)) != 0
    }

    /// Appends an integer to the vector.
    ///
    /// Behavior is undefined if `bits > 64` or `self.len() + bits > usize::MAX`.
    ///
    /// # Arguments
    ///
    /// * `value`: The integer to be appended.
    /// * `bits`: The width of the value in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::new();
    /// v.push_int(123, 8);
    /// assert_eq!(v.get_int(0, 8), 123);
    /// ```
    pub fn push_int(&mut self, value: u64, bits: usize) {
        if self.bit_len + bits > bits::words_to_bits(self.data.len()) {
            self.data.push(0);
        }
        bits::write_int(&mut self.data, self.bit_len, value, bits);
        self.bit_len += bits;
    }

    /// Removes and returns the last `bits` bits from the vector, or returns `None` if there are not enough bits.
    ///
    /// Behavior is undefined if `bits > 64`.
    ///
    /// # Examples
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::new();
    /// v.push_int(123, 8);
    /// assert_eq!(v.pop_int(8), Some(123));
    /// ```
    pub fn pop_int(&mut self, bits: usize) -> Option<u64> {
        if self.bit_len >= bits {
            let result = self.get_int(self.bit_len - bits, bits);
            self.bit_len -= bits;
            self.data.resize(bits::bits_to_words(self.len()), 0); // Avoid using unnecessary words.
            let len_copy = self.len();
            Self::set_unused_bits(&mut self.data, len_copy, false);
            Some(result)
        } else {
            None
        }
    }

    /// Writes an integer to the vector.
    ///
    /// Behavior is undefined if `bits > 64` or `bit_offset + bits > self.len()`.
    ///
    /// # Arguments
    ///
    /// * `bit_offset`: Starting offset in the bit array.
    /// * `value`: The integer to be written.
    /// * `bits`: The width of the integer in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::with_len(137, false);
    /// v.set_int(66, 123, 8);
    /// assert_eq!(v.get_int(66, 8), 123);
    /// ```
    pub fn set_int(&mut self, bit_offset: usize, value: u64, bits: usize) {
        bits::write_int(&mut self.data, bit_offset, value, bits);
    }

    /// Reads an integer from the vector.
    ///
    /// Behavior is undefined if `bits > 64` or `bit_offset + bits > self.len()`.
    ///
    /// # Arguments
    ///
    /// * `bit_offset`: Starting offset in the bit array.
    /// * `bits`: The width of the integer in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVector;
    ///
    /// let mut v = RawVector::new();
    /// v.push_int(123, 8);
    /// assert_eq!(v.get_int(0, 8), 123);
    /// ```
    pub fn get_int(&self, bit_offset: usize, bits: usize) -> u64 {
        bits::read_int(&self.data, bit_offset, bits)
    }

    // Set the unused bits in the last integer to the specified value.
    fn set_unused_bits(data: &mut Vec<u64>, bit_len: usize, bit: bool) {
        let (index, bits) = bits::split_offset(bit_len);
        if bits > 0 {
            if bit {
                data[index] |= !bits::low_set(bits);
            }
            else {
                data[index] &= bits::low_set(bits);
            }
        }
    }
}

//-----------------------------------------------------------------------------

impl Serialize for RawVector {
    // The header also contains the header of the data, because the values are
    // related and depend on the values inserted into the vector. This makes
    // it easier for the writer to overwrite the header fields when closing
    // the file. 
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.bit_len.serialize(writer)?;
        self.data.serialize_header(writer)?;
        Ok(())
    }

    fn serialize_data<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.data.serialize_data(writer)?;
        Ok(())
    }

    fn load<T: io::Read>(&mut self, reader: &mut T) -> io::Result<()> {
        self.bit_len.load(reader)?;
        self.data.load(reader)?;
        Ok(())
    }

    fn size_in_bytes(&self) -> usize {
        self.bit_len.size_in_bytes() + self.data.size_in_bytes()
    }  
}

impl FromIterator<bool> for RawVector {
    fn from_iter<I: IntoIterator<Item=bool>>(iter: I) -> Self {
        let mut result = RawVector::new();
        for bit in iter {
            result.push_bit(bit);
        }
        result
    }
}

impl FromIterator<(u64, usize)> for RawVector {
    fn from_iter<I: IntoIterator<Item=(u64, usize)>>(iter: I) -> Self {
        let mut result = RawVector::new();
        for (value, bits) in iter {
            result.push_int(value, bits);
        }
        result
    }
}

//-----------------------------------------------------------------------------

/// A buffered file writer compatible with the serialization format of [`RawVector`].
///
/// When the writer goes out of scope, the internal buffer is flushed, the file is closed, and all errors are ignored.
/// Call [`RawVectorWriter::close`] explicitly to handle the errors.
#[derive(Debug)]
pub struct RawVectorWriter {
    bit_len: usize,
    buf_len: usize,
    file: Option<File>,
    buf: RawVector,
}

impl RawVectorWriter {
    /// Default buffer size in bits.
    pub const DEFAULT_BUFFER_SIZE: usize = 8 * 1024 * 1024;

    /// Returns the length of the vector in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVectorWriter;
    /// use simple_sds::serialize;
    /// use std::{fs, mem};
    ///
    /// let filename = serialize::temp_file_name("raw-vector-writer-len");
    /// let mut v = RawVectorWriter::new(&filename).unwrap();
    /// assert_eq!(v.len(), 0);
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    pub fn len(&self) -> usize {
        self.bit_len
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
    /// assert_eq!(v.len(), 0);
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Any errors from [`OpenOptions::open`], [`File::seek`], and [`Serialize::serialize`] will be passed through.
    pub fn new<P: AsRef<Path>>(filename: P) -> io::Result<RawVectorWriter> {
        let mut options = OpenOptions::new();
        let file = options.create(true).write(true).truncate(true).open(filename)?;
        // Allocate one extra word for overflow.
        let buf = RawVector::with_capacity(Self::DEFAULT_BUFFER_SIZE + bits::WORD_BITS);
        let mut result = RawVectorWriter {
            bit_len: 0,
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
    /// assert_eq!(v.len(), 0);
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Any errors from [`OpenOptions::open`], [`File::seek`], and [`Serialize::serialize`] will be passed through.
    pub fn with_buf_len<P: AsRef<Path>>(filename: P, buf_len: usize) -> io::Result<RawVectorWriter> {
        let buf_len = bits::round_up_to_word_size(buf_len);
        let mut options = OpenOptions::new();
        let file = options.create(true).write(true).truncate(true).open(filename)?;
        // Allocate one extra word for overflow.
        let buf = RawVector::with_capacity(buf_len + bits::WORD_BITS);
        let mut result = RawVectorWriter {
            bit_len: 0,
            buf_len: buf_len,
            file: Some(file),
            buf: buf,
        };
        result.write_header()?;
        Ok(result)
    }

    /// Tells whether the file is open for writing.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVectorWriter;
    /// use simple_sds::serialize;
    /// use std::{fs, mem};
    ///
    /// let filename = serialize::temp_file_name("raw-vector-writer-is-open");
    /// let mut v = RawVectorWriter::new(&filename).unwrap();
    /// assert!(v.is_open());
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    pub fn is_open(&self) -> bool {
        match self.file {
            Some(_) => true,
            None => false,
        }
    }

    /// Flushes the buffer and closes the file.
    ///
    /// No effect if the file is closed.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVectorWriter;
    /// use simple_sds::serialize;
    /// use std::fs;
    ///
    /// let filename = serialize::temp_file_name("raw-vector-writer-close");
    /// let mut v = RawVectorWriter::new(&filename).unwrap();
    /// assert!(v.is_open());
    /// v.close();
    /// assert!(!v.is_open());
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Any errors from [`File::seek`] and [`Serialize::serialize`] will be passed through.
    pub fn close(&mut self) -> io::Result<()> {
        self.flush(true)?;
        self.write_header()?;
        self.file = None;
        Ok(())
    }

    /// Appends a bit to the vector
    ///
    /// Behavior is undefined if the file is closed.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVectorWriter;
    /// use simple_sds::serialize;
    /// use std::{fs, mem};
    ///
    /// let filename = serialize::temp_file_name("raw-vector-writer-push-bit");
    /// let mut v = RawVectorWriter::new(&filename).unwrap();
    /// assert_eq!(v.len(), 0);
    /// v.push_bit(true).unwrap();
    /// v.push_bit(false).unwrap();
    /// assert_eq!(v.len(), 2);
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Any errors from [`Serialize::serialize`] will be passed through.
    pub fn push_bit(&mut self, bit: bool) -> io::Result<()> {
        self.buf.push_bit(bit); self.bit_len += 1;
        if self.buf.len() >= self.buf_len {
            self.flush(false)?;
        }
        Ok(())
    }

    /// Appends an integer to the vector.
    ///
    /// Behavior is undefined if `bits > 64`, if `self.len() + bits > usize::MAX`, or if the file is closed.
    ///
    /// # Arguments
    ///
    /// * `value`: The integer to be appended.
    /// * `bits`: The width of the value in bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use simple_sds::raw_vector::RawVectorWriter;
    /// use simple_sds::serialize;
    /// use std::{fs, mem};
    ///
    /// let filename = serialize::temp_file_name("raw-vector-writer-push-int");
    /// let mut v = RawVectorWriter::new(&filename).unwrap();
    /// assert_eq!(v.len(), 0);
    /// v.push_int(123, 8).unwrap();
    /// v.push_int(456789, 20).unwrap();
    /// assert_eq!(v.len(), 28);
    /// mem::drop(v);
    /// fs::remove_file(&filename).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Any errors from [`Serialize::serialize`] will be passed through.
    pub fn push_int(&mut self, value: u64, bits: usize) -> io::Result<()> {
        self.buf.push_int(value, bits); self.bit_len += bits;
        if self.buf.len() >= self.buf_len {
            self.flush(false)?;
        }
        Ok(())
    }

    // Flushes the the buffer and optionally also the overflow bits.
    // This should only be called when the buffer is full or the file is going to be closed.
    // Otherwise there may be unused bits in the last serialized word.
    fn flush(&mut self, with_overflow: bool) -> io::Result<()> {
        if !self.is_open() || self.buf.len() == 0 {
            return Ok(());
        }

        // Handle the overflow if not serializing the entire buffer.
        let mut overflow: (u64, usize) = (0, 0);
        if !with_overflow && self.buf.len() > self.buf_len {
            overflow = (self.buf.get_int(self.buf_len, self.buf.len() - self.buf_len), self.buf.len() - self.buf_len);
            self.buf.resize(self.buf_len, false);
        }

        // Serialize and clear the buffer.
        let f = self.file.as_mut().unwrap();
        self.buf.serialize_data(f)?;
        self.buf.clear();

        // Push the overflow back to the buffer.
        if !with_overflow && overflow.1 > 0 {
            self.buf.push_int(overflow.0, overflow.1);
        }

        Ok(())
    }

    // Rewinds the file and writes the header.
    fn write_header(&mut self) -> io::Result<()> {
        if !self.is_open() {
            return Ok(());
        }
        let f = self.file.as_mut().unwrap();
        f.seek(SeekFrom::Start(0))?;
        self.bit_len.serialize(f)?;
        let words: usize = bits::bits_to_words(self.bit_len);
        words.serialize(f)?;
        Ok(())
    }
}

//-----------------------------------------------------------------------------

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
    fn empty_raw_vector() {
        let empty = RawVector::new();
        assert_eq!(empty.len(), 0, "Created a non-empty empty vector");
        assert_eq!(empty.capacity(), 0, "Reserved unnecessary memory for an empty vector");

        let with_capacity = RawVector::with_capacity(137);
        assert_eq!(with_capacity.len(), 0, "Created a non-empty vector by specifying capacity");
        assert!(with_capacity.capacity() >= 137, "Vector capacity is lower than specified");
    }

    #[test]
    fn with_len_and_clear() {
        let mut v = RawVector::with_len(137, true);
        assert_eq!(v.len(), 137, "Vector length is not as specified");
        v.clear();
        assert_eq!(v.len(), 0, "Could not clear the vector");
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
        let copied = original.clone();
        original.reserve(31 + original.capacity() - original.len());

        assert!(original.capacity() >= 137 + 31, "Reserving additional capacity failed");
        assert_eq!(original, copied, "Reserving additional capacity changed the vector");
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

        let from_iter: RawVector = source.clone().into_iter().collect();
        assert_eq!(from_iter, v, "Vector built from an iterator is invalid");

        let mut popped: Vec<bool> = Vec::new();
        while let Some(bit) = v.pop_bit() {
            popped.push(bit);
        }
        popped.reverse();
        assert_eq!(v.len(), 0, "Non-empty vector after popping all bits");
        assert_eq!(popped, source, "Invalid sequence of bits popped from RawVector");
    }

    #[test]
    fn set_bits() {
        let mut v = RawVector::with_len(137, false);
        let mut w = RawVector::with_len(137, true);
        for i in 0..137 {
            v.set_bit(i, i & 1 == 1);
            w.set_bit(i, i & 1 == 1);
        }
        assert_eq!(v.len(), 137, "Invalid vector length");

        for i in 0..137 {
            assert_eq!(v.get_bit(i), i & 1 == 1, "Invalid bit {}", i);
        }
        assert_eq!(v, w, "Fully overwritten vector still depends on the initialization value");
    }

    #[test]
    fn push_ints() {
        let mut correct: Vec<(u64, usize)> = Vec::new();
        for i in 0..64 {
            correct.push((i, 63)); correct.push((i * (i + 1), 64));
        }

        let mut v = RawVector::new();
        for (value, bits) in correct.iter() {
            v.push_int(*value, *bits);
        }
        assert_eq!(v.len(), 64 * (63 + 64), "Invalid vector length");

        let from_iter: RawVector = correct.clone().into_iter().collect();
        assert_eq!(from_iter, v, "Vector built from an iterator is invalid");

        correct.reverse();
        let mut popped: Vec<(u64, usize)> = Vec::new();
        for i in 0..correct.len() {
            let bits = correct[i].1;
            if let Some(value) = v.pop_int(bits) {
                popped.push((value, bits));
            }
        }
        assert_eq!(popped.len(), correct.len(), "Invalid number of popped ints");
        assert_eq!(v.len(), 0, "Non-empty vector after popping all ints");
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
            assert_eq!(v.get_int(bit_offset, 63), i, "Invalid integer [{}].0", i); bit_offset += 63;
            assert_eq!(v.get_int(bit_offset, 64), i * (i + 1), "Invalid integer [{}].1", i); bit_offset += 64;
        }
        assert_eq!(v, w, "Fully overwritten vector still depends on the initialization value");
    }

    #[test]
    fn serialize_raw_vector() {
        let mut original = RawVector::new();
        for i in 0..64 {
            original.push_int(i * (i + 1) * (i + 2), 16);
        }
        assert_eq!(original.size_in_bytes(), 144, "Invalid RawVector size in bytes");

        let filename = serialize::temp_file_name("raw-vector");
        serialize::serialize_to(&original, &filename).unwrap();

        let mut copy = RawVector::new();
        serialize::load_from(&mut copy, &filename).unwrap();
        assert_eq!(copy, original, "Serialization changed the RawVector");

        fs::remove_file(&filename).unwrap();
    }

    // len(), new(), with_buf_len()
    // is_open(), close()
    // push_bit(), push_int()

    #[test]
    fn empty_writer() {
        let first = serialize::temp_file_name("empty-writer");
        let second = serialize::temp_file_name("empty-writer");

        let v = RawVectorWriter::new(&first).unwrap();
        assert_eq!(v.len(), 0, "Created a non-empty empty writer");
        assert!(v.is_open(), "Newly created writer is not open");

        let w = RawVectorWriter::with_buf_len(&second, 1024).unwrap();
        assert_eq!(w.len(), 0, "Created a non-empty empty writer with custom buffer size");
        assert!(w.is_open(), "Newly created writer is not open with custom buffer size");

        fs::remove_file(&first).unwrap();
        fs::remove_file(&second).unwrap();
    }

    #[test]
    fn push_bits_to_writer() {
        let filename = serialize::temp_file_name("push-bits-to-writer");

        let mut correct: Vec<bool> = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..3523 {
            correct.push(rng.gen());
        }

        let mut v = RawVectorWriter::with_buf_len(&filename, 1024).unwrap();
        for bit in correct.iter() {
            v.push_bit(*bit).unwrap();
        }
        assert_eq!(v.len(), correct.len(), "Invalid size for the writer");
        v.close().unwrap();
        assert!(!v.is_open(), "Could not close the writer");

        let mut w = RawVector::new();
        serialize::load_from(&mut w, &filename).unwrap();
        assert_eq!(w.len(), correct.len(), "Invalid size for the loaded vector");
        for i in 0..correct.len() {
            assert_eq!(w.get_bit(i), correct[i], "Invalid bit {}", i);
        }

        fs::remove_file(&filename).unwrap();
    }

    #[test]
    fn push_ints_to_writer() {
        let filename = serialize::temp_file_name("push-ints-to-writer");

        let mut correct: Vec<u64> = Vec::new();
        let mut rng = rand::thread_rng();
        let bits = 31;
        for _ in 0..71 {
            let value: u64 = rng.gen();
            correct.push(value & bits::low_set(bits));
        }

        let mut v = RawVectorWriter::with_buf_len(&filename, 1024).unwrap();
        for value in correct.iter() {
            v.push_int(*value, bits).unwrap();
        }
        assert_eq!(v.len(), correct.len() * bits, "Invalid size for the writer");
        v.close().unwrap();
        assert!(!v.is_open(), "Could not close the writer");

        let mut w = RawVector::new();
        serialize::load_from(&mut w, &filename).unwrap();
        assert_eq!(w.len(), correct.len() * bits, "Invalid size for the loaded vector");
        for i in 0..correct.len() {
            assert_eq!(w.get_int(i * bits, bits), correct[i], "Invalid integer {}", i);
        }

        fs::remove_file(&filename).unwrap();
    }

    #[test]
    #[ignore]
    fn large_writer() {
        let filename = serialize::temp_file_name("large_writer");

        let mut correct: Vec<u64> = Vec::new();
        let mut rng = rand::thread_rng();
        let bits = 31;
        for _ in 0..620001 {
            let value: u64 = rng.gen();
            correct.push(value & bits::low_set(bits));
        }

        let mut v = RawVectorWriter::new(&filename).unwrap();
        for value in correct.iter() {
            v.push_int(*value, bits).unwrap();
        }
        assert_eq!(v.len(), correct.len() * bits, "Invalid size for the writer");
        v.close().unwrap();
        assert!(!v.is_open(), "Could not close the writer");

        let mut w = RawVector::new();
        serialize::load_from(&mut w, &filename).unwrap();
        assert_eq!(w.len(), correct.len() * bits, "Invalid size for the loaded vector");
        for i in 0..correct.len() {
            assert_eq!(w.get_int(i * bits, bits), correct[i], "Invalid integer {}", i);
        }

        fs::remove_file(&filename).unwrap();
    }
}

//-----------------------------------------------------------------------------
