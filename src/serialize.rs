//! Simple serialization interface.
//!
//! The serialized representation closely mirrors the in-memory representation with 8-byte alignment.
//! This makes it easy to develop memory-mapped versions of the structures.
//!
//! Note that loading serialized structures is fundamentally unsafe.
//! Some [`Serialize::load`] implementations do simple sanity checks on the headers.
//! However, it is not feasible to validate all loaded data in high-performance code.
//! The behavior of corrupted data structures is always undefined.
//!
//! Function [`test()`] offers a convenient way of testing that the serialization interface works correctly for a custom type.
//!
//! # Serialization formats
//!
//! The serialization format of a structure, as implemented with trait [`Serialize`], is split into the header and the body.
//! Both contain a concatenation of 0 or more structures, and at least one of them must be non-empty.
//! The header and the body can be serialized separately with [`Serialize::serialize_header`] and [`Serialize::serialize_body`].
//! Method [`Serialize::serialize`] provides an easy way of calling both.
//! A serialized structure is always loaded with a single [`Serialize::load`] call.
//!
//! There are currently five basic serialization types:
//!
//! * [`Serializable`]: A fixed-size type that can be serialized as one or more [`u64`] elements.
//!   The header is empty and the body contains the value.
//! * [`Vec`] of a type that implements [`Serializable`].
//!   The header stores the number of items in the vector as [`usize`].
//!   The body stores the items.
//! * [`Vec`] of [`u8`].
//!   The header stores the number of items in the vector as [`usize`].
//!   The body stores the items followed by a padding of `0` bytes to make the size of the body a multiple of 8 bytes.
//! * [`String`] stored as a [`Vec`] of [`u8`] using the UTF-8 encoding.
//! * [`Option`]`<T>` for a type `T` that implements [`Serialize`].
//!   The header stores the number of [`u64`] elements in the body as [`usize`].
//!   The body stores `T` for [`Some`]`(T)` and is empty for [`None`].
//!
//! See also: [https://github.com/jltsiren/simple-sds/blob/main/SERIALIZATION.md](https://github.com/jltsiren/simple-sds/blob/main/SERIALIZATION.md).
//!
//! # Memory-mapped structures
//!
//! [`MemoryMap`] implements a highly unsafe interface of memory mapping files as arrays of [`u64`] elements.
//! The file can be opened for reading and writing ([`MappingMode::Mutable`]) or as read-only ([`MappingMode::ReadOnly`]).
//! While the contents of the file can be changed, the file cannot be resized.
//!
//! A file may contain multiple nested or concatenated structures.
//! Trait [`MemoryMapped`] represents a memory-mapped structure that borrows an interval of the memory map.
//! There are four implementations of [`MemoryMapped`] for basic serialization types:
//!
//! * [`MappedSlice`] matches the serialization format of [`Vec`] of a [`Serializable`] type.
//! * [`MappedBytes`] matches the serialization format of [`Vec`] of [`u8`].
//! * [`MappedStr`] matches the serialization format of [`String`].
//! * [`MappedOption`] matches the serialization format of [`Option`].

use crate::bits;

use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{Error, ErrorKind, Read, Write};
use std::ops::{Deref, Index};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{env, fs, io, marker, mem, process, ptr, slice, str};

#[cfg(test)]
mod tests;

//-----------------------------------------------------------------------------

/// Serialize a data structure.
///
/// `self.size_in_elements()` should always be nonzero.
///
/// A structure that implements `Serialize` may also have an associated function `size_by_params`.
/// The function determines the size of a serialized structure with the given parameters in [`u64`] elements without building the structure.
///
/// # Examples
///
/// ```
/// use simple_sds::serialize::Serialize;
/// use simple_sds::serialize;
/// use std::{fs, io, mem};
///
/// #[derive(PartialEq, Eq, Debug)]
/// struct Example(i32, u32);
///
/// impl Serialize for Example {
///     fn serialize_header<T: io::Write>(&self, _: &mut T) -> io::Result<()> {
///         Ok(())
///     }
///
///     fn serialize_body<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
///         let bytes: [u8; mem::size_of::<Self>()] = unsafe { mem::transmute_copy(self) };
///         writer.write_all(&bytes)?;
///         Ok(())
///     }
///
///     fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
///         let mut bytes = [0u8; mem::size_of::<Self>()];
///         reader.read_exact(&mut bytes)?;
///         let value: Example = unsafe { mem::transmute_copy(&bytes) };
///         Ok(value)
///     }
///
///     fn size_in_elements(&self) -> usize {
///         1
///     }
/// }
///
/// let example = Example(-123, 456);
/// assert_eq!(example.size_in_bytes(), 8);
///
/// let filename = serialize::temp_file_name("serialize");
/// serialize::serialize_to(&example, &filename).unwrap();
///
/// let copy: Example = serialize::load_from(&filename).unwrap();
/// assert_eq!(copy, example);
///
/// fs::remove_file(&filename).unwrap();
/// ```
pub trait Serialize: Sized {
    /// Serializes the struct to the writer.
    ///
    /// Equivalent to calling [`Serialize::serialize_header`] and [`Serialize::serialize_body`].
    ///
    /// # Errors
    ///
    /// Any errors from the writer may be passed through.
    fn serialize<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        self.serialize_header(writer)?;
        self.serialize_body(writer)?;
        Ok(())
    }

    /// Serializes the header to the writer.
    ///
    /// # Errors
    ///
    /// Any errors from the writer may be passed through.
    fn serialize_header<T: Write>(&self, writer: &mut T) -> io::Result<()>;

    /// Serializes the body to the writer.
    ///
    /// # Errors
    ///
    /// Any errors from the writer may be passed through.
    fn serialize_body<T: Write>(&self, writer: &mut T) -> io::Result<()>;

    /// Loads the struct from the reader.
    ///
    /// # Errors
    ///
    /// Any errors from the reader may be passed through.
    /// [`ErrorKind::InvalidData`] should be used to indicate that the data failed sanity checks.
    fn load<T: Read>(reader: &mut T) -> io::Result<Self>;

    /// Returns the size of the serialized struct in [`u64`] elements.
    ///
    /// This is usually closely related to the size of the in-memory struct.
    fn size_in_elements(&self) -> usize;

    /// Returns the size of the serialized struct in bytes.
    ///
    /// This is usually closely related to the size of the in-memory struct.
    fn size_in_bytes(&self) -> usize {
        bits::words_to_bytes(self.size_in_elements())
    }
}

//-----------------------------------------------------------------------------

/// A fixed-size type that can be serialized as one or more [`u64`] elements.
pub trait Serializable: Sized + Default {
    /// Returns the number of elements needed for serializing the type.
    fn elements() -> usize {
        mem::size_of::<Self>() / bits::WORD_BYTES
    }
}

impl Serializable for u64 {}
impl Serializable for usize {}
impl Serializable for (u64, u64) {}

impl<V: Serializable> Serialize for V {
    fn serialize_header<T: Write>(&self, _: &mut T) -> io::Result<()> {
        Ok(())
    }

    fn serialize_body<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        unsafe {
            let buf: &[u8] = slice::from_raw_parts(self as *const Self as *const u8, mem::size_of::<Self>());
            writer.write_all(buf)?;
        }
        Ok(())
    }

    fn load<T: Read>(reader: &mut T) -> io::Result<Self> {
        let mut value = Self::default();
        unsafe {
            let buf: &mut [u8] = slice::from_raw_parts_mut(&mut value as *mut Self as *mut u8, mem::size_of::<Self>());
            reader.read_exact(buf)?;
        }
        Ok(value)
    }

    fn size_in_elements(&self) -> usize {
        Self::elements()
    }
}

impl<V: Serializable> Serialize for Vec<V> {
    fn serialize_header<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        let size = self.len();
        size.serialize(writer)?;
        Ok(())
    }

    fn serialize_body<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        unsafe {
            let buf: &[u8] = slice::from_raw_parts(self.as_ptr() as *const u8, self.len() * mem::size_of::<V>());
            writer.write_all(buf)?;
        }
        Ok(())
    }

    fn load<T: Read>(reader: &mut T) -> io::Result<Self> {
        let size = usize::load(reader)?;
        let mut value: Vec<V> = Vec::with_capacity(size);

        unsafe {
            let buf: &mut [u8] = slice::from_raw_parts_mut(value.as_mut_ptr() as *mut u8, size * mem::size_of::<V>());
            reader.read_exact(buf)?;
            value.set_len(size);
        }

        Ok(value)
    }

    fn size_in_elements(&self) -> usize {
        1 + self.len() * V::elements()
    }
}

impl Serialize for Vec<u8> {
    fn serialize_header<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        let size = self.len();
        size.serialize(writer)?;
        Ok(())
    }

    fn serialize_body<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        writer.write_all(self.as_slice())?;
        let padded_len = bits::round_up_to_word_bytes(self.len());
        if padded_len > self.len() {
            let padding = [0u8; bits::WORD_BYTES];
            writer.write_all(&padding[0..padded_len - self.len()])?;
        }
        Ok(())
    }

    fn load<T: Read>(reader: &mut T) -> io::Result<Self> {
        let size = usize::load(reader)?;
        let mut value: Vec<u8> = vec![0; size];
        reader.read_exact(value.as_mut_slice())?;

        // Skip padding.
        let padded_len = bits::round_up_to_word_bytes(value.len());
        if padded_len > value.len() {
            let mut padding = [0u8; bits::WORD_BYTES];
            reader.read_exact(&mut padding[0..padded_len - value.len()])?;
        }

        Ok(value)
    }

    fn size_in_elements(&self) -> usize {
        1 + bits::bytes_to_words(self.len())
    }
}

impl Serialize for String {
    fn serialize_header<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        let size = self.len();
        size.serialize(writer)?;
        Ok(())
    }

    fn serialize_body<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        writer.write_all(self.as_bytes())?;
        let padded_len = bits::round_up_to_word_bytes(self.len());
        if padded_len > self.len() {
            let padding = [0u8; bits::WORD_BYTES];
            writer.write_all(&padding[0..padded_len - self.len()])?;
        }
        Ok(())
    }

    fn load<T: Read>(reader: &mut T) -> io::Result<Self> {
        let bytes = Vec::<u8>::load(reader)?;
        String::from_utf8(bytes).map_err(|_| Error::new(ErrorKind::InvalidData, "Invalid UTF-8"))
    }

    fn size_in_elements(&self) -> usize {
        1 + bits::bytes_to_words(self.len())
    }
}

impl<V: Serialize> Serialize for Option<V> {
    fn serialize_header<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        let mut size: usize = 0;
        if let Some(value) = self {
            size = value.size_in_elements();
        }
        size.serialize(writer)?;
        Ok(())
    }

    fn serialize_body<T: Write>(&self, writer: &mut T) -> io::Result<()> {
        if let Some(value) = self {
            value.serialize(writer)?;
        }
        Ok(())
    }

    fn load<T: Read>(reader: &mut T) -> io::Result<Self> {
        let size = usize::load(reader)?;
        if size == 0 {
            Ok(None)
        } else {
            let value = V::load(reader)?;
            // Here we could check that `value.size_in_elements() == size`. However, if
            // the value contains inner optional structures that were present in the
            // file but were skipped by `load`, the size of the in-memory structure is
            // too small. And because we do not require `io::Seek` from `T`, we cannot
            // check that we advanced by `size` elements in the reader.
            Ok(Some(value))
        }
    }

    fn size_in_elements(&self) -> usize {
        let mut result: usize = 1;
        if let Some(value) = self {
            result += value.size_in_elements();
        }
        result
    }
}

//-----------------------------------------------------------------------------

/// Modes of memory mapping a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MappingMode {
    /// The file is read-only.
    ReadOnly,
    /// Both read and write operations are supported.
    Mutable,
}

/// A memory-mapped file as an array of [`u64`].
///
/// This interface is highly unsafe.
/// The file remains open until the `MemoryMap` is dropped.
/// Memory-mapped structures should implement the [`MemoryMapped`] trait.
///
/// # Examples
///
/// ```
/// use simple_sds::serialize::{MemoryMap, MappingMode, Serialize};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let v: Vec<u64> = vec![123, 456];
/// let filename = serialize::temp_file_name("memory-map");
/// serialize::serialize_to(&v, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// assert_eq!(map.mode(), MappingMode::ReadOnly);
/// assert_eq!(map.len(), 3);
/// unsafe {
///     let slice: &[u64] = map.as_ref();
///     assert_eq!(slice[0], 2);
///     assert_eq!(slice[1], 123);
///     assert_eq!(slice[2], 456);
/// }
///
/// drop(map);
/// fs::remove_file(&filename).unwrap();
/// ```
#[derive(Debug)]
pub struct MemoryMap {
    _file: File, // The compiler might otherwise get overzealous and complain that we don't touch the file.
    filename: PathBuf,
    mode: MappingMode,
    ptr: *mut u64,
    len: usize,
}

// TODO: implement madvise()?
impl MemoryMap {
    /// Returns a memory map for the specified file in the given mode.
    ///
    /// # Arguments
    ///
    /// * `filename`: Name of the file.
    /// * `mode`: Memory mapping mode.
    ///
    /// # Errors
    ///
    /// The call may fail for a number of reasons, including:
    ///
    /// * File `filename` does not exist.
    /// * The file cannot be opened for writing with mode `MappingMode::Mutable`.
    /// * The size of the file is not a multiple of 8 bytes.
    /// * Memory mapping the file fails.
    pub fn new<P: AsRef<Path>>(filename: P, mode: MappingMode) -> io::Result<MemoryMap> {
        let write = match mode {
            MappingMode::ReadOnly => false,
            MappingMode::Mutable => true,
        };
        let mut options = OpenOptions::new();
        let file = options.read(true).write(write).open(&filename)?;

        let metadata = file.metadata()?;
        let len = metadata.len() as usize;
        if len != bits::round_up_to_word_bytes(len) {
            return Err(Error::new(ErrorKind::Other, "File size must be a multiple of 8 bytes"));
        }

        let prot = match mode {
            MappingMode::ReadOnly => libc::PROT_READ,
            MappingMode::Mutable => libc::PROT_READ | libc::PROT_WRITE,
        };
        let ptr = unsafe { libc::mmap(ptr::null_mut(), len, prot, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr.is_null() {
            return Err(Error::new(ErrorKind::Other, "Memory mapping failed"));
        }

        let mut buf = PathBuf::new();
        buf.push(&filename);
        Ok(MemoryMap {
            _file: file,
            filename: buf,
            mode,
            ptr: ptr.cast::<u64>(),
            len: bits::bytes_to_words(len),
        })
    }

    /// Returns the name of the memory mapped file.
    pub fn filename(&self) -> &Path {
        self.filename.as_path()
    }

    /// Returns the memory mapping mode for the file.
    #[inline]
    pub fn mode(&self) -> MappingMode {
        self.mode
    }

    /// Returns a mutable slice corresponding to the file.
    ///
    /// # Safety
    ///
    /// Behavior is undefined if the file was opened with mode `MappingMode::ReadOnly`.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u64] {
        slice::from_raw_parts_mut(self.ptr, self.len)
    }

    /// Returns the length of the memory-mapped file.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the file is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsRef<[u64]> for MemoryMap {
    fn as_ref(&self) -> &[u64] {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MemoryMap {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::munmap(self.ptr.cast::<libc::c_void>(), self.len);
        }
    }
}

//-----------------------------------------------------------------------------

/// A memory-mapped structure that borrows an interval of a memory map.
///
/// # Example
///
/// ```
/// use simple_sds::serialize::{MappingMode, MemoryMap, MemoryMapped, Serialize};
/// use simple_sds::serialize;
/// use std::io::{Error, ErrorKind};
/// use std::{fs, io, slice};
///
/// // This can read a serialized `Vec<u64>`.
/// #[derive(Debug)]
/// struct Example<'a> {
///     data: &'a [u64],
///     offset: usize,
/// }
///
/// impl<'a> Example<'a> {
///     pub fn as_slice(&self) -> &[u64] {
///         self.data
///     }
/// }
///
/// impl<'a> MemoryMapped<'a> for Example<'a> {
///     fn new(map: &'a MemoryMap, offset: usize) -> io::Result<Self> {
///         if offset >= map.len() {
///             return Err(Error::new(ErrorKind::UnexpectedEof, "The starting offset is out of range"));
///         }
///         let slice: &[u64] = map.as_ref();
///         let len = slice[offset] as usize;
///         if offset + 1 + len > map.len() {
///             return Err(Error::new(ErrorKind::UnexpectedEof, "The file is too short"));
///         }
///         Ok(Example {
///             data: &slice[offset + 1 .. offset + 1 + len],
///             offset: offset,
///         })
///     }
///
///     fn map_offset(&self) -> usize {
///         self.offset
///     }
///
///     fn map_len(&self) -> usize {
///         self.data.len() + 1
///     }
/// }
///
/// let v: Vec<u64> = vec![123, 456, 789];
/// let filename = serialize::temp_file_name("memory-mapped");
/// serialize::serialize_to(&v, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// let mapped = Example::new(&map, 0).unwrap();
/// assert_eq!(mapped.map_offset(), 0);
/// assert_eq!(mapped.map_len(), v.len() + 1);
/// assert_eq!(mapped.as_slice(), v.as_slice());
/// drop(mapped); drop(map);
///
/// fs::remove_file(&filename).unwrap();
/// ```
pub trait MemoryMapped<'a>: Sized {
    /// Returns an immutable memory-mapped structure corresponding to an interval in the file.
    ///
    /// # Arguments
    ///
    /// * `map`: Memory-mapped file.
    /// * `offset`: Starting offset in the file.
    ///
    /// # Errors
    ///
    /// Implementing types should use [`ErrorKind::UnexpectedEof`] where appropriate.
    /// [`ErrorKind::InvalidData`] should be used to indicate that the data failed sanity checks.
    fn new(map: &'a MemoryMap, offset: usize) -> io::Result<Self>;

    /// Returns the starting offset in the file.
    fn map_offset(&self) -> usize;

    /// Returns the length of the interval corresponding to the structure.
    fn map_len(&self) -> usize;
}

//-----------------------------------------------------------------------------

/// An immutable memory-mapped slice of a type that implements [`Serializable`].
///
/// The slice is compatible with the serialization format of [`Vec`] of the same type.
///
/// # Examples
///
/// ```
/// use simple_sds::serialize::{MappedSlice, MappingMode, MemoryMap, MemoryMapped, Serialize};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let v: Vec<(u64, u64)> = vec![(123, 456), (789, 101112)];
/// let filename = serialize::temp_file_name("mapped-slice");
/// serialize::serialize_to(&v, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// let mapped = MappedSlice::<(u64, u64)>::new(&map, 0).unwrap();
/// assert_eq!(mapped.len(), v.len());
/// assert_eq!(mapped[0], (123, 456));
/// assert_eq!(mapped[1], (789, 101112));
/// assert_eq!(*mapped, *v);
/// drop(mapped); drop(map);
///
/// fs::remove_file(&filename).unwrap();
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct MappedSlice<'a, T: Serializable> {
    data: &'a [T],
    offset: usize,
}

impl<'a, T: Serializable> MappedSlice<'a, T> {
    /// Returns the length of the slice.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T: Serializable> AsRef<[T]> for MappedSlice<'a, T> {
    fn as_ref(&self) -> &[T] {
        self.data
    }
}

impl<'a, T: Serializable> Deref for MappedSlice<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T: Serializable> Index<usize> for MappedSlice<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T: Serializable> MemoryMapped<'a> for MappedSlice<'a, T> {
    fn new(map: &'a MemoryMap, offset: usize) -> io::Result<Self> {
        if offset >= map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The starting offset is out of range"));
        }
        let slice: &[u64] = map.as_ref();
        let len = slice[offset] as usize;
        if offset + 1 + len * T::elements() > map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The file is too short"));
        }
        let source: &[u64] = &slice[offset + 1 ..];
        let data: &[T] = unsafe { slice::from_raw_parts(source.as_ptr() as *const T, len) };
        Ok(MappedSlice {
            data, offset,
        })
    }

    fn map_offset(&self) -> usize {
        self.offset
    }

    fn map_len(&self) -> usize {
        self.len() * T::elements() + 1
    }
}

//-----------------------------------------------------------------------------

/// An immutable memory-mapped slice of [`u8`].
///
/// The slice is compatible with the serialization format of [`Vec`] of [`u8`].
///
/// # Examples
///
/// ```
/// use simple_sds::serialize::{MappedBytes, MappingMode, MemoryMap, MemoryMapped, Serialize};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let v: Vec<u8> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233];
/// let filename = serialize::temp_file_name("mapped-bytes");
/// serialize::serialize_to(&v, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// let mapped = MappedBytes::new(&map, 0).unwrap();
/// assert_eq!(mapped.len(), v.len());
/// assert_eq!(mapped[3], 3);
/// assert_eq!(mapped[6], 13);
/// assert_eq!(*mapped, *v);
/// drop(mapped); drop(map);
///
/// fs::remove_file(&filename).unwrap();
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct MappedBytes<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> MappedBytes<'a> {
    /// Returns the length of the slice.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> AsRef<[u8]> for MappedBytes<'a> {
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

impl<'a> Deref for MappedBytes<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a> Index<usize> for MappedBytes<'a> {
    type Output = u8;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a> MemoryMapped<'a> for MappedBytes<'a> {
    fn new(map: &'a MemoryMap, offset: usize) -> io::Result<Self> {
        if offset >= map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The starting offset is out of range"));
        }
        let slice: &[u64] = map.as_ref();
        let len = slice[offset] as usize;
        if offset + 1 + bits::bytes_to_words(len) > map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The file is too short"));
        }
        let source: &[u64] = &slice[offset + 1 ..];
        let data: &[u8] = unsafe { slice::from_raw_parts(source.as_ptr() as *const u8, len) };
        Ok(MappedBytes {
            data, offset,
        })
    }

    fn map_offset(&self) -> usize {
        self.offset
    }

    fn map_len(&self) -> usize {
        bits::bytes_to_words(self.len()) + 1
    }
}

//-----------------------------------------------------------------------------

/// An immutable memory-mapped string slice.
///
/// The slice is compatible with the serialization format of [`String`].
///
/// # Examples
///
/// ```
/// use simple_sds::serialize::{MappedStr, MappingMode, MemoryMap, MemoryMapped, Serialize};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let s = String::from("GATTACA");
/// let filename = serialize::temp_file_name("mapped-str");
/// serialize::serialize_to(&s, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// let mapped = MappedStr::new(&map, 0).unwrap();
/// assert_eq!(mapped.len(), s.len());
/// assert_eq!(*mapped, *s);
/// drop(mapped); drop(map);
///
/// fs::remove_file(&filename).unwrap();
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct MappedStr<'a> {
    data: &'a str,
    offset: usize,
}

impl<'a> MappedStr<'a> {
    /// Returns the length of the slice in bytes.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> AsRef<str> for MappedStr<'a> {
    fn as_ref(&self) -> &str {
        self.data
    }
}

impl<'a> Deref for MappedStr<'a> {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}
impl<'a> MemoryMapped<'a> for MappedStr<'a> {
    fn new(map: &'a MemoryMap, offset: usize) -> io::Result<Self> {
        if offset >= map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The starting offset is out of range"));
        }
        let slice: &[u64] = map.as_ref();
        let len = slice[offset] as usize;
        if offset + 1 + bits::bytes_to_words(len) > map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The file is too short"));
        }
        let source: &[u64] = &slice[offset + 1 ..];
        let bytes: &[u8] = unsafe { slice::from_raw_parts(source.as_ptr() as *const u8, len) };
        let data = str::from_utf8(bytes).map_err(|_| Error::new(ErrorKind::InvalidData, "Invalid UTF-8"))?;
        Ok(MappedStr {
            data, offset,
        })
    }

    fn map_offset(&self) -> usize {
        self.offset
    }

    fn map_len(&self) -> usize {
        bits::bytes_to_words(self.len()) + 1
    }
}

//-----------------------------------------------------------------------------

/// An optional immutable memory-mapped structure.
///
/// This is compatible with the serialization format of [`Option`] of the same type.
///
/// # Examples
///
/// ```
/// use simple_sds::serialize::{MappedOption, MappedSlice, MappingMode, MemoryMap, MemoryMapped, Serialize};
/// use simple_sds::serialize;
/// use std::fs;
///
/// let some: Option<Vec<u64>> = Some(vec![123, 456, 789]);
/// let filename = serialize::temp_file_name("mapped-option");
/// serialize::serialize_to(&some, &filename);
///
/// let map = MemoryMap::new(&filename, MappingMode::ReadOnly).unwrap();
/// let mapped = MappedOption::<MappedSlice<u64>>::new(&map, 0).unwrap();
/// assert_eq!(mapped.unwrap().as_ref(), some.unwrap().as_slice());
/// drop(mapped); drop(map);
///
/// fs::remove_file(&filename).unwrap();
/// ```
#[derive(PartialEq, Eq, Debug)]
pub struct MappedOption<'a, T: MemoryMapped<'a>> {
    data: Option<T>,
    offset: usize,
    data_len: usize,
    _marker: marker::PhantomData<&'a ()>,
}

impl<'a, T: MemoryMapped<'a>> MappedOption<'a, T> {
    /// Returns `true` if the option is a [`Some`] value.
    pub fn is_some(&self) -> bool {
        self.data.is_some()
    }

    /// Returns `true` if the option is a [`None`] value.
    pub fn is_none(&self) -> bool {
        self.data.is_none()
    }

    /// Returns an immutable reference to the possibly contained value.
    ///
    /// # Panics
    ///
    /// Panics if the option is a [`None`] value.
    pub fn unwrap(&self) -> &T {
        match &self.data {
            Some(value) => value,
            None => panic!("MappedOption::unwrap(): No value to unwrap"),
        }
    }

    /// Returns [`Option`]`<&T>` referencing the possibly contained value.
    pub fn as_ref(&self) -> Option<&T> {
        match &self.data {
            Some(value) => Some(value),
            None => None,
        }
    }
}

impl<'a, T: MemoryMapped<'a>> MemoryMapped<'a> for MappedOption<'a, T> {
    fn new(map: &'a MemoryMap, offset: usize) -> io::Result<Self> {
        if offset >= map.len() {
            return Err(Error::new(ErrorKind::UnexpectedEof, "The starting offset is out of range"));
        }
        let mut result = MappedOption {
            data: None,
            offset,
            data_len: map.as_ref()[offset] as usize,
            _marker: marker::PhantomData,
        };
        if result.data_len > 0 {
            let value = T::new(map, offset + 1)?;
            result.data = Some(value)
        }
        Ok(result)
    }

    fn map_offset(&self) -> usize {
        self.offset
    }

    fn map_len(&self) -> usize {
        self.data_len + 1
    }
}

//-----------------------------------------------------------------------------

/// Serializes the item to the specified file, creating or overwriting the file if necessary.
///
/// See [`Serialize`] for an example.
///
/// # Errors
///
/// Any errors from [`OpenOptions::open`] and [`Serialize::serialize`] will be passed through.
pub fn serialize_to<T: Serialize, P: AsRef<Path>>(item: &T, filename: P) -> io::Result<()> {
    let mut options = OpenOptions::new();
    let mut file = options.create(true).write(true).truncate(true).open(filename)?;
    item.serialize(&mut file)?;
    Ok(())
}

/// Loads the item from the specified file.
///
/// See [`Serialize`] for an example.
///
/// # Errors
///
/// Any errors from [`OpenOptions::open`] and [`Serialize::load`] will be passed through.
pub fn load_from<T: Serialize, P: AsRef<Path>>(filename: P) -> io::Result<T> {
    let mut options = OpenOptions::new();
    let mut file = options.read(true).open(filename)?;
    <T as Serialize>::load(&mut file)
}

/// Serializes an absent optional structure of any type.
///
/// # Errors
///
/// Any errors from the writer will be passed through.
pub fn absent_option<T: Write>(writer: &mut T) -> io::Result<()> {
    let size: usize = 0;
    size.serialize(writer)?;
    Ok(())
}

/// Skips a serialized optional structure.
///
/// # Errors
///
/// Any errors from the reader will be passed through.
pub fn skip_option<T: Read>(reader: &mut T) -> io::Result<()> {
    let elements = usize::load(reader)?;
    if elements > 0 {
        io::copy(&mut reader.by_ref().take((elements * bits::WORD_BYTES) as u64), &mut io::sink())?;
    }
    Ok(())
}

/// Returns the size of an absent optional structure (of any type) in elements.
pub fn absent_option_size() -> usize {
    1
}

// Counter used for temporary file names.
static TEMP_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Returns a name for a temporary file using the provided name part.
///
/// # Examples
///
/// ```
/// use simple_sds::serialize;
///
/// let filename = serialize::temp_file_name("example");
/// assert!(filename.into_os_string().into_string().unwrap().contains("example"));
/// ```
pub fn temp_file_name(name_part: &str) -> PathBuf {
    let count = TEMP_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let mut buf = env::temp_dir();
    buf.push(format!("{}_{}_{}", name_part, process::id(), count));
    buf
}

/// Tests that the [`Serialize`] implementation works correctly.
///
/// Returns the name of the temporary file if `remove == false` and removes the file if `remove == true`.
/// The type must also implement [`PartialEq`] and [`Debug`] for the tests.
///
/// # Arguments
///
/// * `original`: Structure to be serialized.
/// * `name`: Name of the structure (for temporary file names and error messages).
/// * `expected_size`: Expected size in elements, or [`None`] if not known.
/// * `remove`: Should the temporary file be removed instead of returning its name.
///
/// # Examples
///
/// ```
/// use simple_sds::serialize;;
///
/// let v: Vec<u64> = vec![1, 11, 111, 1111];
/// let _ = serialize::test(&v, "vec-u64", Some(1 + v.len()), true);
/// ```
///
/// # Panics
///
/// Will panic if any of the tests fails.
pub fn test<T: Serialize + PartialEq + Debug>(original: &T, name: &str, expected_size: Option<usize>, remove: bool) -> Option<PathBuf> {
    if let Some(value) = expected_size {
        assert_eq!(original.size_in_elements(), value, "Size estimate for the serialized {} is not as expected", name);
    }

    let filename = temp_file_name(name);
    serialize_to(original, &filename).unwrap();

    let metadata = fs::metadata(&filename).unwrap();
    let len = metadata.len() as usize;
    assert_eq!(original.size_in_bytes(), len, "Invalid size estimate for the serialized {}", name);

    let copy: T = load_from(&filename).unwrap();
    assert_eq!(copy, *original, "Serialization changed the {}", name);

    if remove {
        fs::remove_file(&filename).unwrap();
        None
    } else {
        Some(filename)
    }
}
//-----------------------------------------------------------------------------
