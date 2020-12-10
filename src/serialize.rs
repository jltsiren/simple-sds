//! Simple serialization interface.
//!
//! The serialized representation closely mirrors the in-memory representation with 8-byte alignment.
//! This makes it easy to develop memory-mapped versions of the structures.
//!
//! The serialization format of a structure is split into the header and the data.
//! They can be serialized separately with [`Serialize::serialize_header`] and [`Serialize::serialize_data`].
//! Method [`Serialize::serialize`] provides an easy way of calling both.
//! A serialized structure is always loaded with a single [`Serialize::load`] call.
//!
//! # Wrapper structures
//!
//! Assume that we have wrapper structure `A` around `B`, which is in turn a wrapper structure around `C`.
//! The serialization format of `A` should be the following:
//!
//! * Header of `A`.
//!   * Header information.
//!   * Header of `B`.
//!     * Header information.
//!     * Header of `C`.
//! * Data in `C`.
//!
//! The header of the outer structure should always end with the header of the inner structure.
//! If we want to generate `A` directly to a file, we can then start by writing a placeholder header of `A`.
//! After we have finished writing the data, we go back to the beginning and write the true header.
//!
//! # Composite structures
//!
//! Assume that structure `A` contains `B` and `C`.
//! The serialization format of `A` should be the following:
//!
//! * Header of `A`.
//! * Structure `B`.
//!   * Header of `B`.
//!   * Data in `B`.
//! * Structure `C`.
//!   * Header of `C`.
//!   * Data in `C`.
//!
//! In this case, each structure is responsible for its own header.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{env, io, mem, process, slice};

//-----------------------------------------------------------------------------

/// Serialize a data structure.
///
/// `self.size_in_bytes()` should always be nonzero.
///
/// # Examples
///
/// ```
/// use simple_sds::serialize;
/// use simple_sds::serialize::Serialize;
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
///     fn serialize_data<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
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
///     fn size_in_bytes(&self) -> usize {
///         mem::size_of::<Self>()
///     }
/// }
///
/// let example = Example(-123, 456);
/// assert_eq!(example.size_in_bytes(), 8);
///
/// serialize::serialize_to(&example, "example.dat").unwrap();
///
/// let copy: Example = serialize::load_from("example.dat").unwrap();
/// assert_eq!(copy, example);
///
/// fs::remove_file("example.dat").unwrap();
/// ```
pub trait Serialize: Sized {
    /// Serializes the struct to the writer.
    ///
    /// Equivalent to calling [`Serialize::serialize_header`] and [`Serialize::serialize_data`].
    ///
    /// # Errors
    ///
    /// Any errors from the writer may be passed through.
    fn serialize<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        self.serialize_header(writer)?;
        self.serialize_data(writer)?;
        Ok(())
    }

    /// Serializes the header to the writer.
    ///
    /// # Errors
    ///
    /// Any errors from the writer may be passed through.
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()>;

    /// Serializes the data to the writer.
    ///
    /// # Errors
    ///
    /// Any errors from the writer may be passed through.
    fn serialize_data<T: io::Write>(&self, writer: &mut T) -> io::Result<()>;

    /// Loads the struct from the reader.
    ///
    /// # Errors
    ///
    /// Any errors from the reader may be passed through.
    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self>;

    /// Returns the size of the serialized struct in bytes.
    /// This should be closely related to the size of the in-memory struct.
    fn size_in_bytes(&self) -> usize;
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

//-----------------------------------------------------------------------------

macro_rules! serialize_int {
    ($t:ident) => {
        impl Serialize for $t {
            fn serialize_header<T: io::Write>(&self, _: &mut T) -> io::Result<()> {
                Ok(())
            }

            fn serialize_data<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
                let bytes = self.to_ne_bytes();
                writer.write_all(&bytes)?;
                Ok(())
            }

            fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
                let mut bytes = [0u8; mem::size_of::<Self>()];
                reader.read_exact(&mut bytes)?;
                let value = Self::from_ne_bytes(bytes);
                Ok(value)
            }

            fn size_in_bytes(&self) -> usize {
                mem::size_of::<Self>()
            }
        }
    }
}

serialize_int!(usize);

//-----------------------------------------------------------------------------

macro_rules! serialize_int_vec {
    ($t:ident) => {
        impl Serialize for Vec<$t> {
            fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
                let size = self.len();
                size.serialize(writer)?;
                Ok(())
            }

            fn serialize_data<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
                let source_slice: &[$t] = self;
                let target_slice: &[u8] = unsafe {
                    slice::from_raw_parts(source_slice.as_ptr() as *const u8, source_slice.len() * mem::size_of::<$t>())
                };
                writer.write_all(&target_slice)?;
                Ok(())
            }

            fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
                let size = usize::load(reader)?;

                let mut value: Vec<$t> = Vec::with_capacity(size);
                value.resize(size, 0);
                let target_slice: &mut [$t] = &mut value;
                let mut source_slice: &mut [u8] = unsafe {
                    slice::from_raw_parts_mut(target_slice.as_ptr() as *mut u8, target_slice.len() * mem::size_of::<$t>())
                };
                reader.read_exact(&mut source_slice)?;

                Ok(value)
            }

            fn size_in_bytes(&self) -> usize {
                mem::size_of::<usize>() + self.len() * mem::size_of::<$t>()
            }
        }
    }
}

serialize_int_vec!(u64);

//-----------------------------------------------------------------------------

impl<V: Serialize> Serialize for Option<V> {
    fn serialize_header<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        let mut size: usize = 0;
        if let Some(value) = self {
            size = value.size_in_bytes();
        }
        size.serialize(writer)?;
        Ok(())
    }

    fn serialize_data<T: io::Write>(&self, writer: &mut T) -> io::Result<()> {
        if let Some(value) = self {
            value.serialize(writer)?;
        }
        Ok(())
    }

    fn load<T: io::Read>(reader: &mut T) -> io::Result<Self> {
        let size = usize::load(reader)?;
        if size == 0 {
            Ok(None)
        } else {
            let value = V::load(reader)?;
            Ok(Some(value))
        }
    }

    fn size_in_bytes(&self) -> usize {
        let mut result = mem::size_of::<usize>();
        if let Some(value) = self {
            result += value.size_in_bytes();
        }
        result
    }
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn serialize_usize() {
        let filename = temp_file_name("usize");

        let original: usize = 0x1234_5678_9ABC_DEF0;
        assert_eq!(original.size_in_bytes(), 8, "Invalid serialized size for usize");
        serialize_to(&original, &filename).unwrap();

        let copy: usize = load_from(&filename).unwrap();
        assert_eq!(copy, original, "Serialization changed the value of usize");

        fs::remove_file(&filename).unwrap();
    }

    #[test]
    fn serialize_option() {
        let filename = temp_file_name("option");

        {
            let original: Option<usize> = None;
            assert_eq!(original.size_in_bytes(), 8, "Invalid serialized size for empty Option<usize>");
            serialize_to(&original, &filename).unwrap();
            let copy: Option<usize> = load_from(&filename).unwrap();
            assert_eq!(copy, original, "Serialization changed the value of empty Option<usize>");
        }

        {
            let original: Option<usize> = Some(123456);
            assert_eq!(original.size_in_bytes(), 16, "Invalid serialized size for non-empty Option<usize>");
            serialize_to(&original, &filename).unwrap();
            let copy: Option<usize> = load_from(&filename).unwrap();
            assert_eq!(copy, original, "Serialization changed the value of non-empty Option<usize>");
        }
    }

    #[test]
    fn serialize_vec_u64() {
        let filename = temp_file_name("vec_u64");

        let original: Vec<u64> = vec![1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89];
        assert_eq!(original.size_in_bytes(), 8 + 8 * original.len(), "Invalid serialized size for Vec<u64>");
        serialize_to(&original, &filename).unwrap();

        let copy: Vec<u64> = load_from(&filename).unwrap();
        assert_eq!(copy, original, "Serialization changed the value of Vec<u64>");

        fs::remove_file(&filename).unwrap();
    }
}

//-----------------------------------------------------------------------------
