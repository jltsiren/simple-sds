//! Low-level functions for bit manipulation.
//!
//! Most operations are built around storing bits in integer arrays of type `u64` using the least significant bits first.

use std::ops::{Index, IndexMut};

//-----------------------------------------------------------------------------

/// Number of bits in `u64`.
pub const WORD_BITS: usize = 64;

// Bit shift for transforming a bit offset into an array index.
const INDEX_SHIFT: usize = 6;

// Bit mask for transforming a bit offset into an offset in `u64`.
const OFFSET_MASK: usize = 0b111111;

/// Returns an integer with the lowest `n` bits set.
///
/// Behavior is undefined if `n > 64`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::low_set(13), 0x1FFF);
/// ```
pub fn low_set(n: usize) -> u64 {
    (0xFFFF_FFFF_FFFF_FFFFu128 >> (WORD_BITS - n)) as u64
}

/// Returns an integer with the highest `n` bits set.
///
/// Behavior is undefined if `n > 64`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::high_set(13), 0xFFF8_0000_0000_0000);
/// ```
pub fn high_set(n: usize) -> u64 {
    (0xFFFF_FFFF_FFFF_FFFFu128 << (WORD_BITS - n)) as u64
}

/// Returns the length of the binary representation of integer `n`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::bit_len(0), 1);
/// assert_eq!(bits::bit_len(0x1FFF), 13);
/// ```
pub fn bit_len(n: u64) -> usize {
    if n == 0 {
        1
    } else {
        WORD_BITS - (n.leading_zeros() as usize)
    }
}

/// Returns the number of bits that can be stored in `n` integers of type `u64`.
///
/// Behavior is undefined if `n * 64 > usize::MAX`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::words_to_bits(3), 192);
/// ```
pub fn words_to_bits(n: usize) -> usize {
    n * WORD_BITS
}

/// Returns the number of integers of type `u64` required to store `n` bits.
///
/// Behavior is undefined if `n + 63 > usize::MAX`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::bits_to_words(64), 1);
/// assert_eq!(bits::bits_to_words(65), 2);
/// ```
pub fn bits_to_words(n: usize) -> usize {
    (n + WORD_BITS - 1) / WORD_BITS
}

/// Rounds `n` up to the next positive multiple of 64.
///
/// Behavior is undefined if `n + 63 > usize::MAX`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::round_up_to_word_size(0), 64);
/// assert_eq!(bits::round_up_to_word_size(64), 64);
/// assert_eq!(bits::round_up_to_word_size(65), 128);
/// ```
pub fn round_up_to_word_size(n: usize) -> usize {
    if n == 0 {
        return WORD_BITS;
    }
    words_to_bits(bits_to_words(n))
}

/// Returns an `u64` value consisting entirely of bit `bit`.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::filler_value(false), 0);
/// assert_eq!(bits::filler_value(true), !0u64);
/// ```
pub fn filler_value(bit: bool) -> u64 {
    if bit { !0u64 } else { 0 }
}

/// Splits a bit offset into an index in an array of `u64` and an offset within the integer.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// assert_eq!(bits::split_offset(123), (1, 59));
/// ```
pub fn split_offset(bit_offset: usize) -> (usize, usize) {
    (bit_offset >> INDEX_SHIFT, bit_offset & OFFSET_MASK)
}

/// Writes an integer into a bit array implemented as an array of `u64` values.
///
/// Behavior is undefined if `width > 64`.
///
/// # Arguments
///
/// * `bit_offset`: Starting offset in the bit array.
/// * `value`: The integer to be written.
/// * `width`: The width of the integer in bits.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// let mut array: Vec<u64> = vec![0];
/// bits::write_int(&mut array, 0, 4, 7);
/// bits::write_int(&mut array, 4, 4, 3);
/// assert_eq!(array[0], 0x37);
/// ```
///
/// # Panics
///
/// Panics if `width > 64`.
/// May panic if `(bit_offset + width - 1) / 64` is not a valid index in the array.
pub fn write_int<T: IndexMut<usize, Output = u64>>(array: &mut T, bit_offset: usize, width: usize, value: u64) {

    let value = value & low_set(width);
    let (index, offset) = split_offset(bit_offset);

    if offset + width <= WORD_BITS {
        array[index] &= high_set(WORD_BITS - width - offset) | low_set(offset);
        array[index] |= value << offset;
    } else {
        array[index] &= low_set(offset);
        array[index] |= value << offset;
        array[index + 1] &= high_set(2 * WORD_BITS - width - offset);
        array[index + 1] |= value >> (WORD_BITS - offset);
    }
}

/// Reads an integer from an bit array implemented as an array of `u64` values.
///
/// Behavior is undefined if `width > 64`.
///
/// # Arguments
///
/// * `bit_offset`: Starting offset in the bit array.
/// * `width`: The width of the integer in bits.
///
/// # Examples
///
/// ```
/// use simple_sds::bits;
///
/// let array: Vec<u64> = vec![0x37];
/// assert_eq!(bits::read_int(&array, 0, 4), 7);
/// assert_eq!(bits::read_int(&array, 4, 4), 3);
/// ```
///
/// # Panics
///
/// May panic if `(bit_offset + width - 1) / 64` is not a valid index in the array.
pub fn read_int<T: Index<usize, Output = u64>>(array: &T, bit_offset: usize, width: usize) -> u64 {

    let (index, offset) = split_offset(bit_offset);
    let first = array[index] >> offset;

    if offset + width <= WORD_BITS {
        first & low_set(width)
    } else {
        first | ((array[index + 1] & low_set((offset + width) & OFFSET_MASK)) << (WORD_BITS - offset))
    }
}

//-----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;
    use rand::Rng;

    #[test]
    fn low_set_test() {
        assert_eq!(low_set(0), 0u64, "low_set(0) failed");
        assert_eq!(low_set(13), 0x1FFF, "low_set(13) failed");
        assert_eq!(low_set(64), !0u64, "low_set(64) failed")
    }

    #[test]
    fn high_set_test() {
        assert_eq!(high_set(0), 0, "high_set(0) failed");
        assert_eq!(high_set(13), 0xFFF8_0000_0000_0000, "high_set(13) failed");
        assert_eq!(high_set(64), !0u64, "high_set(64) failed")
    }

    #[test]
    fn bit_len_test() {
        assert_eq!(bit_len(0), 1, "bit_len(0) failed");
        assert_eq!(bit_len(1), 1, "bit_len(1) failed");
        assert_eq!(bit_len(0x1FFF), 13, "bit_len(0x1FFF) failed");
        assert_eq!(bit_len(0x7FFF_FFFF_FFFF_FFFF), 63, "bit_len(0x7FFF_FFFF_FFFF_FFFF) failed");
        assert_eq!(bit_len(0xFFFF_FFFF_FFFF_FFFF), 64, "bit_len(0xFFFF_FFFF_FFFF_FFFF) failed");
    }

    #[test]
    fn read_write_test() {
        let mut correct: Vec<(u64, u64, u64, u64)> = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..64 {
            let mut tuple: (u64, u64, u64, u64) = rng.gen();
            tuple.0 &= low_set(31); tuple.1 &= low_set(64); tuple.2 &= low_set(35); tuple.3 &= low_set(63);
            correct.push(tuple);
        }

        let mut array: Vec<u64> = vec![0; 256];
        let mut bit_offset = 0;
        for i in 0..64 {
            write_int(&mut array, bit_offset, 31, correct[i].0); bit_offset += 31;
            write_int(&mut array, bit_offset, 64, correct[i].1); bit_offset += 64;
            write_int(&mut array, bit_offset, 35, correct[i].2); bit_offset += 35;
            write_int(&mut array, bit_offset, 63, correct[i].3); bit_offset += 63;
        }

        bit_offset = 0;
        for i in 0..64 {
            assert_eq!(read_int(&array, bit_offset, 31), correct[i].0, "Invalid value at [{}].0", i); bit_offset += 31;
            assert_eq!(read_int(&array, bit_offset, 64), correct[i].1, "Invalid value at [{}].1", i); bit_offset += 64;
            assert_eq!(read_int(&array, bit_offset, 35), correct[i].2, "Invalid value at [{}].2", i); bit_offset += 35;
            assert_eq!(read_int(&array, bit_offset, 63), correct[i].3, "Invalid value at [{}].3", i); bit_offset += 63;
        }
    }

    #[test]
    fn no_extra_bits() {
        let mut array: Vec<u64> = vec![0; 2];
        write_int(&mut array, 16, 16, 2);
        write_int(&mut array, 48, 16, 2);
        write_int(&mut array, 32, 16, !0u64); // This should not overwrite the integer at offset 48.
        write_int(&mut array, 0, 16, !0u64); // This should not overwrite the other integers.
        assert_eq!(read_int(&array, 0, 16), 0xFFFF, "Incorrect 16-bit integer at offset 0");
        assert_eq!(read_int(&array, 16, 16), 2, "Incorrect 16-bit integer at offset 16");
        assert_eq!(read_int(&array, 32, 16), 0xFFFF, "Incorrect 16-bit integer at offset 32");
        assert_eq!(read_int(&array, 48, 16), 2, "Incorrect 16-bit integer at offset 48");
    }

    #[test]
    fn environment() {
        assert_eq!(mem::size_of::<usize>(), 8, "Things may not work if usize is not 64 bits");
//        assert_eq!(cfg!(target_endian), "little", "Only little-endian systems are supported");
    }
}

//-----------------------------------------------------------------------------
